"""Entropy by fine-grained token type, not just reasoning vs. action.

Splits the "action" channel from reasoning_vs_action_entropy.py further into:
  - prose:          plain assistant message content (outside tool calls)
  - tool_structure: tool-call function names + JSON punctuation/keys
  - short_value:    short JSON string values (flags, paths, one-liners)
  - code_payload:   long / multiline JSON string values (commands with
                     heredocs, patches, file content -- the actual "doing"
                     text, as opposed to boilerplate)

Uses only already-collected data:
  - entropy_log.jsonl (per-position entropy + accepted, keyed by req_id)
  - jobs/<job>/*/agent/mini-swe-agent.trajectory.json (raw API responses)

Method: for each assistant turn, rebuild the exact reasoning_content +
action_text string (same construction as reasoning_vs_action_entropy.py) as
an ordered list of (category, text) segments, tokenize each segment with the
model's own tokenizer, and use cumulative token-count boundaries to label
each entropy-log record's reconstructed absolute position with a category.

Run from the project root: `uv run python analysis/entropy_by_token_type.py`
"""

import bisect
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = Path(os.environ.get("JOB_DIR", str(ROOT / "jobs" / "muse-glimmer-swebench-50")))
ENTROPY_LOG = Path(os.environ.get("ENTROPY_LOG", str(ROOT / "entropy_log.jsonl")))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/mnt/data/muse-glimmer-30b")
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

BLUE = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#82817b"
GRID = "#e4e2dc"

CATEGORY_ORDER = ["reasoning", "prose", "tool_structure", "short_value", "code_payload"]
CATEGORY_COLORS = {
    "reasoning": "#e34948",
    "prose": "#f0a336",
    "tool_structure": "#82817b",
    "short_value": "#5fb0d6",
    "code_payload": "#2a78d6",
}

_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_reasoning_and_content(msg: dict) -> tuple[str, str]:
    """Return (reasoning_text, content_text). Some models (e.g. muse-glimmer)
    put reasoning in a separate `reasoning_content` field; others (e.g.
    Qwen3) inline it as <think>...</think> inside `content`."""
    reasoning_content = msg.get("reasoning_content")
    if reasoning_content:
        return reasoning_content, msg.get("content") or ""
    content = msg.get("content") or ""
    matches = _THINK_RE.findall(content)
    if not matches:
        return "", content
    reasoning_text = "".join(matches)
    remainder = _THINK_RE.sub("", content)
    return reasoning_text, remainder


def n_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))


def classify_json_arguments(args_str: str) -> list[tuple[str, str]]:
    """Split a raw JSON-arguments string into (category, substring) segments,
    covering the whole string in order (so token counts sum consistently)."""
    if not args_str:
        return []
    segments = []
    pos = 0
    for m in _STRING_RE.finditer(args_str):
        start, end = m.span()
        if start > pos:
            segments.append(("tool_structure", args_str[pos:start]))
        s = m.group(0)
        j = end
        while j < len(args_str) and args_str[j] in " \t\n\r":
            j += 1
        is_key = j < len(args_str) and args_str[j] == ":"
        if is_key:
            segments.append(("tool_structure", s))
        else:
            inner = s[1:-1]
            if "\\n" in inner or len(inner) > 60:
                segments.append(("code_payload", s))
            else:
                segments.append(("short_value", s))
        pos = end
    if pos < len(args_str):
        segments.append(("tool_structure", args_str[pos:]))
    return segments


def build_segments(msg: dict) -> list[tuple[str, str]]:
    segments = []
    reasoning_text, content_text = split_reasoning_and_content(msg)
    if reasoning_text:
        segments.append(("reasoning", reasoning_text))
    if content_text:
        segments.append(("prose", content_text))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if name:
            segments.append(("tool_structure", name))
        segments.extend(classify_json_arguments(fn.get("arguments") or ""))
    return segments


def load_turns() -> dict:
    """response_id -> {"boundaries": [cum_token_count...], "categories": [...], "n_total": int}"""
    turns = {}
    for traj_path in JOB_DIR.glob("*/agent/mini-swe-agent.trajectory.json"):
        try:
            data = json.loads(traj_path.read_text())
        except Exception:
            continue
        for msg in data.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            resp = (msg.get("extra") or {}).get("response")
            if not resp:
                continue
            rid = resp.get("id")
            usage = resp.get("usage") or {}
            total = usage.get("completion_tokens")
            if not rid or not total:
                continue
            segments = build_segments(msg)
            categories = []
            boundaries = []
            cum = 0
            for cat, text in segments:
                nt = n_tokens(text)
                if nt == 0:
                    continue
                cum += nt
                categories.append(cat)
                boundaries.append(cum)
            turns[rid] = {"boundaries": boundaries, "categories": categories, "n_total": int(total)}
    return turns


def load_entropy_by_reqid() -> dict:
    by_req = defaultdict(list)
    with open(ENTROPY_LOG) as f:
        for line in f:
            rec = json.loads(line)
            if rec["req_id"].startswith("_warmup_"):
                continue
            by_req[rec["req_id"]].append(rec)
    return by_req


def category_for_pos(abs_pos: int, boundaries: list[int], categories: list[str]) -> str | None:
    if not boundaries:
        return None
    idx = bisect.bisect_right(boundaries, abs_pos)
    if idx >= len(categories):
        return None
    return categories[idx]


def main() -> None:
    print("Tokenizing turns from trajectory files into typed segments...")
    turns = load_turns()
    print(f"  {len(turns)} assistant turns with usable usage info")

    print("Loading entropy log...")
    by_req = load_entropy_by_reqid()
    print(f"  {len(by_req)} distinct req_ids in entropy log")

    response_ids = list(turns.keys())

    rows = []
    matched_turns = 0
    for req_id, records in by_req.items():
        rid = next((r for r in response_ids if req_id.startswith(r)), None)
        if rid is None:
            continue
        turn = turns[rid]
        boundaries, categories = turn["boundaries"], turn["categories"]

        by_t = defaultdict(list)
        for r in records:
            by_t[r["t"]].append(r)
        steps = sorted(by_t.items(), key=lambda kv: kv[0])

        abs_pos = 0
        turn_matched = False
        for _, step_records in steps:
            step_records.sort(key=lambda r: r["pos"])
            accepted_len = step_records[0]["accepted_len"]
            for r in step_records:
                if r["pos"] >= accepted_len:
                    continue
                cat = category_for_pos(abs_pos, boundaries, categories)
                if cat is not None:
                    rows.append({"req_id": req_id, "abs_pos": abs_pos, "entropy": r["entropy"], "category": cat})
                    turn_matched = True
                abs_pos += 1
        if turn_matched:
            matched_turns += 1

    df = pd.DataFrame(rows)
    print(f"\nMatched {matched_turns} turns, {len(df)} categorized entropy records")

    print("\n=== Mean entropy by token type ===")
    summary_df = df.groupby("category")["entropy"].agg(["mean", "median", "std", "count"])
    summary_df = summary_df.reindex([c for c in CATEGORY_ORDER if c in summary_df.index])
    print(summary_df)

    out = {
        "n_turns_matched": matched_turns,
        "n_records": len(df),
        "by_category": [
            {
                "category": cat,
                "mean_entropy": round(float(row["mean"]), 4),
                "median_entropy": round(float(row["median"]), 4),
                "std_entropy": round(float(row["std"]), 4) if pd.notna(row["std"]) else 0.0,
                "count": int(row["count"]),
            }
            for cat, row in summary_df.iterrows()
        ],
    }

    # within tool-call args specifically: code_payload vs short_value gap
    args_df = df[df["category"].isin(["short_value", "code_payload"])]
    if len(args_df):
        gap = (
            args_df.groupby("category")["entropy"].mean().to_dict()
        )
        out["code_payload_vs_short_value_gap"] = {
            "code_payload_mean": round(float(gap.get("code_payload", float("nan"))), 4),
            "short_value_mean": round(float(gap.get("short_value", float("nan"))), 4),
        }

    out_path = OUT_DIR / "entropy_by_token_type.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cats = [b["category"] for b in out["by_category"]]
    means = [b["mean_entropy"] for b in out["by_category"]]
    counts = [b["count"] for b in out["by_category"]]
    stds = [b["std_entropy"] for b in out["by_category"]]
    yerr = [s / np.sqrt(c) for s, c in zip(stds, counts)]
    colors = [CATEGORY_COLORS.get(c, BLUE) for c in cats]

    ax.bar(cats, means, color=colors, edgecolor="none")
    ax.errorbar(cats, means, yerr=yerr, fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=3)
    for i, (c, n) in enumerate(zip(cats, counts)):
        ax.text(i, means[i] + 0.01, f"n={n:,}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_ylabel("Mean target-distribution entropy (nats)", color=TEXT)
    ax.set_title(
        f"Entropy by fine-grained token type\n"
        f"n={out['n_records']:,} records, {out['n_turns_matched']} turns",
        color=TEXT,
        fontsize=11,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "entropy_by_token_type.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
