"""Does DFlash2 acceptance rate actually dip at the reasoning->action boundary,
or does it only look that way because reasoning_vs_action_entropy.py's
boundary_curve only plots the entropy of ALREADY-ACCEPTED tokens (it drops
rejected draft positions before plotting)?

This script answers the question directly: for every DRAFTED position
(accepted or rejected), reconstruct its virtual position in the final output
sequence (using the running count of realized tokens, i.e. cumulative
accepted_len across time-ordered verification steps -- rejected extra
positions within a block don't advance this counter), compute its distance
from the reasoning->action boundary, and plot the raw acceptance rate
(not entropy) as a function of that distance.

Run from the project root: `uv run python analysis/acceptance_by_boundary_distance.py`
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

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


ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = Path(os.environ.get("JOB_DIR", str(ROOT / "jobs" / "muse-glimmer-swebench-50")))
ENTROPY_LOG = Path(os.environ.get("ENTROPY_LOG", str(ROOT / "entropy_log.jsonl")))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/mnt/data/muse-glimmer-30b")
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

BLUE = "#2a78d6"
RED = "#e34948"
TEXT = "#0b0b0b"
GRID = "#e4e2dc"


def n_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))


def load_turns() -> dict:
    """response_id -> n_reasoning_tokens"""
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
            reasoning_text, _ = split_reasoning_and_content(msg)
            turns[rid] = n_tokens(reasoning_text)
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


def main() -> None:
    print("Tokenizing reasoning_content lengths from trajectory files...")
    n_reasoning_by_rid = load_turns()
    print(f"  {len(n_reasoning_by_rid)} assistant turns")

    print("Loading entropy log...")
    by_req = load_entropy_by_reqid()
    print(f"  {len(by_req)} distinct req_ids in entropy log")

    response_ids = list(n_reasoning_by_rid.keys())

    rows = []
    matched_turns = 0
    for req_id, records in by_req.items():
        rid = next((r for r in response_ids if req_id.startswith(r)), None)
        if rid is None:
            continue
        n_reasoning = n_reasoning_by_rid[rid]

        by_t = defaultdict(list)
        for r in records:
            by_t[r["t"]].append(r)
        steps = sorted(by_t.items(), key=lambda kv: kv[0])

        counter = 0  # virtual position at start of this step, in the realized output
        turn_matched = False
        for _, step_records in steps:
            step_records.sort(key=lambda r: r["pos"])
            accepted_len = step_records[0]["accepted_len"]
            for r in step_records:
                virtual_pos = counter + r["pos"]
                dist = virtual_pos - n_reasoning
                rows.append(
                    {
                        "req_id": req_id,
                        "dist_from_boundary": dist,
                        "accepted": r["pos"] < accepted_len,
                        "entropy": r["entropy"],
                    }
                )
                turn_matched = True
            counter += accepted_len  # advance only by realized (accepted) tokens
        if turn_matched:
            matched_turns += 1

    df = pd.DataFrame(rows)
    print(f"\nMatched {matched_turns} turns, {len(df)} drafted-position records (accepted + rejected)")

    df["boundary_bucket"] = pd.cut(
        df["dist_from_boundary"], bins=list(range(-20, 21, 2)), include_lowest=True
    )
    curve = (
        df.groupby("boundary_bucket", observed=True)
        .agg(acceptance_rate=("accepted", "mean"), mean_entropy=("entropy", "mean"), n=("accepted", "count"))
        .reset_index()
    )
    curve["bucket_mid"] = curve["boundary_bucket"].apply(lambda b: (b.left + b.right) / 2)
    curve = curve[curve["n"] >= 50]

    print("\n=== Acceptance rate vs. distance from reasoning->action boundary (ALL drafted positions) ===")
    print(curve[["bucket_mid", "acceptance_rate", "mean_entropy", "n"]].to_string(index=False))

    out = {
        "n_turns_matched": matched_turns,
        "n_records": len(df),
        "curve": [
            {
                "dist": round(float(r["bucket_mid"]), 1),
                "acceptance_rate": round(float(r["acceptance_rate"]), 4),
                "mean_entropy": round(float(r["mean_entropy"]), 4),
                "n": int(r["n"]),
            }
            for _, r in curve.iterrows()
        ],
    }
    out_path = OUT_DIR / "acceptance_by_boundary_distance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    x = [c["dist"] for c in out["curve"]]
    ax1.plot(x, [c["acceptance_rate"] * 100 for c in out["curve"]], color=BLUE, marker="o", markersize=4, linewidth=2, label="acceptance rate")
    ax1.axvline(0, color=RED, linestyle="--", linewidth=1.5, label="reasoning -> action boundary")
    ax1.set_xlabel("Token distance from reasoning->action boundary (negative = still reasoning)")
    ax1.set_ylabel("Acceptance rate (%)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(x, [c["mean_entropy"] for c in out["curve"]], color=MUTED if (MUTED := "#82817b") else "#82817b", marker="s", markersize=3, linewidth=1.2, linestyle=":", label="mean entropy")
    ax2.set_ylabel("Mean entropy (nats)", color="#82817b")
    ax2.tick_params(axis="y", labelcolor="#82817b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title(
        "Does DFlash2 acceptance actually dip at the reasoning->action boundary?\n"
        "(ALL drafted positions, accepted + rejected -- not just realized tokens)",
        fontsize=11,
    )
    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acceptance_by_boundary_distance.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
