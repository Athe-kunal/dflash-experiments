"""Test: 'code and tool calls start with low entropy; reasoning is the
uncertain part' -- using ONLY already-collected data:

  - entropy_log.jsonl: per-position entropy + accepted_len, keyed by req_id
    (a vLLM chat-completion id, e.g. "chatcmpl-<id>-<suffix>")
  - jobs/<job>/*/agent/mini-swe-agent.trajectory.json: the full raw API
    response for every assistant turn, including reasoning_content, the
    tool-call / final content text, and ground-truth usage.completion_tokens.

Method: for each assistant turn, tokenize its reasoning_content and its
action text (content + tool-call JSON args) separately with the model's own
tokenizer to estimate how many of that turn's tokens were "reasoning" vs
"action". Reconstruct each entropy-log record's absolute position in the
turn (cumulative accepted_len across time-ordered verification steps, kept
only for records that were actually realized in the output) and label it
reasoning/action by comparing to the estimated reasoning-token count.

This is an approximation (channel-framing tokens like `to=self<|message|>`
aren't separately accounted for) but is a real test using existing data,
no re-run required.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = ROOT / "jobs" / "muse-glimmer-swebench-50"
ENTROPY_LOG = ROOT / "entropy_log.jsonl"
TOKENIZER_PATH = "/mnt/data/muse-glimmer-30b"

tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def n_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))


def load_turns() -> dict:
    """response_id -> {n_reasoning_tokens, n_action_tokens, n_total}"""
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
            reasoning_text = msg.get("reasoning_content") or ""
            content_text = msg.get("content") or ""
            action_text = content_text
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                action_text += (fn.get("name") or "") + (fn.get("arguments") or "")
            turns[rid] = {
                "n_reasoning_tokens": n_tokens(reasoning_text),
                "n_action_tokens": n_tokens(action_text),
                "n_total": int(total),
            }
    return turns


def load_entropy_by_reqid() -> dict:
    """req_id (from entropy_log) -> list of records"""
    by_req = defaultdict(list)
    with open(ENTROPY_LOG) as f:
        for line in f:
            rec = json.loads(line)
            if rec["req_id"].startswith("_warmup_"):
                continue
            by_req[rec["req_id"]].append(rec)
    return by_req


def main():
    print("Tokenizing turns from trajectory files...")
    turns = load_turns()
    print(f"  {len(turns)} assistant turns with usable usage info")

    print("Loading entropy log...")
    by_req = load_entropy_by_reqid()
    print(f"  {len(by_req)} distinct req_ids in entropy log")

    # match entropy req_ids to a turn's response id via prefix
    response_ids = list(turns.keys())
    response_ids.sort(key=len, reverse=True)  # longest first for safety, though ids are fixed-length

    rows = []
    matched_turns = 0
    for req_id, records in by_req.items():
        # req_id looks like "<response.id>-<suffix>"; response.id is the prefix
        rid = next((r for r in response_ids if req_id.startswith(r)), None)
        if rid is None:
            continue
        turn = turns[rid]
        n_reasoning = turn["n_reasoning_tokens"]

        # group records into verification steps by timestamp, order steps by time,
        # keep only realized (accepted) positions, assign cumulative absolute position
        by_t = defaultdict(list)
        for r in records:
            by_t[r["t"]].append(r)
        steps = sorted(by_t.items(), key=lambda kv: kv[0])

        abs_pos = 0
        matched_turns_flag = False
        for _, step_records in steps:
            step_records.sort(key=lambda r: r["pos"])
            accepted_len = step_records[0]["accepted_len"]
            for r in step_records:
                if r["pos"] >= accepted_len:
                    continue  # not realized in final output
                rows.append(
                    {
                        "req_id": req_id,
                        "abs_pos": abs_pos,
                        "entropy": r["entropy"],
                        "channel": "reasoning" if abs_pos < n_reasoning else "action",
                        "dist_from_boundary": abs_pos - n_reasoning,
                    }
                )
                abs_pos += 1
                matched_turns_flag = True
        if matched_turns_flag:
            matched_turns += 1

    df = pd.DataFrame(rows)
    print(f"\nMatched {matched_turns} turns, {len(df)} realized-token entropy records")

    print("\n=== Mean entropy by channel ===")
    print(df.groupby("channel")["entropy"].agg(["mean", "median", "std", "count"]))

    # first few tokens of the action channel vs mid-reasoning tokens
    first_action = df[(df["dist_from_boundary"] >= 0) & (df["dist_from_boundary"] < 3)]
    mid_reasoning = df[(df["channel"] == "reasoning") & (df["dist_from_boundary"] < -3)]
    print("\n=== First 3 tokens of action channel vs. mid-reasoning tokens ===")
    print("first_action_tokens: mean=%.4f n=%d" % (first_action["entropy"].mean(), len(first_action)))
    print("mid_reasoning_tokens: mean=%.4f n=%d" % (mid_reasoning["entropy"].mean(), len(mid_reasoning)))

    # entropy as a function of distance from the reasoning->action boundary
    df["boundary_bucket"] = pd.cut(
        df["dist_from_boundary"], bins=list(range(-20, 21, 2)), include_lowest=True
    )
    boundary_curve = (
        df.groupby("boundary_bucket", observed=True)["entropy"].agg(["mean", "count"]).reset_index()
    )
    boundary_curve["bucket_mid"] = boundary_curve["boundary_bucket"].apply(lambda b: (b.left + b.right) / 2)
    boundary_curve = boundary_curve[boundary_curve["count"] >= 20]

    out = {
        "n_turns_matched": matched_turns,
        "n_records": len(df),
        "mean_entropy_reasoning": float(df[df["channel"] == "reasoning"]["entropy"].mean()),
        "mean_entropy_action": float(df[df["channel"] == "action"]["entropy"].mean()),
        "mean_entropy_first3_action": float(first_action["entropy"].mean()),
        "mean_entropy_mid_reasoning": float(mid_reasoning["entropy"].mean()),
        "boundary_curve": [
            {"dist": round(float(r["bucket_mid"]), 1), "mean_entropy": round(float(r["mean"]), 4), "n": int(r["count"])}
            for _, r in boundary_curve.iterrows()
        ],
    }
    out_path = Path(__file__).resolve().parent / "reasoning_vs_action_entropy.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(
        [b["dist"] for b in out["boundary_curve"]],
        [b["mean_entropy"] for b in out["boundary_curve"]],
        color="#2a78d6",
        marker="o",
        markersize=4,
        linewidth=2,
    )
    ax.axvline(0, color="#e34948", linestyle="--", linewidth=1.5, label="reasoning -> action boundary")
    ax.set_xlabel("Token distance from reasoning->action boundary (negative = still reasoning)")
    ax.set_ylabel("Mean entropy (nats)")
    ax.set_title("Entropy around the reasoning -> action (tool call / code) transition")
    ax.legend()
    ax.grid(color="#e4e2dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "reasoning_vs_action_entropy.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
