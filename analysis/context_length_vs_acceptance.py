"""Does DFlash2 acceptance length change as the agent's context (prompt)
length grows over a SWE-bench episode? Uses only already-collected data:
each turn's usage.prompt_tokens (context length at that point) from the
trajectory files, joined to that turn's mean accepted block length from
entropy_log.jsonl via the req_id prefix match.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = ROOT / "jobs" / "muse-glimmer-swebench-50"
ENTROPY_LOG = ROOT / "entropy_log.jsonl"


def load_turn_context_lengths() -> dict:
    """response_id -> prompt_tokens (context length at that turn)"""
    out = {}
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
            prompt_tokens = usage.get("prompt_tokens")
            if not rid or prompt_tokens is None:
                continue
            out[rid] = int(prompt_tokens)
    return out


def load_mean_accepted_len_per_reqid() -> dict:
    """entropy-log req_id -> mean accepted_len across its distinct blocks"""
    block_lens = defaultdict(list)
    seen_steps = defaultdict(set)
    with open(ENTROPY_LOG) as f:
        for line in f:
            rec = json.loads(line)
            if rec["req_id"].startswith("_warmup_"):
                continue
            key = (rec["req_id"], rec["t"])
            if key in seen_steps:
                continue
            seen_steps[key] = True
            block_lens[rec["req_id"]].append(rec["accepted_len"])
    return {rid: float(np.mean(v)) for rid, v in block_lens.items()}


def main():
    print("Loading turn context lengths from trajectory files...")
    context_lengths = load_turn_context_lengths()
    print(f"  {len(context_lengths)} turns with prompt_tokens")

    print("Loading mean accepted length per request from entropy log...")
    accepted_by_reqid = load_mean_accepted_len_per_reqid()
    print(f"  {len(accepted_by_reqid)} distinct req_ids")

    response_ids = list(context_lengths.keys())

    rows = []
    for req_id, mean_accepted in accepted_by_reqid.items():
        rid = next((r for r in response_ids if req_id.startswith(r)), None)
        if rid is None:
            continue
        rows.append({"req_id": req_id, "prompt_tokens": context_lengths[rid], "mean_accepted_len": mean_accepted})

    df = pd.DataFrame(rows)
    print(f"\nMatched {len(df)} turns")

    pearson = df["prompt_tokens"].corr(df["mean_accepted_len"])
    spearman = df["prompt_tokens"].corr(df["mean_accepted_len"], method="spearman")
    print(f"\nPearson r = {pearson:.4f}, Spearman rho = {spearman:.4f}")

    # bucket by prompt length
    edges = [0, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 200000]
    df["bucket"] = pd.cut(df["prompt_tokens"], bins=edges, include_lowest=True)
    agg = df.groupby("bucket", observed=True)["mean_accepted_len"].agg(["mean", "std", "count"]).reset_index()
    agg = agg[agg["count"] >= 5]
    print("\n=== Mean accepted length by context-length bucket ===")
    print(agg.to_string(index=False))

    out = {
        "n_turns": len(df),
        "pearson": round(float(pearson), 4),
        "spearman": round(float(spearman), 4),
        "buckets": [
            {
                "bucket": str(row["bucket"]),
                "mean_accepted_len": round(float(row["mean"]), 3),
                "std": round(float(row["std"]) if pd.notna(row["std"]) else 0.0, 3),
                "count": int(row["count"]),
            }
            for _, row in agg.iterrows()
        ],
    }
    out_path = Path(__file__).resolve().parent / "context_length_vs_acceptance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.scatter(df["prompt_tokens"], df["mean_accepted_len"], s=8, alpha=0.15, color="#2a78d6", linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("Context length at this turn (prompt tokens, log scale)")
    ax.set_ylabel("Mean accepted block length (tokens)")
    ax.set_title(
        f"Context length vs. DFlash2 acceptance length\n"
        f"n={len(df):,} turns (Pearson r={out['pearson']}, Spearman rho={out['spearman']})",
        fontsize=11,
    )
    ax.grid(color="#e4e2dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "context_length_vs_acceptance.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
