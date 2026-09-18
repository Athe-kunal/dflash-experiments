"""Depth-controlled version of acceptance_by_boundary_distance.py.

The raw acceptance-rate-vs-boundary-distance curve is confounded: acceptance
and entropy both depend heavily on a drafted position's depth *within its
own block* (position 0 accepts ~99%, position 10 accepts ~10%, established
in entropy_vs_acceptance.py). If the mix of block-depths sampled happens to
shift across the reasoning->action boundary window, the raw curve can show
an apparent dip that has nothing to do with the boundary itself.

This script controls for that: for every drafted position we know its
within-block position `pos`. We compute a baseline acceptance-rate-by-pos
curve from this same matched dataset, then for every record compute a
residual = actual_accepted - baseline_acceptance_rate[pos]. Bucketing that
residual by distance from the boundary answers: after removing the "what
depth were we drafting at" effect, is there still a boundary-specific dip?

Also plots the raw acceptance rate faceted by a few fixed `pos` values
directly, as a second, more intuitive way to see the same thing: if the dip
persists within a single fixed depth (e.g. always comparing pos=1 to pos=1
before/after the boundary), it's a real boundary effect, not depth-mixing.

Run from the project root: `uv run python analysis/acceptance_by_boundary_controlled.py`
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
MUTED = "#82817b"
GRID = "#e4e2dc"
FACET_POS = [0, 1, 2, 3, 4, 5]
FACET_COLORS = ["#2a78d6", "#5fb0d6", "#82817b", "#f0a336", "#e34948", "#9b59b6"]


def n_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))


def load_n_reasoning() -> dict:
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


def build_records(n_reasoning_by_rid: dict, by_req: dict) -> pd.DataFrame:
    response_ids = list(n_reasoning_by_rid.keys())
    rows = []
    for req_id, records in by_req.items():
        rid = next((r for r in response_ids if req_id.startswith(r)), None)
        if rid is None:
            continue
        n_reasoning = n_reasoning_by_rid[rid]

        by_t = defaultdict(list)
        for r in records:
            by_t[r["t"]].append(r)
        steps = sorted(by_t.items(), key=lambda kv: kv[0])

        counter = 0
        for _, step_records in steps:
            step_records.sort(key=lambda r: r["pos"])
            accepted_len = step_records[0]["accepted_len"]
            for r in step_records:
                virtual_pos = counter + r["pos"]
                rows.append(
                    {
                        "req_id": req_id,
                        "dist_from_boundary": virtual_pos - n_reasoning,
                        "pos": r["pos"],
                        "accepted": r["pos"] < accepted_len,
                        "entropy": r["entropy"],
                    }
                )
            counter += accepted_len
    return pd.DataFrame(rows)


def main() -> None:
    print("Tokenizing reasoning_content lengths from trajectory files...")
    n_reasoning_by_rid = load_n_reasoning()
    print(f"  {len(n_reasoning_by_rid)} assistant turns")

    print("Loading entropy log...")
    by_req = load_entropy_by_reqid()
    print(f"  {len(by_req)} distinct req_ids in entropy log")

    print("Building matched per-position records...")
    df = build_records(n_reasoning_by_rid, by_req)
    print(f"  {len(df):,} drafted-position records")

    # baseline acceptance-rate-by-depth, from THIS SAME matched dataset
    baseline = df.groupby("pos")["accepted"].mean()
    print("\n=== Baseline acceptance rate by within-block position (this dataset) ===")
    print(baseline.head(16))

    df["baseline_at_pos"] = df["pos"].map(baseline)
    df["residual"] = df["accepted"].astype(float) - df["baseline_at_pos"]

    df["boundary_bucket"] = pd.cut(
        df["dist_from_boundary"], bins=list(range(-20, 21, 2)), include_lowest=True
    )
    resid_curve = (
        df.groupby("boundary_bucket", observed=True)
        .agg(mean_residual=("residual", "mean"), mean_pos=("pos", "mean"), n=("residual", "count"))
        .reset_index()
    )
    resid_curve["bucket_mid"] = resid_curve["boundary_bucket"].apply(lambda b: (b.left + b.right) / 2)
    resid_curve = resid_curve[resid_curve["n"] >= 50]

    print("\n=== Depth-controlled residual acceptance vs. distance from boundary ===")
    print("(mean_residual = actual_accepted - baseline_rate_for_that_depth; 0 = exactly as expected for its depth)")
    print(resid_curve[["bucket_mid", "mean_residual", "mean_pos", "n"]].to_string(index=False))

    # faceted raw curves at a few fixed depths
    facet_rows = []
    for p in FACET_POS:
        sub = df[df["pos"] == p].copy()
        sub["boundary_bucket"] = pd.cut(
            sub["dist_from_boundary"], bins=list(range(-20, 21, 4)), include_lowest=True
        )
        agg = sub.groupby("boundary_bucket", observed=True)["accepted"].agg(["mean", "count"]).reset_index()
        agg["bucket_mid"] = agg["boundary_bucket"].apply(lambda b: (b.left + b.right) / 2)
        agg = agg[agg["count"] >= 20]
        for _, r in agg.iterrows():
            facet_rows.append({"pos": p, "dist": float(r["bucket_mid"]), "acceptance_rate": float(r["mean"]), "n": int(r["count"])})

    out = {
        "n_records": int(len(df)),
        "baseline_acceptance_by_pos": {int(k): round(float(v), 4) for k, v in baseline.head(20).items()},
        "residual_curve": [
            {
                "dist": round(float(r["bucket_mid"]), 1),
                "mean_residual": round(float(r["mean_residual"]), 4),
                "mean_pos": round(float(r["mean_pos"]), 2),
                "n": int(r["n"]),
            }
            for _, r in resid_curve.iterrows()
        ],
        "facet_curves": facet_rows,
    }
    out_path = OUT_DIR / "acceptance_by_boundary_controlled.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    ax = axes[0]
    x = [c["dist"] for c in out["residual_curve"]]
    y = [c["mean_residual"] * 100 for c in out["residual_curve"]]
    ax.plot(x, y, color=RED, marker="o", markersize=4, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="-")
    ax.axvline(0, color=BLUE, linestyle="--", linewidth=1.5, label="reasoning -> action boundary")
    ax.set_xlabel("Token distance from boundary")
    ax.set_ylabel("Acceptance rate minus depth-baseline (pp)")
    ax.set_title("Depth-controlled residual acceptance\n(0 = exactly what depth alone predicts)", fontsize=10)
    ax.legend()

    ax = axes[1]
    for p, color in zip(FACET_POS, FACET_COLORS):
        sub = [c for c in out["facet_curves"] if c["pos"] == p]
        if not sub:
            continue
        ax.plot(
            [c["dist"] for c in sub], [c["acceptance_rate"] * 100 for c in sub],
            color=color, marker="o", markersize=3, linewidth=1.5, label=f"pos={p}",
        )
    ax.axvline(0, color=RED, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Token distance from boundary")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Raw acceptance rate, faceted by fixed within-block depth", fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Is the boundary acceptance dip real, or a within-block-depth mixing artifact?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acceptance_by_boundary_controlled.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
