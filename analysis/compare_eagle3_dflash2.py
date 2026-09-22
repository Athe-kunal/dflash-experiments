"""Combined (overlaid) Eagle3 vs. DFlash2 comparison plots.

Unlike the per-model plots in qwen3-eagle3/ and qwen3-dflash2/ (same chart,
two separate images), this script draws both drafters on the SAME axes for
each metric, so the comparison is visible in one glance. Reuses the already
-computed JSON summaries where possible; only the entropy-spread-vs
-acceptance comparison needs a quick recompute from the raw logs, since that
one was never exported to JSON by entropy_distribution_shape.py.

Run from the project root: `uv run python analysis/compare_eagle3_dflash2.py`
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EAGLE3_DIR = ROOT / "analysis" / "qwen3-eagle3"
DFLASH2_DIR = ROOT / "analysis" / "qwen3-dflash2"
EAGLE3_LOG = ROOT / "logs" / "entropy_log_qwen3_eagle3.jsonl"
DFLASH2_LOG = ROOT / "logs" / "entropy_log_qwen3_dflash2.jsonl"
OUT_DIR = ROOT / "analysis" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EAGLE3_COLOR = "#2a78d6"
DFLASH2_COLOR = "#e3762a"
TEXT = "#0b0b0b"
GRID = "#e4e2dc"

plt.rcParams["axes.edgecolor"] = "#999"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)


def load(d: Path, name: str) -> dict:
    return json.loads((d / name).read_text())


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, facecolor="white", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT_DIR / name}")


# ---------------------------------------------------------------- 1. hazard
def plot_hazard():
    e = load(EAGLE3_DIR, "entropy_distribution_shape.json")["hazard_curve"]
    d = load(DFLASH2_DIR, "entropy_distribution_shape.json")["hazard_curve"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["entropy"] for r in e], [r["p_accept"] * 100 for r in e],
            color=EAGLE3_COLOR, marker="o", markersize=3, linewidth=1.8, label="Eagle3 (AR)")
    ax.plot([r["entropy"] for r in d], [r["p_accept"] * 100 for r in d],
            color=DFLASH2_COLOR, marker="o", markersize=3, linewidth=1.8, label="DFlash2 (diffusion)")
    ax.set_xlim(0, 3.6)
    style(ax, "Hazard curve: P(accept) vs. entropy, Eagle3 vs. DFlash2", "Target-distribution entropy (nats)", "P(accept | entropy in bin) (%)")
    savefig(fig, "entropy_hazard_curve.png")


# -------------------------------------------------------------- 2. survival
def plot_survival():
    e = load(EAGLE3_DIR, "entropy_distribution_shape.json")["survival_curve"]
    d = load(DFLASH2_DIR, "entropy_distribution_shape.json")["survival_curve"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["threshold"] for r in e], [r["p_accept_given_ge"] * 100 for r in e],
            color=EAGLE3_COLOR, linewidth=2, label="Eagle3 (AR)")
    ax.plot([r["threshold"] for r in d], [r["p_accept_given_ge"] * 100 for r in d],
            color=DFLASH2_COLOR, linewidth=2, label="DFlash2 (diffusion)")
    ax.set_xlim(0, 3.6)
    style(ax, "Survival curve: P(accept | entropy >= x), Eagle3 vs. DFlash2", "Entropy threshold x (nats)", "P(accept | entropy >= x) (%)")
    savefig(fig, "entropy_survival_curve.png")


# ---------------------------------------------------------- 3. position decay
def plot_position():
    e = load(EAGLE3_DIR, "entropy_acceptance_by_position.json")
    d = load(DFLASH2_DIR, "entropy_acceptance_by_position.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([r["pos"] for r in e], [r["mean_entropy"] for r in e], color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot([r["pos"] for r in d], [r["mean_entropy"] for r in d], color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    style(ax, "Entropy by position within block", "Position within draft block", "Mean target entropy (nats)")

    ax = axes[1]
    ax.plot([r["pos"] for r in e], [r["acceptance_rate"] * 100 for r in e], color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot([r["pos"] for r in d], [r["acceptance_rate"] * 100 for r in d], color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    style(ax, "Acceptance rate by position within block", "Position within draft block", "Acceptance rate (%)")

    fig.suptitle("Confidence collapses within a block -- Eagle3 vs. DFlash2", fontsize=12, color=TEXT)
    savefig(fig, "entropy_acceptance_by_position.png")


# ------------------------------------------------- 4. entropy vs accepted_len
def plot_entropy_vs_acceptance():
    e = load(EAGLE3_DIR, "entropy_vs_acceptance_data.json")["buckets"]
    d = load(DFLASH2_DIR, "entropy_vs_acceptance_data.json")["buckets"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([b["entropy"] for b in e], [b["mean_accepted_len"] for b in e],
            color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot([b["entropy"] for b in d], [b["mean_accepted_len"] for b in d],
            color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    style(ax, "Entropy vs. mean accepted block length, Eagle3 vs. DFlash2", "Target-distribution entropy (nats)", "Mean accepted block length (tokens)")
    savefig(fig, "entropy_vs_acceptance.png")


# --------------------------------------------- 5. reasoning->action boundary
def plot_reasoning_boundary():
    e = load(EAGLE3_DIR, "reasoning_vs_action_entropy.json")["boundary_curve"]
    d = load(DFLASH2_DIR, "reasoning_vs_action_entropy.json")["boundary_curve"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["dist"] for r in e], [r["mean_entropy"] for r in e], color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot([r["dist"] for r in d], [r["mean_entropy"] for r in d], color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    ax.axvline(0, color="#999", linestyle="--", linewidth=1.2)
    style(ax, "Entropy around the reasoning->action boundary, Eagle3 vs. DFlash2",
          "Token distance from boundary (negative = still reasoning)", "Mean entropy (nats)")
    savefig(fig, "reasoning_vs_action_entropy.png")


# --------------------------------------------------- 6. token type breakdown
def plot_token_type():
    e = {b["category"]: b for b in load(EAGLE3_DIR, "entropy_by_token_type.json")["by_category"]}
    d = {b["category"]: b for b in load(DFLASH2_DIR, "entropy_by_token_type.json")["by_category"]}
    cats = [c for c in ["reasoning", "prose", "tool_structure", "short_value", "code_payload"] if c in e and c in d]
    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, [e[c]["mean_entropy"] for c in cats], width, color=EAGLE3_COLOR, label="Eagle3 (AR)")
    ax.bar(x + width / 2, [d[c]["mean_entropy"] for c in cats], width, color=DFLASH2_COLOR, label="DFlash2 (diffusion)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    style(ax, "Mean entropy by token type, Eagle3 vs. DFlash2", "", "Mean entropy (nats)")
    savefig(fig, "entropy_by_token_type.png")


# --------------------------------------------- 7. boundary acceptance (raw)
def plot_boundary_acceptance():
    e = load(EAGLE3_DIR, "acceptance_by_boundary_distance.json")["curve"]
    d = load(DFLASH2_DIR, "acceptance_by_boundary_distance.json")["curve"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["dist"] for r in e], [r["acceptance_rate"] * 100 for r in e], color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot([r["dist"] for r in d], [r["acceptance_rate"] * 100 for r in d], color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    ax.axvline(0, color="#999", linestyle="--", linewidth=1.2)
    style(ax, "Raw acceptance rate around the reasoning->action boundary", "Token distance from boundary", "Acceptance rate (%)")
    savefig(fig, "acceptance_by_boundary_distance.png")


# --------------------------------------------- 8. boundary acceptance (depth-controlled)
def plot_boundary_controlled():
    e = load(EAGLE3_DIR, "acceptance_by_boundary_controlled.json")["residual_curve"]
    d = load(DFLASH2_DIR, "acceptance_by_boundary_controlled.json")["residual_curve"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["dist"] for r in e], [r["mean_residual"] * 100 for r in e], color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot([r["dist"] for r in d], [r["mean_residual"] * 100 for r in d], color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    ax.axhline(0, color="#999", linewidth=1)
    ax.axvline(0, color="#999", linestyle="--", linewidth=1.2)
    style(ax, "Depth-controlled residual acceptance around the boundary", "Token distance from boundary", "Acceptance minus depth-baseline (pp)")
    savefig(fig, "acceptance_by_boundary_controlled.png")


# --------------------------------------------------- 9. context length
def plot_context_length():
    e = load(EAGLE3_DIR, "context_length_vs_acceptance.json")["buckets"]
    d = load(DFLASH2_DIR, "context_length_vs_acceptance.json")["buckets"]
    e_by_bucket = {b["bucket"]: b for b in e}
    d_by_bucket = {b["bucket"]: b for b in d}
    buckets = [b for b in e_by_bucket if b in d_by_bucket]  # keep only buckets both have enough data for
    x = np.arange(len(buckets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, [e_by_bucket[b]["mean_accepted_len"] for b in buckets], width, color=EAGLE3_COLOR, label="Eagle3 (AR)")
    ax.bar(x + width / 2, [d_by_bucket[b]["mean_accepted_len"] for b in buckets], width, color=DFLASH2_COLOR, label="DFlash2 (diffusion)")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=30, ha="right", fontsize=8)
    style(ax, "Context length vs. mean accepted length, Eagle3 vs. DFlash2", "Context length bucket (prompt tokens)", "Mean accepted block length (tokens)")
    savefig(fig, "context_length_vs_acceptance.png")


# --------------------------------------------- 10. entropy spread vs accepted_len (recomputed)
def load_block_level(log_path: Path) -> pd.DataFrame:
    rows = []
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            if r["req_id"].startswith("_warmup_"):
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    g = df.groupby(["req_id", "t"])
    out = g.agg(
        mean_entropy=("entropy", "mean"),
        std_entropy=("entropy", "std"),
        accepted_len=("accepted_len", "first"),
        block_size=("block_size", "first"),
    ).reset_index()
    out = out[out["block_size"] >= 3]
    out["std_entropy"] = out["std_entropy"].fillna(0.0)
    return out


def plot_variance():
    print("  loading raw logs for entropy-spread comparison (this is the slow step)...")
    e_df = load_block_level(EAGLE3_LOG)
    d_df = load_block_level(DFLASH2_LOG)

    def bucket_curve(df):
        edges = np.linspace(0, df["std_entropy"].quantile(0.995), 16)
        d = df.copy()
        d["bucket"] = pd.cut(d["std_entropy"], bins=edges, include_lowest=True)
        agg = d.groupby("bucket", observed=True)["accepted_len"].agg(["mean", "count"]).reset_index()
        agg["mid"] = agg["bucket"].apply(lambda b: (b.left + b.right) / 2)
        return agg[agg["count"] >= 20]

    e_curve = bucket_curve(e_df)
    d_curve = bucket_curve(d_df)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(e_curve["mid"], e_curve["mean"], color=EAGLE3_COLOR, marker="o", markersize=4, linewidth=2, label="Eagle3 (AR)")
    ax.plot(d_curve["mid"], d_curve["mean"], color=DFLASH2_COLOR, marker="o", markersize=4, linewidth=2, label="DFlash2 (diffusion)")
    style(ax, "Accepted length vs. within-block entropy spread, Eagle3 vs. DFlash2", "Block entropy std (nats)", "Mean accepted length (tokens)")
    savefig(fig, "entropy_variance_vs_acceptance.png")


def main():
    print("Plotting combined comparisons...")
    plot_hazard()
    plot_survival()
    plot_position()
    plot_entropy_vs_acceptance()
    plot_reasoning_boundary()
    plot_token_type()
    plot_boundary_acceptance()
    plot_boundary_controlled()
    plot_context_length()
    plot_variance()
    print(f"\nAll combined plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
