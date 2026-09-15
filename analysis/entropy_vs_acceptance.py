"""Entropy vs. DFlash2 speculative-decoding acceptance-length analysis.

Reads entropy_log.jsonl (written by muse_glimmer.entropy_plugin during
vLLM serving) and produces:
  - analysis/entropy_vs_acceptance_data.json  (binned summary stats)
  - analysis/entropy_vs_acceptance.png        (binned mean bar chart)
  - analysis/entropy_vs_acceptance_scatter.png (raw per-block scatter)

Run from the project root: `uv run python analysis/entropy_vs_acceptance.py`
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "entropy_log.jsonl"
OUT_DIR = Path(__file__).resolve().parent

BLUE = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#82817b"
GRID = "#e4e2dc"


def load_all() -> pd.DataFrame:
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df[~df["req_id"].str.startswith("_warmup_")]


def load_blocks(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pos"] == 0][["req_id", "entropy", "accepted_len"]].drop_duplicates()


def by_position(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("pos")
        .agg(mean_entropy=("entropy", "mean"), acceptance_rate=("accepted", "mean"), n=("entropy", "count"))
        .reset_index()
    )


def summarize(block_df: pd.DataFrame) -> dict:
    max_e = block_df["entropy"].quantile(0.995)
    edges = np.linspace(0, max_e, 21)
    block_df = block_df.copy()
    block_df["bucket"] = pd.cut(block_df["entropy"], bins=edges, include_lowest=True)
    agg = (
        block_df.groupby("bucket", observed=True)["accepted_len"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    agg["bucket_mid"] = agg["bucket"].apply(lambda b: (b.left + b.right) / 2)
    agg = agg[agg["count"] >= 3]

    buckets = []
    for _, row in agg.iterrows():
        std = row["std"]
        buckets.append(
            {
                "entropy": round(float(row["bucket_mid"]), 3),
                "mean_accepted_len": round(float(row["mean"]), 3),
                "count": int(row["count"]),
                "std": round(float(std) if pd.notna(std) else 0.0, 3),
            }
        )

    return {
        "n_blocks": int(len(block_df)),
        "n_requests": int(block_df["req_id"].nunique()),
        "pearson": round(float(block_df["entropy"].corr(block_df["accepted_len"])), 4),
        "spearman": round(
            float(block_df["entropy"].corr(block_df["accepted_len"], method="spearman")), 4
        ),
        "buckets": buckets,
    }


def plot_bars(summary: dict, out_path: Path) -> None:
    buckets = summary["buckets"]
    x = [b["entropy"] for b in buckets]
    y = [b["mean_accepted_len"] for b in buckets]
    yerr = [b["std"] / np.sqrt(b["count"]) for b in buckets]
    width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 0.1

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x, y, width=width, color=BLUE, edgecolor="none")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=0)
    ax.set_xlabel("Target-distribution entropy (nats)", color=TEXT)
    ax.set_ylabel("Mean accepted block length (tokens)", color=TEXT)
    ax.set_title(
        f"Entropy vs. DFlash2 acceptance length\n"
        f"n={summary['n_blocks']:,} blocks, {summary['n_requests']} requests "
        f"(Pearson r={summary['pearson']}, Spearman ρ={summary['spearman']})",
        color=TEXT,
        fontsize=11,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_scatter(block_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.scatter(
        block_df["entropy"], block_df["accepted_len"], s=6, alpha=0.15, color=BLUE, linewidths=0
    )
    ax.set_xlabel("Target-distribution entropy (nats)", color=TEXT)
    ax.set_ylabel("Accepted block length (tokens)", color=TEXT)
    ax.set_title("Entropy vs. acceptance length — raw per-block scatter", color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_by_position(pos_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax = axes[0]
    ax.plot(pos_df["pos"], pos_df["mean_entropy"], color=BLUE, marker="o", markersize=4, linewidth=2)
    ax.set_xlabel("Position within draft block", color=TEXT)
    ax.set_ylabel("Mean target entropy (nats)", color=TEXT)
    ax.set_title("Entropy grows with draft depth", color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(
        pos_df["pos"], pos_df["acceptance_rate"] * 100, color=BLUE, marker="o", markersize=4, linewidth=2
    )
    ax.set_xlabel("Position within draft block", color=TEXT)
    ax.set_ylabel("Acceptance rate (%)", color=TEXT)
    ax.set_title("...while acceptance rate collapses", color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Why acceptance length is finite: uncertainty compounds within a block", color=TEXT, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    df = load_all()
    block_df = load_blocks(df)
    summary = summarize(block_df)
    pos_df = by_position(df)

    with open(OUT_DIR / "entropy_vs_acceptance_data.json", "w") as f:
        json.dump(summary, f, indent=2)
    pos_df.to_json(OUT_DIR / "entropy_acceptance_by_position.json", orient="records", indent=2)

    plot_bars(summary, OUT_DIR / "entropy_vs_acceptance.png")
    plot_scatter(block_df, OUT_DIR / "entropy_vs_acceptance_scatter.png")
    plot_by_position(pos_df, OUT_DIR / "entropy_acceptance_by_position.png")

    print(json.dumps({k: v for k, v in summary.items() if k != "buckets"}, indent=2))
    print(pos_df.to_string(index=False))
    print(f"Saved plots and data to {OUT_DIR}")


if __name__ == "__main__":
    main()
