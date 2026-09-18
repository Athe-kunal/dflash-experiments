"""Distribution-shape analysis of entropy vs. acceptance, beyond means.

entropy_vs_acceptance.py only reports mean accepted length per entropy
bucket and mean entropy per block position -- both average away shape. This
script asks two sharper questions of the same entropy_log.jsonl:

  1. Hazard / survival curves at the level of individual drafted positions
     (not blocks): P(accept | entropy in bin) and the cumulative
     P(accept | entropy >= x), to look for a threshold rather than a smooth
     linear decline.
  2. Within a draft block, does the *spread* (std) of entropy across
     positions predict accepted length, beyond what the block's mean
     entropy already predicts? Computed via a simple partial-correlation
     (linear-residual) check.

Run from the project root: `uv run python analysis/entropy_distribution_shape.py`
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(os.environ.get("ENTROPY_LOG", str(ROOT / "entropy_log.jsonl")))
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
RED = "#e34948"
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


def hazard_curve(df: pd.DataFrame, n_bins: int = 60) -> pd.DataFrame:
    """P(accept | entropy in bin), at the level of individual positions,
    pooling all positions together (unlike the position-indexed curve in
    entropy_vs_acceptance.py, which conflates position and entropy value)."""
    max_e = df["entropy"].quantile(0.995)
    edges = np.linspace(0, max_e, n_bins + 1)
    d = df.copy()
    d["bucket"] = pd.cut(d["entropy"], bins=edges, include_lowest=True)
    agg = d.groupby("bucket", observed=True)["accepted"].agg(["mean", "count"]).reset_index()
    agg["bucket_mid"] = agg["bucket"].apply(lambda b: (b.left + b.right) / 2)
    agg = agg[agg["count"] >= 50]
    return agg


def survival_curve(df: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    """P(accept | entropy >= x) for a sweep of thresholds x."""
    max_e = df["entropy"].quantile(0.995)
    thresholds = np.linspace(0, max_e, n_points)
    entropy = df["entropy"].to_numpy()
    accepted = df["accepted"].to_numpy()
    order = np.argsort(entropy)
    entropy_sorted = entropy[order]
    accepted_sorted = accepted[order].astype(np.float64)
    # cumulative sum from the right: for threshold x, records with entropy >= x
    n = len(entropy_sorted)
    cum_from_right = np.cumsum(accepted_sorted[::-1])[::-1]
    count_from_right = np.arange(n, 0, -1)
    rows = []
    for x in thresholds:
        idx = np.searchsorted(entropy_sorted, x, side="left")
        if idx >= n:
            continue
        p_accept = cum_from_right[idx] / count_from_right[idx]
        rows.append({"threshold": float(x), "p_accept_given_ge": float(p_accept), "n": int(count_from_right[idx])})
    return pd.DataFrame(rows)


def block_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per drafted block: mean/std entropy across its positions,
    plus its (shared) accepted_len and block_size."""
    g = df.groupby(["req_id", "t"])
    out = g.agg(
        mean_entropy=("entropy", "mean"),
        std_entropy=("entropy", "std"),
        accepted_len=("accepted_len", "first"),
        block_size=("block_size", "first"),
    ).reset_index()
    out = out[out["block_size"] >= 3]  # need >=3 positions for a meaningful std
    out["std_entropy"] = out["std_entropy"].fillna(0.0)
    return out


def partial_corr(y: np.ndarray, x: np.ndarray, control: np.ndarray) -> float:
    """Pearson correlation of y and x after linearly regressing out `control`
    from both (a standard partial-correlation via residuals)."""
    def residualize(v, c):
        A = np.vstack([c, np.ones_like(c)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef

    y_resid = residualize(y, control)
    x_resid = residualize(x, control)
    return float(np.corrcoef(y_resid, x_resid)[0, 1])


def plot_hazard(hazard: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(hazard["bucket_mid"], hazard["mean"] * 100, color=BLUE, marker="o", markersize=3, linewidth=1.8)
    ax.set_xlabel("Target-distribution entropy (nats)", color=TEXT)
    ax.set_ylabel("P(accept | entropy in bin) (%)", color=TEXT)
    ax.set_title(
        "Hazard curve: per-position acceptance probability vs. entropy\n"
        "(pools all block positions -- position and entropy value disentangled)",
        color=TEXT,
        fontsize=11,
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_survival(survival: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(
        survival["threshold"], survival["p_accept_given_ge"] * 100, color=RED, linewidth=2
    )
    ax.set_xlabel("Entropy threshold x (nats)", color=TEXT)
    ax.set_ylabel("P(accept | entropy >= x) (%)", color=TEXT)
    ax.set_title("Survival curve: acceptance probability among high-entropy positions", color=TEXT, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_variance(block_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax = axes[0]
    sc = ax.scatter(
        block_df["mean_entropy"], block_df["std_entropy"], c=block_df["accepted_len"],
        cmap="RdYlBu_r", s=6, alpha=0.35, linewidths=0, vmin=1, vmax=block_df["accepted_len"].quantile(0.95),
    )
    ax.set_xlabel("Block mean entropy (nats)", color=TEXT)
    ax.set_ylabel("Block entropy std (nats)", color=TEXT)
    ax.set_title("Block mean vs. spread of entropy\n(color = accepted length)", color=TEXT, fontsize=10)
    fig.colorbar(sc, ax=ax, label="accepted length")

    ax = axes[1]
    edges = np.linspace(0, block_df["std_entropy"].quantile(0.995), 16)
    block_df = block_df.copy()
    block_df["std_bucket"] = pd.cut(block_df["std_entropy"], bins=edges, include_lowest=True)
    agg = block_df.groupby("std_bucket", observed=True)["accepted_len"].agg(["mean", "count"]).reset_index()
    agg["mid"] = agg["std_bucket"].apply(lambda b: (b.left + b.right) / 2)
    agg = agg[agg["count"] >= 20]
    ax.plot(agg["mid"], agg["mean"], color=BLUE, marker="o", markersize=4, linewidth=2)
    ax.set_xlabel("Block entropy std (nats)", color=TEXT)
    ax.set_ylabel("Mean accepted length", color=TEXT)
    ax.set_title("Accepted length vs. entropy spread\n(raw, not mean-controlled)", color=TEXT, fontsize=10)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    print("Loading entropy log...")
    df = load_all()
    print(f"  {len(df):,} position-level records")

    print("Computing hazard curve (per-position acceptance vs. entropy)...")
    hazard = hazard_curve(df)

    print("Computing survival curve (P(accept | entropy >= x))...")
    survival = survival_curve(df)

    print("Computing block-level mean/std entropy...")
    block_df = block_level(df)
    print(f"  {len(block_df):,} blocks with >=3 positions")

    pearson_mean = block_df["mean_entropy"].corr(block_df["accepted_len"])
    pearson_std = block_df["std_entropy"].corr(block_df["accepted_len"])
    p_corr_std_given_mean = partial_corr(
        block_df["accepted_len"].to_numpy(dtype=np.float64),
        block_df["std_entropy"].to_numpy(dtype=np.float64),
        block_df["mean_entropy"].to_numpy(dtype=np.float64),
    )
    p_corr_mean_given_std = partial_corr(
        block_df["accepted_len"].to_numpy(dtype=np.float64),
        block_df["mean_entropy"].to_numpy(dtype=np.float64),
        block_df["std_entropy"].to_numpy(dtype=np.float64),
    )

    print(f"\naccepted_len vs mean_entropy: Pearson r = {pearson_mean:.4f}")
    print(f"accepted_len vs std_entropy:  Pearson r = {pearson_std:.4f}")
    print(f"partial corr(accepted_len, std_entropy  | mean_entropy) = {p_corr_std_given_mean:.4f}")
    print(f"partial corr(accepted_len, mean_entropy | std_entropy)  = {p_corr_mean_given_std:.4f}")

    out = {
        "n_position_records": int(len(df)),
        "n_blocks_ge3": int(len(block_df)),
        "hazard_curve": [
            {"entropy": round(float(r["bucket_mid"]), 3), "p_accept": round(float(r["mean"]), 4), "n": int(r["count"])}
            for _, r in hazard.iterrows()
        ],
        "survival_curve": [
            {"threshold": round(float(r["threshold"]), 3), "p_accept_given_ge": round(float(r["p_accept_given_ge"]), 4), "n": int(r["n"])}
            for _, r in survival.iterrows()
        ],
        "block_level": {
            "pearson_mean_entropy_vs_accepted_len": round(float(pearson_mean), 4),
            "pearson_std_entropy_vs_accepted_len": round(float(pearson_std), 4),
            "partial_corr_std_given_mean": round(p_corr_std_given_mean, 4),
            "partial_corr_mean_given_std": round(p_corr_mean_given_std, 4),
        },
    }

    with open(OUT_DIR / "entropy_distribution_shape.json", "w") as f:
        json.dump(out, f, indent=2)

    plot_hazard(hazard, OUT_DIR / "entropy_hazard_curve.png")
    plot_survival(survival, OUT_DIR / "entropy_survival_curve.png")
    plot_variance(block_df, OUT_DIR / "entropy_variance_vs_acceptance.png")

    print(f"\nSaved data and plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
