#!/usr/bin/env python3
"""Generate publication-quality heatmap figures for layer × perturbation type analysis.

Usage:
    python plot_heatmaps.py --data-dir Llama-3-8B-results --layers 0 4 8 12 16 20 24 28 32
    python plot_heatmaps.py --data-dir Phi-3.5-MoE-results --auto-layers
    python plot_heatmaps.py --data-dir Phi-3.5-MoE-results --auto-layers --num-layers 10
"""

import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ── Config (defaults, overridden by CLI args) ────────────────────────────────
DATA_DIR = "Qwen2.5-32B-results"
OUT_DIR = Path(DATA_DIR) / "figures"

SAMPLED_LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
BUCKETS = ["immediate", "short", "all_remaining"]
BUCKET_LABELS = {"immediate": "Immediate", "short": "Short", "all_remaining": "All Remaining"}
PERTURBATION_ORDER = ["random_replace", "mask", "delete", "compress", "paraphrase", "semantic_change"]
PRETTY_PERTURB = {
    "random_replace": "Random Replace",
    "mask": "Mask",
    "delete": "Delete",
    "compress": "Compress",
    "paraphrase": "Paraphrase",
    "semantic_change": "Semantic Change",
}
POSITIONS = [0.1, 0.25, 0.5, 0.75, 0.9]
POSITION_LABELS = ["10%", "25%", "50%", "75%", "90%"]
CMAP = "YlOrRd"
DPI = 200


def auto_select_layers(df: pd.DataFrame, num_layers: int = 12) -> list:
    """Pick ~num_layers evenly spaced layers from what exists in the data,
    always including the first and last layer."""
    all_layers = sorted(df["layer"].unique())
    if len(all_layers) <= num_layers:
        return all_layers
    indices = np.linspace(0, len(all_layers) - 1, num_layers, dtype=int)
    return [int(all_layers[i]) for i in np.unique(indices)]


def load_data():
    files = sorted(glob.glob(f"{DATA_DIR}/results_part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}/")
    df = pd.read_parquet(files)
    df = df[df["bucket"].isin(BUCKETS) & df["layer"].isin(SAMPLED_LAYERS)]
    return df


def pivot_perturb_layer(group_df):
    """Pivot to perturbation_type (rows) × layer (columns), values = 1 - mean_similarity."""
    pt = group_df.pivot_table(
        index="perturbation_type", columns="layer", values="mean_similarity", aggfunc="mean"
    )
    pt = pt.reindex(index=PERTURBATION_ORDER, columns=SAMPLED_LAYERS)
    return 1 - pt


def pivot_position_layer(group_df):
    """Pivot to position (rows) × layer (columns), values = 1 - mean_similarity."""
    pt = group_df.pivot_table(
        index="perturbation_position_frac", columns="layer", values="mean_similarity", aggfunc="mean"
    )
    pt = pt.reindex(index=POSITIONS, columns=SAMPLED_LAYERS)
    return 1 - pt


# ── Figure 1: Main Heatmap Triptych ─────────────────────────────────────────
def plot_triptych(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
    pivots = {b: pivot_perturb_layer(df[df["bucket"] == b]) for b in BUCKETS}
    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())

    for ax, bucket in zip(axes, BUCKETS):
        data = pivots[bucket]
        sns.heatmap(
            data, ax=ax, cmap=CMAP, vmin=vmin, vmax=vmax,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            cbar=False, linewidths=0.5, linecolor="white",
            xticklabels=[str(l) for l in SAMPLED_LAYERS],
            yticklabels=[PRETTY_PERTURB[p] for p in PERTURBATION_ORDER] if ax == axes[0] else False,
        )
        ax.set_title(BUCKET_LABELS[bucket], fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Perturbation Type", fontsize=11)
        ax.tick_params(axis="x", rotation=0)

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Perturbation Impact (1 − cosine similarity)", fontsize=11)

    fig.suptitle("Perturbation Impact by Layer and Type", fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / "heatmap_triptych.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'heatmap_triptych.png'}")
    return pivots


# ── Figure 2: Position-Stratified Heatmaps (5×3) ────────────────────────────
def plot_by_position(df):
    fig, axes = plt.subplots(5, 3, figsize=(22, 20), sharey=True, constrained_layout=True)
    all_pivots = {}
    for pos in POSITIONS:
        for bucket in BUCKETS:
            sub = df[(df["perturbation_position_frac"] == pos) & (df["bucket"] == bucket)]
            all_pivots[(pos, bucket)] = pivot_perturb_layer(sub)

    vmin = min(p.min().min() for p in all_pivots.values())
    vmax = max(p.max().max() for p in all_pivots.values())

    for i, pos in enumerate(POSITIONS):
        for j, bucket in enumerate(BUCKETS):
            ax = axes[i, j]
            data = all_pivots[(pos, bucket)]
            show_yticklabels = j == 0
            sns.heatmap(
                data, ax=ax, cmap=CMAP, vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar=False, linewidths=0.5, linecolor="white",
                xticklabels=[str(l) for l in SAMPLED_LAYERS],
                yticklabels=[PRETTY_PERTURB[p] for p in PERTURBATION_ORDER] if show_yticklabels else False,
            )
            if i == 0:
                ax.set_title(BUCKET_LABELS[bucket], fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Position {POSITION_LABELS[i]}", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Layer" if i == len(POSITIONS) - 1 else "", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Perturbation Impact (1 − cosine similarity)", fontsize=11)

    fig.suptitle("Perturbation Impact by Position, Layer, and Type", fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / "heatmap_by_position.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'heatmap_by_position.png'}")


# ── Figure 3: Position Effect Heatmap (1×3) ─────────────────────────────────
def plot_position_effect(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True, constrained_layout=True)
    pivots = {}
    for bucket in BUCKETS:
        pivots[bucket] = pivot_position_layer(df[df["bucket"] == bucket])

    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())

    for ax, bucket in zip(axes, BUCKETS):
        data = pivots[bucket]
        sns.heatmap(
            data, ax=ax, cmap=CMAP, vmin=vmin, vmax=vmax,
            annot=True, fmt=".3f", annot_kws={"size": 8},
            cbar=False, linewidths=0.5, linecolor="white",
            xticklabels=[str(l) for l in SAMPLED_LAYERS],
            yticklabels=POSITION_LABELS if ax == axes[0] else False,
        )
        ax.set_title(BUCKET_LABELS[bucket], fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Perturbation Position", fontsize=11)
        ax.tick_params(axis="x", rotation=0)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Perturbation Impact (1 − cosine similarity)", fontsize=11)

    fig.suptitle("Position × Layer Interaction (Averaged Across Perturbation Types)",
                 fontsize=15, fontweight="bold")
    fig.savefig(OUT_DIR / "heatmap_position_effect.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'heatmap_position_effect.png'}")
    return pivots


# ── Figure 4: Per-Perturbation-Type Position Heatmaps (6×3) ─────────────────
def plot_position_per_type(df):
    fig, axes = plt.subplots(6, 3, figsize=(22, 24), sharey=True, constrained_layout=True)
    all_pivots = {}
    for ptype in PERTURBATION_ORDER:
        for bucket in BUCKETS:
            sub = df[(df["perturbation_type"] == ptype) & (df["bucket"] == bucket)]
            all_pivots[(ptype, bucket)] = pivot_position_layer(sub)

    vmin = min(p.min().min() for p in all_pivots.values())
    vmax = max(p.max().max() for p in all_pivots.values())

    for i, ptype in enumerate(PERTURBATION_ORDER):
        for j, bucket in enumerate(BUCKETS):
            ax = axes[i, j]
            data = all_pivots[(ptype, bucket)]
            show_yticklabels = j == 0
            sns.heatmap(
                data, ax=ax, cmap=CMAP, vmin=vmin, vmax=vmax,
                annot=True, fmt=".3f", annot_kws={"size": 7},
                cbar=False, linewidths=0.5, linecolor="white",
                xticklabels=[str(l) for l in SAMPLED_LAYERS],
                yticklabels=POSITION_LABELS if show_yticklabels else False,
            )
            if i == 0:
                ax.set_title(BUCKET_LABELS[bucket], fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(PRETTY_PERTURB[ptype], fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Layer" if i == len(PERTURBATION_ORDER) - 1 else "", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Perturbation Impact (1 − cosine similarity)", fontsize=11)

    fig.suptitle("Position × Layer Interaction per Perturbation Type",
                 fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / "heatmap_position_per_type.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'heatmap_position_per_type.png'}")


# ── Analysis ─────────────────────────────────────────────────────────────────
def print_analysis(df):
    print("\n" + "=" * 80)
    print("ANALYSIS: Layer × Perturbation Type Heatmap Results")
    print("=" * 80)

    # 1. Strongest perturbation effects by layer
    print("\n## 1. Layers with Strongest Perturbation Effects")
    for bucket in BUCKETS:
        sub = df[df["bucket"] == bucket]
        layer_impact = sub.groupby("layer")["mean_similarity"].mean()
        layer_impact = 1 - layer_impact
        layer_impact = layer_impact[layer_impact.index.isin(SAMPLED_LAYERS)].sort_values(ascending=False)
        top3 = layer_impact.head(3)
        print(f"\n  {BUCKET_LABELS[bucket]}:")
        for layer, val in top3.items():
            print(f"    Layer {layer:2d}: impact = {val:.4f}")

    # 2. Perturbation type ranking
    print("\n## 2. Perturbation Type Ranking (by mean impact)")
    for bucket in BUCKETS:
        sub = df[(df["bucket"] == bucket) & df["layer"].isin(SAMPLED_LAYERS)]
        type_impact = sub.groupby("perturbation_type")["mean_similarity"].mean()
        type_impact = (1 - type_impact).sort_values(ascending=False)
        print(f"\n  {BUCKET_LABELS[bucket]}:")
        for ptype, val in type_impact.items():
            print(f"    {PRETTY_PERTURB[ptype]:20s}: {val:.4f}")

    # 3. Measurement window differences
    print("\n## 3. Measurement Window Comparison")
    for bucket in BUCKETS:
        sub = df[(df["bucket"] == bucket) & df["layer"].isin(SAMPLED_LAYERS)]
        mean_impact = 1 - sub["mean_similarity"].mean()
        std_impact = sub["mean_similarity"].std()
        print(f"  {BUCKET_LABELS[bucket]:15s}: mean impact = {mean_impact:.4f}, std = {std_impact:.4f}")

    # 4. Position sensitivity
    print("\n## 4. Position Sensitivity Patterns")
    for bucket in BUCKETS:
        sub = df[(df["bucket"] == bucket) & df["layer"].isin(SAMPLED_LAYERS)]
        pos_impact = sub.groupby("perturbation_position_frac")["mean_similarity"].mean()
        pos_impact = 1 - pos_impact
        print(f"\n  {BUCKET_LABELS[bucket]}:")
        for pos in POSITIONS:
            label = f"{int(pos*100)}%"
            print(f"    Position {label:>3s}: impact = {pos_impact[pos]:.4f}")

    # 5. Key findings
    print("\n## 5. Key Findings and Conclusions")
    # Find most disruptive perturbation overall
    sub_imm = df[(df["bucket"] == "immediate") & df["layer"].isin(SAMPLED_LAYERS)]
    worst_type = (1 - sub_imm.groupby("perturbation_type")["mean_similarity"].mean()).idxmax()
    print(f"  - Most disruptive perturbation (immediate): {PRETTY_PERTURB[worst_type]}")

    # Find peak layer
    peak_layer = (1 - sub_imm.groupby("layer")["mean_similarity"].mean()).idxmax()
    print(f"  - Peak impact layer (immediate): {peak_layer}")

    # Recovery pattern
    imm_mean = 1 - df[(df["bucket"] == "immediate") & df["layer"].isin(SAMPLED_LAYERS)]["mean_similarity"].mean()
    short_mean = 1 - df[(df["bucket"] == "short") & df["layer"].isin(SAMPLED_LAYERS)]["mean_similarity"].mean()
    all_mean = 1 - df[(df["bucket"] == "all_remaining") & df["layer"].isin(SAMPLED_LAYERS)]["mean_similarity"].mean()
    print(f"  - Recovery pattern: immediate ({imm_mean:.4f}) → short ({short_mean:.4f}) → all_remaining ({all_mean:.4f})")
    if short_mean < imm_mean:
        print(f"    → {((imm_mean - short_mean) / imm_mean * 100):.1f}% reduction from immediate to short")
    if all_mean < imm_mean:
        print(f"    → {((imm_mean - all_mean) / imm_mean * 100):.1f}% reduction from immediate to all_remaining")

    # Position effect
    pos_imm = 1 - sub_imm.groupby("perturbation_position_frac")["mean_similarity"].mean()
    most_sensitive_pos = pos_imm.idxmax()
    least_sensitive_pos = pos_imm.idxmin()
    print(f"  - Most sensitive position (immediate): {int(most_sensitive_pos*100)}%")
    print(f"  - Least sensitive position (immediate): {int(least_sensitive_pos*100)}%")

    # Layer gradient
    early = (1 - sub_imm[sub_imm["layer"] <= 10].groupby("layer")["mean_similarity"].mean()).mean()
    mid = (1 - sub_imm[(sub_imm["layer"] >= 25) & (sub_imm["layer"] <= 40)].groupby("layer")["mean_similarity"].mean()).mean()
    late = (1 - sub_imm[sub_imm["layer"] >= 50].groupby("layer")["mean_similarity"].mean()).mean()
    print(f"  - Layer gradient (immediate): early={early:.4f}, mid={mid:.4f}, late={late:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality heatmap figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing results_part_*.parquet files "
             "(default: Qwen2.5-32B-results).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for output figures (default: <data-dir>/figures).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of layer indices to sample, e.g. --layers 0 4 8 16 24 32.",
    )
    parser.add_argument(
        "--auto-layers",
        action="store_true",
        help="Automatically select evenly-spaced layers from the data "
             "(ignores --layers).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of layers to select when using --auto-layers (default: 12).",
    )
    return parser.parse_args()


def main():
    global DATA_DIR, OUT_DIR, SAMPLED_LAYERS

    args = parse_args()

    # ── Apply CLI overrides ──────────────────────────────────────────────
    if args.data_dir is not None:
        DATA_DIR = args.data_dir

    OUT_DIR = Path(args.out_dir) if args.out_dir else Path(DATA_DIR) / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
    })

    # ── Resolve layers ───────────────────────────────────────────────────
    if args.auto_layers:
        # Read raw data first to discover layers, then filter.
        files = sorted(glob.glob(f"{DATA_DIR}/results_part_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {DATA_DIR}/")
        raw = pd.read_parquet(files)
        SAMPLED_LAYERS = auto_select_layers(raw, num_layers=args.num_layers)
        print(f"Auto-selected {len(SAMPLED_LAYERS)} layers: {SAMPLED_LAYERS}")
        df = raw[raw["bucket"].isin(BUCKETS) & raw["layer"].isin(SAMPLED_LAYERS)]
        del raw
    else:
        if args.layers is not None:
            SAMPLED_LAYERS = sorted(args.layers)
        # else: keep the default SAMPLED_LAYERS
        print("Loading data...")
        df = load_data()

    print(f"Loaded {len(df):,} records (filtered to {len(SAMPLED_LAYERS)} layers, "
          f"{len(BUCKETS)} buckets)")
    print(f"Data dir : {DATA_DIR}")
    print(f"Output   : {OUT_DIR}")

    print("\nGenerating Figure 1: Main Heatmap Triptych...")
    plot_triptych(df)

    print("Generating Figure 2: Position-Stratified Heatmaps...")
    plot_by_position(df)

    print("Generating Figure 3: Position Effect Heatmap...")
    plot_position_effect(df)

    print("Generating Figure 4: Per-Perturbation-Type Position Heatmaps...")
    plot_position_per_type(df)

    print_analysis(df)
    print("\nDone! All figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
