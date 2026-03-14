#!/usr/bin/env python3
"""
Result analysis and hypothesis validation for activation analysis experiments.

Loads experiment results, runs statistical hypothesis tests, generates
publication-quality visualizations, and produces a verdict on research hypotheses.

Usage:
    python scripts/analyze_results.py results/auto_test_XXXXXXXX/
    python scripts/analyze_results.py results/auto_test_XXXXXXXX/ --output-dir custom_report/
    python scripts/analyze_results.py results/auto_test_XXXXXXXX/ --phase 2
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DPI = 150
FIGSIZE_LINE = (12, 6)
FIGSIZE_BAR = (10, 6)
FIGSIZE_HEATMAP = (14, 8)
FIGSIZE_SCATTER = (12, 5)

REQUIRED_COLUMNS_PHASE2 = [
    "layer_index", "cosine_mean", "perturbation_ratio", "position",
]

REQUIRED_COLUMNS_MINIMAL = [
    "layer_index", "perturbation_ratio", "position",
]

LAYER_SHALLOW_MAX = 12
LAYER_MIDDLE_MAX = 31


def _setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(result_dir: Path, phase: int | None = None) -> pd.DataFrame:
    """Auto-discover and load metrics CSVs from an experiment directory."""
    candidates = []
    if phase is not None:
        candidates.append(result_dir / f"phase{phase}" / "metrics.csv")
    else:
        candidates.append(result_dir / "phase2" / "metrics.csv")
        candidates.append(result_dir / "phase3" / "metrics.csv")
        candidates.append(result_dir / "metrics.csv")

    frames = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows from {path}")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No metrics.csv found in {result_dir}. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in REQUIRED_COLUMNS_MINIMAL if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Total: {len(df)} rows, "
                f"{df['layer_index'].nunique()} unique layers, "
                f"{df['perturbation_ratio'].nunique()} unique ratios, "
                f"{df['position'].nunique()} unique positions")
    return df


def load_phase_data(result_dir: Path, phase: int) -> pd.DataFrame | None:
    """Load data for a specific phase, return None if not found."""
    path = result_dir / f"phase{phase}" / "metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    logger.info(f"Phase {phase}: Loaded {len(df)} rows from {path}")
    return df


# ---------------------------------------------------------------------------
# Helper: depth grouping
# ---------------------------------------------------------------------------

def classify_depth(layer_idx: int) -> str:
    if layer_idx <= LAYER_SHALLOW_MAX:
        return "Shallow"
    elif layer_idx <= LAYER_MIDDLE_MAX:
        return "Middle"
    return "Deep"


def add_depth_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["depth_group"] = df["layer_index"].apply(classify_depth)
    return df


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_std


# ---------------------------------------------------------------------------
# Hypothesis Tests
# ---------------------------------------------------------------------------

class HypothesisResult:
    def __init__(self, name: str, supported: bool, details: str):
        self.name = name
        self.supported = supported
        self.details = details

    def __str__(self):
        verdict = "SUPPORTED" if self.supported else "NOT SUPPORTED"
        return f"{self.name}\n{self.details}\n  Verdict: {verdict}"


def test_h1_layer_divergence(df: pd.DataFrame) -> HypothesisResult:
    """H1: Perturbations cause more activation divergence in deeper layers."""
    df = add_depth_group(df)

    shallow = df.loc[df["depth_group"] == "Shallow", "cosine_mean"].values
    deep = df.loc[df["depth_group"] == "Deep", "cosine_mean"].values

    lines = []

    # Spearman correlation: layer_index vs cosine_mean
    rho, p_spearman = stats.spearmanr(df["layer_index"], df["cosine_mean"])
    lines.append(f"  Spearman ρ(layer, cosine) = {rho:.4f}, p = {p_spearman:.6f}")

    # Wilcoxon-like comparison — use Mann-Whitney since samples are independent
    if len(shallow) > 0 and len(deep) > 0:
        u_stat, p_mw = stats.mannwhitneyu(shallow, deep, alternative="greater")
        lines.append(f"  Mann-Whitney U (shallow > deep): U = {u_stat:.1f}, p = {p_mw:.6f}")
    else:
        p_mw = 1.0
        lines.append("  Mann-Whitney U: insufficient data")

    # Effect size
    d = cohens_d(shallow, deep) if len(shallow) > 0 and len(deep) > 0 else float("nan")
    s_mean = shallow.mean() if len(shallow) > 0 else float("nan")
    d_mean = deep.mean() if len(deep) > 0 else float("nan")
    lines.append(f"  Shallow mean = {s_mean:.4f}, Deep mean = {d_mean:.4f}, "
                 f"Delta = {s_mean - d_mean:.4f}, Cohen's d = {d:.4f}")

    # Pass criteria: p < 0.05 AND negative correlation AND |d| > 0.5
    supported = (p_spearman < 0.05 and rho < 0 and abs(d) > 0.5)

    return HypothesisResult("H1: Layer Depth Divergence", supported, "\n".join(lines))


def test_h2_ratio_scaling(df: pd.DataFrame) -> HypothesisResult:
    """H2: Larger perturbation ratios produce proportionally larger activation changes."""
    lines = []

    # Spearman: ratio vs (1 - cosine_mean)
    divergence = 1.0 - df["cosine_mean"]
    rho, p_spearman = stats.spearmanr(df["perturbation_ratio"], divergence)
    lines.append(f"  Spearman ρ(ratio, 1−cosine) = {rho:.4f}, p = {p_spearman:.6f}")

    # Kruskal-Wallis across ratio groups
    ratio_groups = [g["cosine_mean"].values for _, g in df.groupby("perturbation_ratio")]
    if len(ratio_groups) >= 2:
        h_stat, p_kw = stats.kruskal(*ratio_groups)
        lines.append(f"  Kruskal-Wallis H = {h_stat:.4f}, p = {p_kw:.6f}")
    else:
        p_kw = 1.0
        lines.append("  Kruskal-Wallis: insufficient groups")

    # Pairwise Mann-Whitney with Bonferroni
    ratios_sorted = sorted(df["perturbation_ratio"].unique())
    n_pairs = max(len(ratios_sorted) - 1, 1)
    for i in range(len(ratios_sorted) - 1):
        r_low = ratios_sorted[i]
        r_high = ratios_sorted[i + 1]
        a = df.loc[df["perturbation_ratio"] == r_low, "cosine_mean"].values
        b = df.loc[df["perturbation_ratio"] == r_high, "cosine_mean"].values
        if len(a) > 0 and len(b) > 0:
            u, p_pw = stats.mannwhitneyu(a, b, alternative="greater")
            p_adj = min(p_pw * n_pairs, 1.0)  # Bonferroni
            lines.append(f"  Pairwise {r_low} vs {r_high}: U = {u:.1f}, "
                         f"p = {p_pw:.6f}, p_adj = {p_adj:.6f}")

    # Monotonicity check
    group_means = df.groupby("perturbation_ratio")["cosine_mean"].mean().sort_index()
    monotonic = all(
        group_means.iloc[i] >= group_means.iloc[i + 1]
        for i in range(len(group_means) - 1)
    )
    lines.append(f"  Monotonic decrease: {monotonic} (means: {dict(group_means.round(4))})")

    supported = (p_spearman < 0.05 and rho > 0 and monotonic)

    return HypothesisResult("H2: Perturbation Ratio Scaling", supported, "\n".join(lines))


def test_h3_position_effect(df: pd.DataFrame) -> HypothesisResult:
    """H3: Perturbation position affects propagation differently."""
    lines = []

    positions = df["position"].unique()
    pos_groups = [g["cosine_mean"].values for _, g in df.groupby("position")]

    # Kruskal-Wallis
    if len(pos_groups) >= 2:
        h_stat, p_kw = stats.kruskal(*pos_groups)
        lines.append(f"  Kruskal-Wallis H = {h_stat:.4f}, p = {p_kw:.6f}")
    else:
        p_kw = 1.0
        h_stat = float("nan")
        lines.append("  Kruskal-Wallis: insufficient groups")

    # Pairwise comparisons
    pos_list = sorted(positions)
    n_pairs = max(len(pos_list) * (len(pos_list) - 1) // 2, 1)
    for i in range(len(pos_list)):
        for j in range(i + 1, len(pos_list)):
            a = df.loc[df["position"] == pos_list[i], "cosine_mean"].values
            b = df.loc[df["position"] == pos_list[j], "cosine_mean"].values
            if len(a) > 0 and len(b) > 0:
                u, p_pw = stats.mannwhitneyu(a, b, alternative="two-sided")
                p_adj = min(p_pw * n_pairs, 1.0)
                lines.append(f"  {pos_list[i]} vs {pos_list[j]}: "
                             f"U = {u:.1f}, p = {p_pw:.6f}, p_adj = {p_adj:.6f}")

    # Interaction: position × depth group
    df2 = add_depth_group(df)
    for depth in ["Shallow", "Middle", "Deep"]:
        subset = df2[df2["depth_group"] == depth]
        sub_groups = [g["cosine_mean"].values for _, g in subset.groupby("position")]
        if len(sub_groups) >= 2 and all(len(g) > 0 for g in sub_groups):
            h, p = stats.kruskal(*sub_groups)
            lines.append(f"  Position effect in {depth} layers: H = {h:.4f}, p = {p:.6f}")

    supported = (p_kw < 0.05)

    return HypothesisResult("H3: Position Effect", supported, "\n".join(lines))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_layer_divergence_curve(df: pd.DataFrame, output_path: Path):
    """Line: X=layer, Y=cosine_mean, separate lines per ratio, with error bands."""
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for ratio in sorted(df["perturbation_ratio"].unique()):
        sub = df[df["perturbation_ratio"] == ratio]
        layer_stats = sub.groupby("layer_index")["cosine_mean"].agg(["mean", "std"])
        ax.plot(layer_stats.index, layer_stats["mean"], marker="o",
                label=f"ratio={ratio}")
        ax.fill_between(layer_stats.index,
                        layer_stats["mean"] - layer_stats["std"],
                        layer_stats["mean"] + layer_stats["std"],
                        alpha=0.15)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity (mean)")
    ax.set_title("Activation Divergence Across Layers by Perturbation Ratio")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_shallow_vs_deep_boxplot(df: pd.DataFrame, output_path: Path):
    """Violin/box plot of shallow/middle/deep groups."""
    _setup_style()
    df = add_depth_group(df)

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    order = ["Shallow", "Middle", "Deep"]
    sns.violinplot(data=df, x="depth_group", y="cosine_mean", order=order,
                   inner="box", ax=ax, palette="Set2")

    # Add p-value bracket between shallow and deep
    shallow = df.loc[df["depth_group"] == "Shallow", "cosine_mean"].values
    deep = df.loc[df["depth_group"] == "Deep", "cosine_mean"].values
    if len(shallow) > 0 and len(deep) > 0:
        _, p_val = stats.mannwhitneyu(shallow, deep, alternative="two-sided")
        y_max = df["cosine_mean"].max() + 0.02
        ax.plot([0, 0, 2, 2], [y_max, y_max + 0.01, y_max + 0.01, y_max],
                color="black", linewidth=1)
        ax.text(1, y_max + 0.015, f"p = {p_val:.4f}", ha="center", fontsize=10)

    ax.set_xlabel("Layer Depth Group")
    ax.set_ylabel("Cosine Similarity (mean)")
    ax.set_title("Activation Similarity by Layer Depth Group")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_ratio_scaling(df: pd.DataFrame, output_path: Path):
    """Grouped bars: cosine_mean vs ratio, faceted by depth group."""
    _setup_style()
    df = add_depth_group(df)

    agg = df.groupby(["perturbation_ratio", "depth_group"])["cosine_mean"].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    sns.barplot(data=agg, x="perturbation_ratio", y="cosine_mean",
                hue="depth_group", hue_order=["Shallow", "Middle", "Deep"],
                ax=ax, palette="Set2")
    ax.set_xlabel("Perturbation Ratio")
    ax.set_ylabel("Cosine Similarity (mean)")
    ax.set_title("Ratio Scaling by Layer Depth Group")
    ax.legend(title="Depth Group")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_position_effect(df: pd.DataFrame, output_path: Path):
    """Line: X=layer, Y=cosine_mean, separate lines per position."""
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for position in sorted(df["position"].unique()):
        sub = df[df["position"] == position]
        layer_stats = sub.groupby("layer_index")["cosine_mean"].agg(["mean", "std"])
        ax.plot(layer_stats.index, layer_stats["mean"], marker="s",
                label=f"Position: {position}")
        ax.fill_between(layer_stats.index,
                        layer_stats["mean"] - layer_stats["std"],
                        layer_stats["mean"] + layer_stats["std"],
                        alpha=0.15)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity (mean)")
    ax.set_title("Effect of Perturbation Position on Activation Similarity")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_heatmap_overview(df: pd.DataFrame, output_path: Path):
    """Heatmap: rows=ratio×position, cols=layer_index, color=cosine_mean."""
    _setup_style()

    df = df.copy()
    df["condition"] = df["perturbation_ratio"].astype(str) + " / " + df["position"]

    pivot = df.pivot_table(
        index="condition", columns="layer_index",
        values="cosine_mean", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".3f", linewidths=0.5,
                cbar_kws={"label": "Cosine Similarity"})
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Ratio / Position")
    ax.set_title("Cosine Similarity Overview: Ratio × Position × Layer")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_metric_correlation(df: pd.DataFrame, output_path: Path):
    """Scatter: cosine_mean vs l2_mean, and CKA vs cosine_segment with trend lines."""
    _setup_style()

    n_panels = 0
    has_l2 = "l2_mean" in df.columns
    has_cka = "cka" in df.columns and "cosine_segment" in df.columns
    if has_l2:
        n_panels += 1
    if has_cka:
        n_panels += 1

    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No l2_mean or cka/cosine_segment columns available",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if has_l2:
        ax = axes[idx]
        ax.scatter(df["cosine_mean"], df["l2_mean"], alpha=0.4, s=20)
        # Trend line
        mask = df[["cosine_mean", "l2_mean"]].dropna().index
        if len(mask) > 2:
            x, y = df.loc[mask, "cosine_mean"], df.loc[mask, "l2_mean"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, p(xs), "r--", linewidth=1.5, label=f"trend (slope={z[0]:.2f})")
            ax.legend()
        ax.set_xlabel("Cosine Similarity (mean)")
        ax.set_ylabel("L2 Distance (mean)")
        ax.set_title("Cosine vs L2")
        idx += 1

    if has_cka:
        ax = axes[idx]
        ax.scatter(df["cosine_segment"], df["cka"], alpha=0.4, s=20)
        mask = df[["cosine_segment", "cka"]].dropna().index
        if len(mask) > 2:
            x, y = df.loc[mask, "cosine_segment"], df.loc[mask, "cka"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, p(xs), "r--", linewidth=1.5, label=f"trend (slope={z[0]:.2f})")
            ax.legend()
        ax.set_xlabel("Cosine Segment")
        ax.set_ylabel("CKA")
        ax.set_title("CKA vs Cosine Segment")

    fig.suptitle("Metric Correlations", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_effect_size_summary(df: pd.DataFrame, output_path: Path):
    """Bar chart of Cohen's d per layer group and condition."""
    _setup_style()
    df = add_depth_group(df)

    records = []
    # Effect size: each ratio vs baseline (smallest ratio)
    ratios_sorted = sorted(df["perturbation_ratio"].unique())
    if len(ratios_sorted) < 2:
        # Compute depth group effect sizes overall
        shallow = df.loc[df["depth_group"] == "Shallow", "cosine_mean"].values
        middle = df.loc[df["depth_group"] == "Middle", "cosine_mean"].values
        deep = df.loc[df["depth_group"] == "Deep", "cosine_mean"].values
        if len(shallow) > 1 and len(middle) > 1:
            records.append({"Comparison": "Shallow vs Middle", "Cohen_d": cohens_d(shallow, middle)})
        if len(shallow) > 1 and len(deep) > 1:
            records.append({"Comparison": "Shallow vs Deep", "Cohen_d": cohens_d(shallow, deep)})
    else:
        baseline_ratio = ratios_sorted[0]
        for depth in ["Shallow", "Middle", "Deep"]:
            baseline = df.loc[
                (df["depth_group"] == depth) & (df["perturbation_ratio"] == baseline_ratio),
                "cosine_mean"
            ].values
            for ratio in ratios_sorted[1:]:
                target = df.loc[
                    (df["depth_group"] == depth) & (df["perturbation_ratio"] == ratio),
                    "cosine_mean"
                ].values
                if len(baseline) > 1 and len(target) > 1:
                    d = cohens_d(baseline, target)
                    records.append({
                        "Comparison": f"{depth}: {baseline_ratio} vs {ratio}",
                        "Cohen_d": d,
                    })

    if not records:
        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        ax.text(0.5, 0.5, "Insufficient data for effect sizes",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path)
        plt.close(fig)
        return

    edf = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(max(10, len(records) * 1.2), 6))
    colors = ["#e74c3c" if abs(v) > 0.8 else "#f39c12" if abs(v) > 0.5 else "#3498db"
              for v in edf["Cohen_d"]]
    ax.barh(edf["Comparison"], edf["Cohen_d"], color=colors)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="|d|=0.5 (medium)")
    ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5, label="|d|=0.8 (large)")
    ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=-0.8, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Size Summary")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def generate_all_analysis_plots(df: pd.DataFrame, plots_dir: Path):
    """Generate all 7 analysis plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_layer_divergence_curve(df, plots_dir / "layer_divergence_curve.png")
    plot_shallow_vs_deep_boxplot(df, plots_dir / "shallow_vs_deep_boxplot.png")
    plot_ratio_scaling(df, plots_dir / "ratio_scaling.png")
    plot_position_effect(df, plots_dir / "position_effect.png")
    plot_heatmap_overview(df, plots_dir / "heatmap_overview.png")
    plot_metric_correlation(df, plots_dir / "metric_correlation.png")
    plot_effect_size_summary(df, plots_dir / "effect_size_summary.png")

    plt.close("all")
    logger.info(f"All analysis plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Verdict Report
# ---------------------------------------------------------------------------

def generate_verdict(results: list[HypothesisResult]) -> str:
    """Generate verdict report string."""
    n_supported = sum(1 for r in results if r.supported)
    n_total = len(results)

    if n_supported == n_total:
        feasibility = "STRONG"
    elif n_supported >= n_total - 1:
        feasibility = "PROMISING"
    elif n_supported >= 1:
        feasibility = "MIXED"
    else:
        feasibility = "WEAK"

    lines = ["=" * 50]
    lines.append("  HYPOTHESIS TESTING REPORT")
    lines.append("=" * 50)
    lines.append("")

    for r in results:
        lines.append(str(r))
        lines.append("")

    lines.append("=" * 50)
    lines.append("  OVERALL VERDICT")
    lines.append("=" * 50)
    lines.append(f"  Hypotheses supported: {n_supported}/{n_total}")
    lines.append(f"  Idea feasibility: {feasibility}")
    lines.append("")

    supported_names = [r.name for r in results if r.supported]
    unsupported_names = [r.name for r in results if not r.supported]

    if supported_names:
        lines.append(f"  Supported: {', '.join(supported_names)}")
    if unsupported_names:
        lines.append(f"  Not supported: {', '.join(unsupported_names)}")

    lines.append("")
    if feasibility == "STRONG":
        lines.append("  Recommendation: All hypotheses confirmed. The research approach")
        lines.append("  is well-validated. Proceed to write-up and deeper analysis.")
    elif feasibility == "PROMISING":
        lines.append("  Recommendation: Core hypotheses confirmed. Investigate the")
        lines.append("  unsupported hypothesis further with additional experiments.")
    elif feasibility == "MIXED":
        lines.append("  Recommendation: Results are inconclusive. Consider refining")
        lines.append("  perturbation parameters or expanding the experiment grid.")
    else:
        lines.append("  Recommendation: The current approach does not show strong")
        lines.append("  signal. Reconsider the experimental design or model choice.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comprehensive TXT analysis generators
# ---------------------------------------------------------------------------

def generate_descriptive_stats(df: pd.DataFrame, label: str) -> str:
    """Generate detailed descriptive statistics for a dataframe."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  DESCRIPTIVE STATISTICS — {label}")
    lines.append(f"{'=' * 70}")
    lines.append(f"")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Unique experiments: {df['experiment_id'].nunique()}")
    lines.append(f"Layers: {sorted(df['layer_index'].unique())}")
    lines.append(f"Ratios: {sorted(df['perturbation_ratio'].unique())}")
    lines.append(f"Positions: {sorted(df['position'].unique())}")
    if "context_length" in df.columns:
        lines.append(f"Context lengths: {sorted(df['context_length'].unique())}")
    if "perturbation_type" in df.columns:
        lines.append(f"Perturbation types: {sorted(df['perturbation_type'].unique())}")
    lines.append("")

    # Per-metric summary
    numeric_cols = ["cosine_mean", "cosine_std", "l2_mean", "l2_std", "cosine_segment", "cka"]
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                lines.append(f"  {col}:")
                lines.append(f"    count={len(s)}, mean={s.mean():.6f}, std={s.std():.6f}")
                lines.append(f"    min={s.min():.6f}, Q1={s.quantile(0.25):.6f}, "
                             f"median={s.median():.6f}, Q3={s.quantile(0.75):.6f}, max={s.max():.6f}")
                lines.append("")

    return "\n".join(lines)


def generate_per_layer_analysis(df: pd.DataFrame, metric: str, label: str) -> str:
    """Generate per-layer breakdown for a given metric."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  PER-LAYER ANALYSIS ({metric}) — {label}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    if metric not in df.columns or df[metric].dropna().empty:
        lines.append(f"  Metric '{metric}' not available in this dataset.")
        return "\n".join(lines)

    layer_stats = df.groupby("layer_index")[metric].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(6)
    lines.append(f"  {'Layer':>6}  {'Count':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    lines.append(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for layer_idx, row in layer_stats.iterrows():
        lines.append(f"  {layer_idx:>6}  {int(row['count']):>6}  {row['mean']:>10.6f}  "
                     f"{row['std']:>10.6f}  {row['min']:>10.6f}  {row['max']:>10.6f}")
    lines.append("")

    # Depth group summary
    df2 = add_depth_group(df)
    depth_stats = df2.groupby("depth_group")[metric].agg(["mean", "std", "count"]).round(6)
    lines.append("  Depth group summary:")
    for group in ["Shallow", "Middle", "Deep"]:
        if group in depth_stats.index:
            r = depth_stats.loc[group]
            lines.append(f"    {group:>8}: mean={r['mean']:.6f}, std={r['std']:.6f}, n={int(r['count'])}")
    lines.append("")

    # Spearman correlation: layer_index vs metric
    valid = df[[metric, "layer_index"]].dropna()
    if len(valid) > 2:
        rho, p = stats.spearmanr(valid["layer_index"], valid[metric])
        lines.append(f"  Spearman correlation (layer_index vs {metric}): rho={rho:.6f}, p={p:.2e}")
    lines.append("")

    return "\n".join(lines)


def generate_per_condition_analysis(df: pd.DataFrame, metric: str, label: str) -> str:
    """Generate per-condition (ratio x position x context_length) breakdown."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  PER-CONDITION ANALYSIS ({metric}) — {label}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    if metric not in df.columns or df[metric].dropna().empty:
        lines.append(f"  Metric '{metric}' not available in this dataset.")
        return "\n".join(lines)

    # Per ratio
    lines.append("  --- By Perturbation Ratio ---")
    ratio_stats = df.groupby("perturbation_ratio")[metric].agg(["mean", "std", "count"]).round(6)
    for ratio, row in ratio_stats.iterrows():
        lines.append(f"    ratio={ratio}: mean={row['mean']:.6f}, std={row['std']:.6f}, n={int(row['count'])}")
    lines.append("")

    # Per position
    lines.append("  --- By Position ---")
    pos_stats = df.groupby("position")[metric].agg(["mean", "std", "count"]).round(6)
    for pos, row in pos_stats.iterrows():
        lines.append(f"    {pos}: mean={row['mean']:.6f}, std={row['std']:.6f}, n={int(row['count'])}")
    lines.append("")

    # Per context length
    if "context_length" in df.columns:
        lines.append("  --- By Context Length ---")
        cl_stats = df.groupby("context_length")[metric].agg(["mean", "std", "count"]).round(6)
        for cl, row in cl_stats.iterrows():
            lines.append(f"    L={int(cl)}: mean={row['mean']:.6f}, std={row['std']:.6f}, n={int(row['count'])}")
        lines.append("")

    # Cross-tabulation: ratio x position
    lines.append("  --- Ratio x Position (mean) ---")
    cross = df.pivot_table(index="perturbation_ratio", columns="position",
                           values=metric, aggfunc="mean")
    if not cross.empty:
        header = f"  {'Ratio':>8}" + "".join(f"  {c:>12}" for c in cross.columns)
        lines.append(header)
        for ratio, row in cross.iterrows():
            vals = "".join(f"  {v:>12.6f}" if pd.notna(v) else f"  {'N/A':>12}" for v in row)
            lines.append(f"  {ratio:>8}{vals}")
    lines.append("")

    # Cross-tabulation: ratio x context_length
    if "context_length" in df.columns:
        lines.append("  --- Ratio x Context Length (mean) ---")
        cross2 = df.pivot_table(index="perturbation_ratio", columns="context_length",
                                values=metric, aggfunc="mean")
        if not cross2.empty:
            header = f"  {'Ratio':>8}" + "".join(f"  {int(c):>10}" for c in cross2.columns)
            lines.append(header)
            for ratio, row in cross2.iterrows():
                vals = "".join(f"  {v:>10.6f}" if pd.notna(v) else f"  {'N/A':>10}" for v in row)
                lines.append(f"  {ratio:>8}{vals}")
    lines.append("")

    # Depth group x ratio
    df2 = add_depth_group(df)
    lines.append("  --- Depth Group x Ratio (mean) ---")
    cross3 = df2.pivot_table(index="depth_group", columns="perturbation_ratio",
                              values=metric, aggfunc="mean")
    if not cross3.empty:
        for depth in ["Shallow", "Middle", "Deep"]:
            if depth in cross3.index:
                vals = ", ".join(f"r={c}:{cross3.loc[depth, c]:.6f}"
                                for c in cross3.columns if pd.notna(cross3.loc[depth, c]))
                lines.append(f"    {depth}: {vals}")
    lines.append("")

    return "\n".join(lines)


def generate_pairwise_tests(df: pd.DataFrame, metric: str, label: str) -> str:
    """Generate comprehensive pairwise statistical tests."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  PAIRWISE STATISTICAL TESTS ({metric}) — {label}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    if metric not in df.columns or df[metric].dropna().empty:
        lines.append(f"  Metric '{metric}' not available.")
        return "\n".join(lines)

    # Depth group pairwise
    lines.append("  --- Depth Group Pairwise (Mann-Whitney U) ---")
    df2 = add_depth_group(df)
    for g1, g2 in [("Shallow", "Middle"), ("Shallow", "Deep"), ("Middle", "Deep")]:
        a = df2.loc[df2["depth_group"] == g1, metric].dropna().values
        b = df2.loc[df2["depth_group"] == g2, metric].dropna().values
        if len(a) > 0 and len(b) > 0:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a, b)
            lines.append(f"    {g1} vs {g2}: U={u:.1f}, p={p:.2e}, Cohen's d={d:.4f}, "
                         f"mean_diff={a.mean()-b.mean():.6f}")
    lines.append("")

    # Ratio pairwise
    lines.append("  --- Ratio Pairwise (Mann-Whitney U, Bonferroni corrected) ---")
    ratios = sorted(df["perturbation_ratio"].unique())
    n_comparisons = len(ratios) * (len(ratios) - 1) // 2
    for i in range(len(ratios)):
        for j in range(i + 1, len(ratios)):
            a = df.loc[df["perturbation_ratio"] == ratios[i], metric].dropna().values
            b = df.loc[df["perturbation_ratio"] == ratios[j], metric].dropna().values
            if len(a) > 0 and len(b) > 0:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                p_adj = min(p * max(n_comparisons, 1), 1.0)
                d = cohens_d(a, b)
                lines.append(f"    ratio {ratios[i]} vs {ratios[j]}: U={u:.1f}, p={p:.2e}, "
                             f"p_adj={p_adj:.2e}, Cohen's d={d:.4f}")
    lines.append("")

    # Position pairwise
    lines.append("  --- Position Pairwise (Mann-Whitney U, Bonferroni corrected) ---")
    positions = sorted(df["position"].unique())
    n_comparisons = len(positions) * (len(positions) - 1) // 2
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            a = df.loc[df["position"] == positions[i], metric].dropna().values
            b = df.loc[df["position"] == positions[j], metric].dropna().values
            if len(a) > 0 and len(b) > 0:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                p_adj = min(p * max(n_comparisons, 1), 1.0)
                d = cohens_d(a, b)
                lines.append(f"    {positions[i]} vs {positions[j]}: U={u:.1f}, p={p:.2e}, "
                             f"p_adj={p_adj:.2e}, Cohen's d={d:.4f}")
    lines.append("")

    # Context length pairwise
    if "context_length" in df.columns:
        lines.append("  --- Context Length Pairwise (Mann-Whitney U) ---")
        cls = sorted(df["context_length"].unique())
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                a = df.loc[df["context_length"] == cls[i], metric].dropna().values
                b = df.loc[df["context_length"] == cls[j], metric].dropna().values
                if len(a) > 0 and len(b) > 0:
                    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    d = cohens_d(a, b)
                    lines.append(f"    L={int(cls[i])} vs L={int(cls[j])}: U={u:.1f}, p={p:.2e}, "
                                 f"Cohen's d={d:.4f}")
        lines.append("")

    return "\n".join(lines)


def generate_interaction_analysis(df: pd.DataFrame, metric: str, label: str) -> str:
    """Analyze interactions between experimental factors."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  INTERACTION ANALYSIS ({metric}) — {label}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    if metric not in df.columns or df[metric].dropna().empty:
        lines.append(f"  Metric '{metric}' not available.")
        return "\n".join(lines)

    df2 = add_depth_group(df)

    # Position effect within each depth group
    lines.append("  --- Position Effect within Depth Groups (Kruskal-Wallis) ---")
    for depth in ["Shallow", "Middle", "Deep"]:
        subset = df2[df2["depth_group"] == depth]
        groups = [g[metric].dropna().values for _, g in subset.groupby("position")]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h, p = stats.kruskal(*groups)
            lines.append(f"    {depth}: H={h:.4f}, p={p:.2e}")
            for pos_name, g_data in subset.groupby("position"):
                s = g_data[metric].dropna()
                lines.append(f"      {pos_name}: mean={s.mean():.6f}, n={len(s)}")
    lines.append("")

    # Ratio effect within each depth group
    lines.append("  --- Ratio Effect within Depth Groups (Kruskal-Wallis) ---")
    for depth in ["Shallow", "Middle", "Deep"]:
        subset = df2[df2["depth_group"] == depth]
        groups = [g[metric].dropna().values for _, g in subset.groupby("perturbation_ratio")]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h, p = stats.kruskal(*groups)
            lines.append(f"    {depth}: H={h:.4f}, p={p:.2e}")
            for ratio_name, g_data in subset.groupby("perturbation_ratio"):
                s = g_data[metric].dropna()
                lines.append(f"      ratio={ratio_name}: mean={s.mean():.6f}, n={len(s)}")
    lines.append("")

    # Ratio effect within each position
    lines.append("  --- Ratio Effect within Positions (Kruskal-Wallis) ---")
    for pos in sorted(df["position"].unique()):
        subset = df[df["position"] == pos]
        groups = [g[metric].dropna().values for _, g in subset.groupby("perturbation_ratio")]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h, p = stats.kruskal(*groups)
            lines.append(f"    {pos}: H={h:.4f}, p={p:.2e}")
    lines.append("")

    # Context length effect within depth groups
    if "context_length" in df.columns:
        lines.append("  --- Context Length Effect within Depth Groups (Spearman) ---")
        for depth in ["Shallow", "Middle", "Deep"]:
            subset = df2[df2["depth_group"] == depth]
            valid = subset[[metric, "context_length"]].dropna()
            if len(valid) > 2:
                rho, p = stats.spearmanr(valid["context_length"], valid[metric])
                lines.append(f"    {depth}: rho={rho:.6f}, p={p:.2e}")
        lines.append("")

    return "\n".join(lines)


def generate_metric_correlation_analysis(df: pd.DataFrame, label: str) -> str:
    """Analyze correlations between different metrics."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  METRIC CORRELATION ANALYSIS — {label}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    metric_pairs = [
        ("cosine_mean", "l2_mean"),
        ("cosine_mean", "cosine_segment"),
        ("cosine_mean", "cka"),
        ("cosine_segment", "cka"),
        ("l2_mean", "cka"),
    ]

    for m1, m2 in metric_pairs:
        if m1 in df.columns and m2 in df.columns:
            valid = df[[m1, m2]].dropna()
            if len(valid) > 2:
                rho_s, p_s = stats.spearmanr(valid[m1], valid[m2])
                rho_p, p_p = stats.pearsonr(valid[m1], valid[m2])
                lines.append(f"  {m1} vs {m2} (n={len(valid)}):")
                lines.append(f"    Pearson:  r={rho_p:.6f}, p={p_p:.2e}")
                lines.append(f"    Spearman: rho={rho_s:.6f}, p={p_s:.2e}")
                lines.append("")

    return "\n".join(lines)


def generate_extreme_cases(df: pd.DataFrame, metric: str, label: str, n: int = 10) -> str:
    """Identify extreme cases — highest and lowest metric values."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  EXTREME CASES ({metric}) — {label}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    if metric not in df.columns or df[metric].dropna().empty:
        lines.append(f"  Metric '{metric}' not available.")
        return "\n".join(lines)

    valid = df.dropna(subset=[metric]).copy()

    # Lowest (most divergent)
    lines.append(f"  --- Top {n} LOWEST {metric} (most divergent) ---")
    lowest = valid.nsmallest(n, metric)
    cols = ["experiment_id", "layer_index", "perturbation_ratio", "position", metric]
    if "context_length" in lowest.columns:
        cols.insert(1, "context_length")
    for _, row in lowest.iterrows():
        parts = [f"{c}={row[c]}" for c in cols if c in row.index]
        lines.append(f"    {', '.join(parts)}")
    lines.append("")

    # Highest (most similar)
    lines.append(f"  --- Top {n} HIGHEST {metric} (most preserved) ---")
    highest = valid.nlargest(n, metric)
    for _, row in highest.iterrows():
        parts = [f"{c}={row[c]}" for c in cols if c in row.index]
        lines.append(f"    {', '.join(parts)}")
    lines.append("")

    return "\n".join(lines)


def generate_sanity_check_analysis(result_dir: Path) -> str:
    """Analyze sanity check results."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  SANITY CHECK ANALYSIS")
    lines.append(f"{'=' * 70}")
    lines.append("")

    sanity_dir = result_dir / "sanity"
    if not sanity_dir.exists():
        lines.append("  No sanity check directory found.")
        return "\n".join(lines)

    for fname in ["identity_metrics.csv", "minimal_metrics.csv"]:
        path = sanity_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            lines.append(f"  --- {fname} ({len(df)} rows) ---")
            for col in ["cosine_mean", "cosine_std", "cosine_segment", "cka"]:
                if col in df.columns:
                    s = df[col].dropna()
                    if len(s) > 0:
                        lines.append(f"    {col}: mean={s.mean():.8f}, min={s.min():.8f}, max={s.max():.8f}")
            # Per layer
            if "cosine_mean" in df.columns:
                lines.append(f"    Per-layer cosine_mean:")
                for _, row in df.iterrows():
                    lines.append(f"      layer {int(row['layer_index']):>3}: "
                                 f"cosine_mean={row['cosine_mean']:.8f}")
            lines.append("")

    return "\n".join(lines)


def generate_comprehensive_txt_reports(result_dir: Path, output_dir: Path):
    """Generate all comprehensive txt analysis files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    sanity_txt = generate_sanity_check_analysis(result_dir)
    (output_dir / "01_sanity_check.txt").write_text(sanity_txt)
    print("  Written: 01_sanity_check.txt")

    for phase_num, phase_label in [(2, "Phase 2 (Type1: Content Replacement)"),
                                    (3, "Phase 3 (Type2: Semantic Paraphrasing)")]:
        df = load_phase_data(result_dir, phase_num)
        if df is None:
            continue

        prefix = f"phase{phase_num}"

        # Determine primary metric
        has_cosine_mean = "cosine_mean" in df.columns and df["cosine_mean"].dropna().any()
        primary_metric = "cosine_mean" if has_cosine_mean else "cosine_segment"
        secondary_metrics = []
        for m in ["cosine_segment", "cka", "l2_mean"]:
            if m in df.columns and df[m].dropna().any() and m != primary_metric:
                secondary_metrics.append(m)

        # 1. Descriptive stats
        txt = generate_descriptive_stats(df, phase_label)
        fname = f"02_{prefix}_descriptive_stats.txt"
        (output_dir / fname).write_text(txt)
        print(f"  Written: {fname}")

        # 2. Per-layer analysis for each metric
        for metric in [primary_metric] + secondary_metrics:
            txt = generate_per_layer_analysis(df, metric, phase_label)
            fname = f"03_{prefix}_per_layer_{metric}.txt"
            (output_dir / fname).write_text(txt)
            print(f"  Written: {fname}")

        # 3. Per-condition analysis
        txt = generate_per_condition_analysis(df, primary_metric, phase_label)
        fname = f"04_{prefix}_per_condition.txt"
        (output_dir / fname).write_text(txt)
        print(f"  Written: {fname}")

        # 4. Pairwise tests
        txt = generate_pairwise_tests(df, primary_metric, phase_label)
        fname = f"05_{prefix}_pairwise_tests.txt"
        (output_dir / fname).write_text(txt)
        print(f"  Written: {fname}")

        # 5. Interaction analysis
        txt = generate_interaction_analysis(df, primary_metric, phase_label)
        fname = f"06_{prefix}_interactions.txt"
        (output_dir / fname).write_text(txt)
        print(f"  Written: {fname}")

        # 6. Metric correlations
        txt = generate_metric_correlation_analysis(df, phase_label)
        fname = f"07_{prefix}_metric_correlations.txt"
        (output_dir / fname).write_text(txt)
        print(f"  Written: {fname}")

        # 7. Extreme cases
        txt = generate_extreme_cases(df, primary_metric, phase_label)
        fname = f"08_{prefix}_extreme_cases.txt"
        (output_dir / fname).write_text(txt)
        print(f"  Written: {fname}")

        # 8. Hypothesis tests (only for phase2 with cosine_mean)
        if has_cosine_mean:
            results = [
                test_h1_layer_divergence(df),
                test_h2_ratio_scaling(df),
                test_h3_position_effect(df),
            ]
            verdict = generate_verdict(results)
            fname = f"09_{prefix}_hypothesis_verdict.txt"
            (output_dir / fname).write_text(verdict)
            print(f"  Written: {fname}")

    print(f"\nAll txt reports written to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and validate research hypotheses."
    )
    parser.add_argument("result_dir", type=Path,
                        help="Path to experiment result directory (e.g., results/auto_test_XXXX/)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for report (default: {result_dir}/analysis_report)")
    parser.add_argument("--phase", type=int, default=None, choices=[2, 3],
                        help="Load only a specific phase (2=Type1, 3=Type2)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        logger.error(f"Result directory not found: {result_dir}")
        sys.exit(1)

    output_dir = args.output_dir or result_dir / "analysis_report"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comprehensive txt reports (per-phase)
    print("\n=== Generating comprehensive text analysis reports ===\n")
    generate_comprehensive_txt_reports(result_dir, output_dir)

    # Generate plots for phase2 (which has cosine_mean)
    df_p2 = load_phase_data(result_dir, 2)
    if df_p2 is not None and "cosine_mean" in df_p2.columns and df_p2["cosine_mean"].dropna().any():
        print("\n=== Generating plots (Phase 2) ===\n")
        plots_dir = output_dir / "plots"
        generate_all_analysis_plots(df_p2, plots_dir)
        print(f"  Plots saved to {plots_dir}/")

    print(f"\nFull report saved to {output_dir}/")


if __name__ == "__main__":
    main()

