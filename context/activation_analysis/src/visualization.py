"""Visualization: heatmaps, line plots, and bar charts for activation analysis."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend for server environments

logger = logging.getLogger(__name__)

# Style defaults
FIGSIZE_HEATMAP = (14, 8)
FIGSIZE_LINE = (12, 6)
FIGSIZE_BAR = (10, 6)
DPI = 150


def _setup_style():
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"


def plot_similarity_heatmap(
    df: pd.DataFrame,
    context_length: int,
    ratio: float,
    position: str,
    metric: str = "cosine_mean",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot similarity heatmap: X=layer index, Y=token position, color=similarity.

    Args:
        df: Metrics DataFrame with columns: layer_index, experiment_id, and the metric.
        context_length: Filter to this context length.
        ratio: Filter to this perturbation ratio.
        position: Filter to this position.
        metric: Column name for the color scale.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    _setup_style()

    subset = df[
        (df["context_length"] == context_length)
        & (df["perturbation_ratio"] == ratio)
        & (df["position"] == position)
    ].copy()

    if subset.empty:
        logger.warning(f"No data for heatmap: length={context_length}, ratio={ratio}, pos={position}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Pivot: rows = experiment pairs, columns = layers
    pivot = subset.pivot_table(
        index="experiment_id", columns="layer_index", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Experiment")
    ax.set_title(
        f"Activation Similarity Heatmap\n"
        f"Context Length={context_length}, Perturbation Ratio={ratio}, Position={position}"
    )

    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved heatmap to {output_path}")

    return fig


def plot_similarity_vs_depth(
    df: pd.DataFrame,
    group_by: str = "perturbation_ratio",
    metric: str = "cosine_mean",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Line plot: X=layer index, Y=mean similarity, grouped by a variable.

    Shows how similarity changes through layers for different conditions.
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    groups = df.groupby(group_by)
    for name, group in groups:
        layer_means = group.groupby("layer_index")[metric].mean()
        layer_stds = group.groupby("layer_index")[metric].std()
        ax.plot(layer_means.index, layer_means.values, marker="o", label=f"{group_by}={name}")
        ax.fill_between(
            layer_means.index,
            layer_means.values - layer_stds.values,
            layer_means.values + layer_stds.values,
            alpha=0.15,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Activation Similarity vs Layer Depth (grouped by {group_by})")
    ax.legend()
    ax.set_ylim(bottom=0)

    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved depth plot to {output_path}")

    return fig


def plot_shallow_vs_deep(
    df: pd.DataFrame,
    metric: str = "cosine_mean",
    group_by: str = "perturbation_ratio",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing shallow/middle/deep layer groups.

    Layers grouped: shallow(0-15), middle(16-31), deep(32-48).
    """
    _setup_style()

    def classify_depth(layer_idx):
        if layer_idx <= 15:
            return "Shallow (0-15)"
        elif layer_idx <= 31:
            return "Middle (16-31)"
        else:
            return "Deep (32-48)"

    df = df.copy()
    df["depth_group"] = df["layer_index"].apply(classify_depth)

    # Aggregate
    agg = df.groupby([group_by, "depth_group"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    depth_order = ["Shallow (0-15)", "Middle (16-31)", "Deep (32-48)"]
    sns.barplot(
        data=agg,
        x="depth_group",
        y=metric,
        hue=group_by,
        order=depth_order,
        ax=ax,
    )

    ax.set_xlabel("Layer Depth Group")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Shallow vs Middle vs Deep Layer Similarity (grouped by {group_by})")
    ax.legend(title=group_by)

    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved bar chart to {output_path}")

    return fig


def plot_position_effect(
    df: pd.DataFrame,
    metric: str = "cosine_mean",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Line plot: how perturbation position affects similarity at each layer."""
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for position in ["beginning", "middle", "end"]:
        subset = df[df["position"] == position]
        if subset.empty:
            continue
        layer_means = subset.groupby("layer_index")[metric].mean()
        layer_stds = subset.groupby("layer_index")[metric].std()
        ax.plot(layer_means.index, layer_means.values, marker="s", label=f"Position: {position}")
        ax.fill_between(
            layer_means.index,
            layer_means.values - layer_stds.values,
            layer_means.values + layer_stds.values,
            alpha=0.15,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Effect of Perturbation Position on Activation Similarity")
    ax.legend()
    ax.set_ylim(bottom=0)

    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved position effect plot to {output_path}")

    return fig


def plot_context_length_effect(
    df: pd.DataFrame,
    metric: str = "cosine_mean",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Line plot: how context length affects similarity at each layer."""
    _setup_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for ctx_len in sorted(df["context_length"].unique()):
        subset = df[df["context_length"] == ctx_len]
        layer_means = subset.groupby("layer_index")[metric].mean()
        layer_stds = subset.groupby("layer_index")[metric].std()
        ax.plot(
            layer_means.index,
            layer_means.values,
            marker="^",
            label=f"Context Length: {ctx_len}",
        )
        ax.fill_between(
            layer_means.index,
            layer_means.values - layer_stds.values,
            layer_means.values + layer_stds.values,
            alpha=0.15,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Effect of Context Length on Activation Similarity")
    ax.legend()
    ax.set_ylim(bottom=0)

    if output_path:
        fig.savefig(output_path)
        logger.info(f"Saved context length plot to {output_path}")

    return fig


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
    metric: str = "cosine_mean",
) -> None:
    """Generate all standard visualization plots from a metrics DataFrame.

    Saves plots to output_dir/plots/.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Similarity vs depth for different ratios
    plot_similarity_vs_depth(
        df, group_by="perturbation_ratio", metric=metric,
        output_path=plots_dir / "similarity_vs_depth_by_ratio.png",
    )

    # 2. Similarity vs depth for different context lengths
    plot_similarity_vs_depth(
        df, group_by="context_length", metric=metric,
        output_path=plots_dir / "similarity_vs_depth_by_length.png",
    )

    # 3. Shallow vs deep comparison
    plot_shallow_vs_deep(
        df, metric=metric, group_by="perturbation_ratio",
        output_path=plots_dir / "shallow_vs_deep.png",
    )

    # 4. Position effect
    plot_position_effect(
        df, metric=metric,
        output_path=plots_dir / "position_effect.png",
    )

    # 5. Context length effect
    plot_context_length_effect(
        df, metric=metric,
        output_path=plots_dir / "context_length_effect.png",
    )

    # 6. Heatmaps for each (context_length, ratio, position) combo
    for ctx_len in df["context_length"].unique():
        for ratio in df["perturbation_ratio"].unique():
            for pos in df["position"].unique():
                plot_similarity_heatmap(
                    df, ctx_len, ratio, pos, metric=metric,
                    output_path=plots_dir / f"heatmap_{ctx_len}_{ratio}_{pos}.png",
                )

    logger.info(f"All plots saved to {plots_dir}")
    plt.close("all")
