#!/usr/bin/env python3
"""CLI: Generate plots from saved experiment results."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage import load_metrics
from src.visualization import (
    generate_all_plots,
    plot_similarity_heatmap,
    plot_similarity_vs_depth,
    plot_shallow_vs_deep,
    plot_position_effect,
    plot_context_length_effect,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots from saved metrics")
    parser.add_argument(
        "metrics_path", type=str,
        help="Path to metrics CSV file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: same directory as metrics file)",
    )
    parser.add_argument(
        "--metric", type=str, default="cosine_mean",
        choices=["cosine_mean", "cosine_segment", "l2_mean", "cka"],
        help="Metric to plot",
    )
    parser.add_argument(
        "--plot-type", type=str, default="all",
        choices=["all", "heatmap", "depth", "bar", "position", "context"],
        help="Which plot type to generate",
    )
    parser.add_argument(
        "--context-length", type=int, default=None,
        help="Context length filter (for heatmap)",
    )
    parser.add_argument(
        "--ratio", type=float, default=None,
        help="Perturbation ratio filter (for heatmap)",
    )
    parser.add_argument(
        "--position", type=str, default=None,
        help="Position filter (for heatmap)",
    )

    args = parser.parse_args()

    df = load_metrics(args.metrics_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.metrics_path).parent
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_type == "all":
        generate_all_plots(df, output_dir, metric=args.metric)
    elif args.plot_type == "heatmap":
        ctx = args.context_length or int(df["context_length"].iloc[0])
        ratio = args.ratio or df["perturbation_ratio"].iloc[0]
        pos = args.position or df["position"].iloc[0]
        plot_similarity_heatmap(
            df, ctx, ratio, pos, metric=args.metric,
            output_path=plots_dir / f"heatmap_{ctx}_{ratio}_{pos}.png",
        )
    elif args.plot_type == "depth":
        plot_similarity_vs_depth(
            df, metric=args.metric,
            output_path=plots_dir / "similarity_vs_depth.png",
        )
    elif args.plot_type == "bar":
        plot_shallow_vs_deep(
            df, metric=args.metric,
            output_path=plots_dir / "shallow_vs_deep.png",
        )
    elif args.plot_type == "position":
        plot_position_effect(
            df, metric=args.metric,
            output_path=plots_dir / "position_effect.png",
        )
    elif args.plot_type == "context":
        plot_context_length_effect(
            df, metric=args.metric,
            output_path=plots_dir / "context_length_effect.png",
        )

    logger.info("Plotting complete!")


if __name__ == "__main__":
    main()
