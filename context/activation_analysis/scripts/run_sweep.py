#!/usr/bin/env python3
"""CLI: Run a full parameter sweep over the experimental grid."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_loader import load_model_and_tokenizer
from src.batch_runner import run_sweep
from src.visualization import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full parameter sweep for activation analysis")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs for explicit placement, e.g. '0,1'"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    # Parse explicit GPU IDs if provided
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model, gpu_ids=gpu_ids)

    logger.info("Starting parameter sweep...")
    df = run_sweep(model, tokenizer, config)

    if df.empty:
        logger.warning("No results collected from sweep")
        sys.exit(1)

    logger.info(f"Sweep complete: {len(df)} metric rows collected")

    if not args.no_plots:
        logger.info("Generating plots...")
        # Find the sweep directory (most recent results subdirectory)
        results_dir = Path(config.output_dir)
        sweep_dirs = sorted(results_dir.glob("sweep_*"))
        if sweep_dirs:
            generate_all_plots(df, sweep_dirs[-1])
            logger.info(f"Plots saved to {sweep_dirs[-1] / 'plots'}")
        else:
            generate_all_plots(df, results_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
