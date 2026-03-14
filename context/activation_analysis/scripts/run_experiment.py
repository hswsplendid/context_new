#!/usr/bin/env python3
"""CLI: Run a single activation analysis experiment."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_loader import load_model_and_tokenizer
from src.prompt_builder import load_template, load_replacements
from src.experiment_runner import run_single_experiment
from src.storage import save_metrics, save_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a single activation analysis experiment")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument("--context-length", type=int, default=1024, help="Target context length in tokens")
    parser.add_argument("--ratio", type=float, default=0.25, help="Perturbation ratio")
    parser.add_argument("--position", type=str, default="middle", choices=["beginning", "middle", "end"])
    parser.add_argument("--pair-index", type=int, default=0, help="Index into replacement pairs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--experiment-id", type=str, default="single_exp", help="Experiment identifier")
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs for explicit placement, e.g. '0,1'"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    output_path = Path(config.output_dir) / args.experiment_id
    output_path.mkdir(parents=True, exist_ok=True)

    save_config(config, output_path / "config.yaml")

    # Parse explicit GPU IDs if provided
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model, gpu_ids=gpu_ids)

    template = load_template(config.template_path)
    replacements = load_replacements(config.replacements_path)

    if args.pair_index >= len(replacements):
        logger.error(f"Pair index {args.pair_index} out of range (have {len(replacements)} pairs)")
        sys.exit(1)

    pair = replacements[args.pair_index]

    result = run_single_experiment(
        model=model,
        tokenizer=tokenizer,
        config=config,
        template=template,
        context_length=args.context_length,
        perturbation_ratio=args.ratio,
        position=args.position,
        original_segment=pair["original"],
        replacement_segment=pair["replacement"],
        experiment_id=args.experiment_id,
    )

    if result.error:
        logger.error(f"Experiment failed: {result.error}")
        sys.exit(1)

    metrics_path = output_path / "metrics.csv"
    save_metrics(result, metrics_path)

    logger.info(f"Results saved to {output_path}")
    for lm in result.layer_metrics:
        cos_str = f"cosine_mean={lm.cosine_mean:.4f}" if lm.cosine_mean is not None else "N/A"
        logger.info(f"  Layer {lm.layer_index:3d}: {cos_str}")


if __name__ == "__main__":
    main()
