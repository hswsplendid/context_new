"""Parameter sweep over experimental grid."""

import logging
import uuid
from itertools import product
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import ExperimentConfig
from .prompt_builder import PromptTemplate, load_template, load_replacements
from .paraphrase_generator import batch_generate_paraphrases
from .experiment_runner import ExperimentResult, run_single_experiment
from .storage import save_metrics, save_config

logger = logging.getLogger(__name__)


def run_sweep(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Run a full parameter sweep over the experimental grid.

    Iterates over: context_lengths x perturbation_ratios x positions x perturbation_pairs.

    Args:
        model: Loaded model (reused across all experiments).
        tokenizer: Model tokenizer.
        config: Full experiment configuration.

    Returns:
        DataFrame with all collected metrics.
    """
    output_dir = Path(config.output_dir)
    sweep_id = uuid.uuid4().hex[:8]
    sweep_dir = output_dir / f"sweep_{sweep_id}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    save_config(config, sweep_dir / "config.yaml")

    # Load template
    template = load_template(config.template_path)

    # Load perturbation pairs
    if config.perturbation.type == "type1":
        replacements = load_replacements(config.replacements_path)
        pairs = [(r["original"], r["replacement"]) for r in replacements]
    else:
        # Type 2: generate paraphrases
        replacements = load_replacements(config.replacements_path)
        originals = [r["original"] for r in replacements]
        cache_path = sweep_dir / "paraphrase_cache.yaml"
        paraphrases = batch_generate_paraphrases(
            model, tokenizer, originals, cache_path=cache_path
        )
        pairs = list(zip(originals, paraphrases))

    # Build experiment grid
    grid = list(product(
        config.perturbation.context_lengths,
        config.perturbation.ratios,
        config.perturbation.positions,
        range(len(pairs)),
    ))

    metrics_path = sweep_dir / "metrics.csv"
    all_results: list[ExperimentResult] = []

    logger.info(
        f"Starting sweep {sweep_id}: {len(grid)} experiments "
        f"({len(config.perturbation.context_lengths)} lengths x "
        f"{len(config.perturbation.ratios)} ratios x "
        f"{len(config.perturbation.positions)} positions x "
        f"{len(pairs)} pairs)"
    )

    for ctx_len, ratio, position, pair_idx in tqdm(grid, desc="Experiments"):
        exp_id = f"{sweep_id}_{ctx_len}_{ratio}_{position}_{pair_idx}"
        orig_seg, repl_seg = pairs[pair_idx]

        try:
            result = run_single_experiment(
                model=model,
                tokenizer=tokenizer,
                config=config,
                template=template,
                context_length=ctx_len,
                perturbation_ratio=ratio,
                position=position,
                original_segment=orig_seg,
                replacement_segment=repl_seg,
                experiment_id=exp_id,
            )

            if result.error:
                logger.warning(f"Experiment {exp_id} error: {result.error}")
            else:
                save_metrics(result, metrics_path)
                all_results.append(result)

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM in experiment {exp_id}, clearing cache and continuing")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.error(f"Unexpected error in {exp_id}: {e}")
            continue

    # Load and return aggregated metrics
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        logger.info(
            f"Sweep {sweep_id} complete: {len(all_results)}/{len(grid)} experiments succeeded, "
            f"{len(df)} metric rows saved"
        )
        return df
    else:
        logger.warning("No metrics were saved during sweep")
        return pd.DataFrame()
