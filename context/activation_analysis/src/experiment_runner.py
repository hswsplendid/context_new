"""Single experiment orchestration pipeline."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import ExperimentConfig
from .prompt_builder import PromptPair, PromptTemplate, build_prompt_pair
from .token_aligner import AlignmentMap, compute_alignment
from .activation_extractor import extract_pair_activations
from .metrics import LayerMetrics, compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single experiment trial."""
    experiment_id: str
    context_length: int
    perturbation_ratio: float
    position: str
    perturbation_type: str
    layer_metrics: list[LayerMetrics] = field(default_factory=list)
    alignment_info: Optional[AlignmentMap] = None
    prompt_pair: Optional[PromptPair] = None
    error: Optional[str] = None


def run_single_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ExperimentConfig,
    template: PromptTemplate,
    context_length: int,
    perturbation_ratio: float,
    position: str,
    original_segment: str,
    replacement_segment: str,
    experiment_id: str = "",
) -> ExperimentResult:
    """Run a single perturbation experiment.

    Pipeline:
    1. Build PromptPair from template + perturbation config
    2. Compute AlignmentMap via token_aligner
    3. Extract activations for original prompt (hooks)
    4. Extract activations for perturbed prompt (hooks)
    5. Compute metrics across aligned tokens at each target layer
    6. Return ExperimentResult
    """
    result = ExperimentResult(
        experiment_id=experiment_id,
        context_length=context_length,
        perturbation_ratio=perturbation_ratio,
        position=position,
        perturbation_type=config.perturbation.type,
    )

    try:
        # 1. Build prompt pair
        logger.info(
            f"Building prompt pair: length={context_length}, ratio={perturbation_ratio}, "
            f"position={position}, type={config.perturbation.type}"
        )
        prompt_pair = build_prompt_pair(
            template=template,
            tokenizer=tokenizer,
            perturbation_type=config.perturbation.type,
            target_length=context_length,
            perturbation_ratio=perturbation_ratio,
            position=position,
            original_segment=original_segment,
            replacement_segment=replacement_segment,
        )
        result.prompt_pair = prompt_pair

        # 2. Compute alignment
        alignment = compute_alignment(prompt_pair)
        result.alignment_info = alignment
        logger.info(
            f"Alignment: type={alignment.alignment_type}, "
            f"aligned={alignment.num_aligned}, trimmed={alignment.num_trimmed}"
        )

        # 3 & 4. Determine which token indices to extract
        if config.perturbation.type == "type1":
            if alignment.num_aligned == 0:
                logger.warning("No aligned tokens for Type 1 comparison")
                result.error = "No aligned tokens"
                return result
            orig_indices = alignment.original_indices
            pert_indices = alignment.perturbed_indices
            token_level = "token" in config.metrics.granularity
            segment_level = "segment" in config.metrics.granularity
        else:
            # Type 2: extract segment tokens
            orig_range = alignment.original_segment_range
            pert_range = alignment.perturbed_segment_range
            orig_indices = list(range(orig_range[0], orig_range[1]))
            pert_indices = list(range(pert_range[0], pert_range[1]))
            token_level = False  # Different tokens, no token-level comparison
            segment_level = True

        # Extract activations
        orig_result, pert_result = extract_pair_activations(
            model=model,
            tokenizer=tokenizer,
            original_ids=prompt_pair.original_token_ids,
            perturbed_ids=prompt_pair.perturbed_token_ids,
            original_indices=orig_indices,
            perturbed_indices=pert_indices,
            layer_indices=config.extraction.layer_indices,
        )

        # 5. Compute metrics
        layer_metrics = compute_all_metrics(
            original_activations=orig_result.activations,
            perturbed_activations=pert_result.activations,
            compute_cosine=config.metrics.cosine,
            compute_l2=config.metrics.l2,
            compute_cka=config.metrics.cka,
            token_level=token_level,
            segment_level=segment_level,
        )
        result.layer_metrics = layer_metrics

        logger.info(
            f"Experiment {experiment_id} complete: {len(layer_metrics)} layers analyzed"
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result.error = "CUDA OOM"
        logger.error(f"Experiment {experiment_id}: CUDA OOM")
    except Exception as e:
        result.error = str(e)
        logger.error(f"Experiment {experiment_id} failed: {e}")

    return result
