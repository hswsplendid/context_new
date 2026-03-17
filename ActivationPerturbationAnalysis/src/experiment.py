"""
Single-experiment runner.

Orchestrates the full pipeline for one (prompt, perturbation) pair:
  1. Tokenize the original prompt.
  2. Apply a perturbation.
  3. Run both sequences through the model.
  4. Compute distance-bucketed similarity metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from .metrics import compute_distance_buckets
from .model import extract_hidden_states
from .perturbations import (
    PerturbationResult,
    TripletResult,
    build_prompt_diff_result,
    build_triplet_result,
    get_strategy,
)

logger = logging.getLogger(__name__)


def run_single_experiment(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    perturbation_type: str,
    perturbation_position_frac: float,
    perturbation_span_length: int,
    measurement_cfg: Dict[str, int],
    seed: int = 42,
    replacement_pool: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Execute a single perturbation experiment and return metric records.

    Parameters
    ----------
    model:
        HuggingFace causal LM with ``output_hidden_states=True``.
    tokenizer:
        Corresponding tokenizer.
    prompt_text:
        The original prompt (already at the desired token length).
    perturbation_type:
        Name of the perturbation strategy (must be in the registry).
    perturbation_position_frac:
        Where in the sequence the perturbation starts (0.0–1.0).
    perturbation_span_length:
        Number of tokens in the perturbed span.
    measurement_cfg:
        Dict with keys ``immediate_window``, ``short_window``,
        ``periodic_sampling_step``, ``periodic_sampling_width``.
    seed:
        Random seed for the perturbation strategy.
    replacement_pool:
        Optional token-id pool for ``semantic_change`` strategy.

    Returns
    -------
    List of metric records (dicts) ready for storage.
    """
    # 1. Tokenize -------------------------------------------------------
    original_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    seq_len = len(original_ids)

    # 2. Compute perturbation span bounds --------------------------------
    span_start = int(perturbation_position_frac * seq_len)
    span_len = min(perturbation_span_length, seq_len - span_start)
    if span_len <= 0:
        logger.warning(
            "Perturbation span has zero length (start=%d, seq_len=%d). Skipping.",
            span_start,
            seq_len,
        )
        return []

    # 3. Apply perturbation ----------------------------------------------
    kwargs = {}
    if perturbation_type == "semantic_change" and replacement_pool:
        kwargs["replacement_pool"] = replacement_pool
    strategy = get_strategy(perturbation_type, tokenizer, seed=seed, **kwargs)
    result: PerturbationResult = strategy.apply(original_ids, span_start, span_len)

    logger.debug(
        "Perturbation '%s': orig len=%d, pert len=%d, post_start_orig=%d, post_start_pert=%d",
        perturbation_type,
        len(result.original_ids),
        len(result.perturbed_ids),
        result.original_span_end,
        result.post_perturbation_start,
    )

    # 4. Forward passes (hidden states stay on GPU) ----------------------
    logger.debug("Running forward pass on original (%d tokens)…", len(result.original_ids))
    orig_hidden = extract_hidden_states(model, result.original_ids)

    logger.debug("Running forward pass on perturbed (%d tokens)…", len(result.perturbed_ids))
    pert_hidden = extract_hidden_states(model, result.perturbed_ids)

    # 5. Compute metrics (vectorized on GPU, only scalars come back) -----
    records = compute_distance_buckets(
        original_states=orig_hidden,
        perturbed_states=pert_hidden,
        post_perturbation_start_orig=result.original_span_end,
        post_perturbation_start_pert=result.post_perturbation_start,
        immediate_window=measurement_cfg.get("immediate_window", 10),
        short_window=measurement_cfg.get("short_window", 100),
        periodic_step=measurement_cfg.get("periodic_sampling_step", 100),
        periodic_width=measurement_cfg.get("periodic_sampling_width", 10),
    )

    # Free GPU memory eagerly.
    del orig_hidden, pert_hidden
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return records


def run_prompt_diff_experiment(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prev_prompt: str,
    curr_prompt: str,
    measurement_cfg: Dict[str, int],
    prompt_pair_index: int,
) -> List[Dict[str, Any]]:
    """Measure forward propagation of natural prompt diffs.

    Treats the natural difference between two consecutive agent prompts as
    the perturbation and measures — in the same forward direction as the
    standard experiments — how the diff affects subsequent token hidden
    states.

    From the divergence point (LCP) onward, the two sequences contain
    *different* token ids.  We compare their hidden states at aligned
    offsets from the divergence point using the standard
    :func:`compute_distance_buckets`.  This captures how divergent content
    causes the hidden-state trajectory to separate as we move further
    from the split.

    Parameters
    ----------
    model:
        HuggingFace causal LM with ``output_hidden_states=True``.
    tokenizer:
        Corresponding tokenizer.
    prev_prompt, curr_prompt:
        Two consecutive prompts (prev is typically a prefix of curr).
    measurement_cfg:
        Dict with keys ``immediate_window``, ``short_window``,
        ``periodic_sampling_step``, ``periodic_sampling_width``.
    prompt_pair_index:
        Index of this consecutive pair (0 = prompts 0→1, etc.).

    Returns
    -------
    List of metric records (dicts) ready for storage.
    """
    # 1. Tokenize both prompts ------------------------------------------
    prev_ids = tokenizer.encode(prev_prompt, add_special_tokens=False)
    curr_ids = tokenizer.encode(curr_prompt, add_special_tokens=False)

    # 2. Find common prefix (LCP) --------------------------------------
    result = build_prompt_diff_result(prev_ids, curr_ids)
    lcp = result.post_perturbation_start  # longest common prefix length

    tokens_after_prev = len(prev_ids) - lcp
    tokens_after_curr = len(curr_ids) - lcp

    if min(tokens_after_prev, tokens_after_curr) <= 0:
        logger.warning(
            "Prompt pair %d: one sequence has no tokens after the divergence "
            "point (LCP=%d, prev=%d, curr=%d). "
            "No forward-propagation measurement possible. Skipping.",
            prompt_pair_index, lcp, len(prev_ids), len(curr_ids),
        )
        return []

    logger.debug(
        "Prompt diff pair %d: prev=%d tokens, curr=%d tokens, LCP=%d, "
        "post-LCP prev=%d, post-LCP curr=%d",
        prompt_pair_index,
        len(prev_ids), len(curr_ids), lcp,
        tokens_after_prev, tokens_after_curr,
    )

    # 3. Forward passes (hidden states stay on GPU) ---------------------
    logger.debug("Running forward pass on prev (%d tokens)…", len(prev_ids))
    prev_hidden = extract_hidden_states(model, prev_ids)

    logger.debug("Running forward pass on curr (%d tokens)…", len(curr_ids))
    curr_hidden = extract_hidden_states(model, curr_ids)

    # 4. Compute forward-direction metrics from the divergence point ----
    #    Identical to the standard experiment: compare hidden states at
    #    aligned offsets from the point where the two sequences diverge.
    records = compute_distance_buckets(
        original_states=prev_hidden,
        perturbed_states=curr_hidden,
        post_perturbation_start_orig=lcp,
        post_perturbation_start_pert=lcp,
        immediate_window=measurement_cfg.get("immediate_window", 10),
        short_window=measurement_cfg.get("short_window", 100),
        periodic_step=measurement_cfg.get("periodic_sampling_step", 100),
        periodic_width=measurement_cfg.get("periodic_sampling_width", 10),
    )

    # Free GPU memory eagerly.
    del prev_hidden, curr_hidden
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return records


def run_triplet_experiment(
    model,
    tokenizer: PreTrainedTokenizerBase,
    p_k_text: str,
    p_k1_text: str,
    p_k2_text: str,
    measurement_cfg: Dict[str, int],
    triplet_index: int,
) -> List[Dict[str, Any]]:
    """Measure forward propagation using the triplet comparison method.

    For three consecutive prefix-chain prompts P_k ⊂ P_{k+1} ⊂ P_{k+2}:
      - P_k     = [prefix]
      - P_{k+1} = [prefix][D1]
      - P_{k+2} = [prefix][D1][D2]

    Constructs:
      - seq_full = [prefix][D1][D2]   (natural sequence, D1 present)
      - seq_skip = [prefix][D2]        (D1 removed)

    D1 is the "perturbation".  We measure how D1's presence/absence affects
    hidden states in the D2 region (forward propagation through D2).

    Parameters
    ----------
    model:
        HuggingFace causal LM with ``output_hidden_states=True``.
    tokenizer:
        Corresponding tokenizer.
    p_k_text, p_k1_text, p_k2_text:
        Three consecutive prompts forming a prefix chain.
    measurement_cfg:
        Dict with keys ``immediate_window``, ``short_window``,
        ``periodic_sampling_step``, ``periodic_sampling_width``.
    triplet_index:
        Index of this triplet (0 = prompts 0→1→2, etc.).

    Returns
    -------
    List of metric records (dicts) ready for storage.
    """
    # 1. Tokenize all three prompts -----------------------------------------
    p_k_ids = tokenizer.encode(p_k_text, add_special_tokens=False)
    p_k1_ids = tokenizer.encode(p_k1_text, add_special_tokens=False)
    p_k2_ids = tokenizer.encode(p_k2_text, add_special_tokens=False)

    # 2. Build triplet -------------------------------------------------------
    triplet = build_triplet_result(p_k_ids, p_k1_ids, p_k2_ids)
    if triplet is None:
        logger.warning(
            "Triplet %d: build_triplet_result returned None. Skipping.",
            triplet_index,
        )
        return []

    logger.debug(
        "Triplet %d: prefix=%d, D1=%d, D2=%d, "
        "seq_full=%d tokens, seq_skip=%d tokens",
        triplet_index,
        triplet.prefix_len,
        triplet.d1_len,
        triplet.d2_len,
        len(triplet.seq_full_ids),
        len(triplet.seq_skip_ids),
    )

    # 3. Forward passes -------------------------------------------------------
    logger.debug(
        "Running forward pass on seq_full (%d tokens)…",
        len(triplet.seq_full_ids),
    )
    full_hidden = extract_hidden_states(model, triplet.seq_full_ids)

    logger.debug(
        "Running forward pass on seq_skip (%d tokens)…",
        len(triplet.seq_skip_ids),
    )
    skip_hidden = extract_hidden_states(model, triplet.seq_skip_ids)

    # 4. Compute forward metrics over D2 region --------------------------------
    #    In seq_full, D2 starts at d2_start_full = prefix_len + d1_len
    #    In seq_skip, D2 starts at d2_start_skip = prefix_len
    #    compute_distance_buckets aligns from these start points forward.
    records = compute_distance_buckets(
        original_states=full_hidden,
        perturbed_states=skip_hidden,
        post_perturbation_start_orig=triplet.d2_start_full,
        post_perturbation_start_pert=triplet.d2_start_skip,
        immediate_window=measurement_cfg.get("immediate_window", 10),
        short_window=measurement_cfg.get("short_window", 100),
        periodic_step=measurement_cfg.get("periodic_sampling_step", 100),
        periodic_width=measurement_cfg.get("periodic_sampling_width", 10),
    )

    # Free GPU memory eagerly.
    del full_hidden, skip_hidden
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return records
