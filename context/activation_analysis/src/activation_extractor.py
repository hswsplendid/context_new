"""Hook-based selective hidden state extraction from transformer layers."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ActivationResult:
    """Container for extracted hidden states.

    activations: dict mapping layer_index -> tensor of shape (num_tokens, hidden_size)
    token_indices: the token positions that were extracted
    """
    activations: dict[int, torch.Tensor] = field(default_factory=dict)
    token_indices: list[int] = field(default_factory=list)
    num_layers: int = 0
    num_tokens: int = 0
    hidden_size: int = 0


def _make_hook(layer_idx: int, token_indices: list[int], storage: dict, dtype: torch.dtype):
    """Create a forward hook that extracts hidden states for specific tokens.

    The hook captures output[0] (hidden states), selects only the tokens of interest,
    converts to the target dtype, moves to CPU, and clones to avoid GPU memory retention.
    """
    idx_tensor = torch.tensor(token_indices, dtype=torch.long)

    def hook_fn(module, input, output):
        # output is a tuple; output[0] is the hidden states tensor: (batch, seq_len, hidden_size)
        hidden = output[0]
        # Select only tokens of interest
        selected = hidden[0, idx_tensor.to(hidden.device), :].to(dtype=dtype, device="cpu").clone()
        storage[layer_idx] = selected

    return hook_fn


def extract_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    token_ids: list[int],
    token_indices: list[int],
    layer_indices: list[int],
    storage_dtype: torch.dtype = torch.float32,
) -> ActivationResult:
    """Extract hidden states at specified layers and token positions using hooks.

    Args:
        model: The loaded model (eval mode, multi-GPU).
        tokenizer: Model tokenizer.
        token_ids: Full input token ID sequence.
        token_indices: Which token positions to extract (0-indexed).
        layer_indices: Which layers to hook (e.g., [0, 4, 8, ...]).
        storage_dtype: dtype for stored activations (float32 for metric precision).

    Returns:
        ActivationResult with activations dict keyed by layer index.
    """
    if not token_indices:
        return ActivationResult()

    storage: dict[int, torch.Tensor] = {}
    handles = []

    # Register hooks on target layers
    try:
        layers = model.model.layers
    except AttributeError:
        # Fallback for different model architectures
        try:
            layers = model.transformer.h
        except AttributeError:
            raise RuntimeError(
                "Cannot find transformer layers. Expected model.model.layers or model.transformer.h"
            )

    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            logger.warning(
                f"Layer {layer_idx} out of range (model has {len(layers)} layers), skipping"
            )
            continue
        hook = _make_hook(layer_idx, token_indices, storage, storage_dtype)
        handle = layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)

    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    # Move to the device of the first model parameter
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)

    # Forward pass
    try:
        with torch.inference_mode():
            model(input_ids)
    finally:
        # Always clean up hooks
        for handle in handles:
            handle.remove()

    hidden_size = 0
    if storage:
        first_tensor = next(iter(storage.values()))
        hidden_size = first_tensor.shape[-1]

    return ActivationResult(
        activations=storage,
        token_indices=token_indices,
        num_layers=len(storage),
        num_tokens=len(token_indices),
        hidden_size=hidden_size,
    )


def extract_pair_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    original_ids: list[int],
    perturbed_ids: list[int],
    original_indices: list[int],
    perturbed_indices: list[int],
    layer_indices: list[int],
    storage_dtype: torch.dtype = torch.float32,
) -> tuple[ActivationResult, ActivationResult]:
    """Extract activations for both original and perturbed prompts.

    Runs two sequential forward passes (not batched, to avoid padding complications
    with different-length sequences).
    """
    logger.info(
        f"Extracting activations: {len(original_indices)} original tokens, "
        f"{len(perturbed_indices)} perturbed tokens, {len(layer_indices)} layers"
    )

    original_result = extract_activations(
        model, tokenizer, original_ids, original_indices, layer_indices, storage_dtype
    )

    perturbed_result = extract_activations(
        model, tokenizer, perturbed_ids, perturbed_indices, layer_indices, storage_dtype
    )

    return original_result, perturbed_result
