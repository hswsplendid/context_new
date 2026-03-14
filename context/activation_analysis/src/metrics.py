"""Similarity metrics: cosine, L2, and CKA for comparing activations."""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LayerMetrics:
    """Metrics for a single layer comparison."""
    layer_index: int
    # Token-level (Type 1)
    cosine_per_token: Optional[torch.Tensor] = None  # (num_tokens,)
    l2_per_token: Optional[torch.Tensor] = None  # (num_tokens,)
    cosine_mean: Optional[float] = None
    cosine_std: Optional[float] = None
    l2_mean: Optional[float] = None
    l2_std: Optional[float] = None
    # Segment-level
    cosine_segment: Optional[float] = None
    cka: Optional[float] = None


def cosine_similarity_paired(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-token cosine similarity between paired activation tensors.

    Args:
        a, b: Tensors of shape (num_tokens, hidden_size).

    Returns:
        Tensor of shape (num_tokens,) with cosine similarity per token.
    """
    # Normalize along hidden dimension
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    # Element-wise dot product
    return (a_norm * b_norm).sum(dim=-1)


def l2_distance_paired(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-token L2 distance.

    Args:
        a, b: Tensors of shape (num_tokens, hidden_size).

    Returns:
        Tensor of shape (num_tokens,) with L2 distance per token.
    """
    return torch.norm(a - b, p=2, dim=-1)


def mean_pool(acts: torch.Tensor) -> torch.Tensor:
    """Mean-pool activations over the token dimension.

    Args:
        acts: Tensor of shape (num_tokens, hidden_size).

    Returns:
        Tensor of shape (hidden_size,).
    """
    return acts.mean(dim=0)


def cosine_similarity_segment(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between mean-pooled activation vectors.

    Args:
        a, b: Tensors of shape (num_tokens_a, hidden_size) and (num_tokens_b, hidden_size).

    Returns:
        Scalar cosine similarity.
    """
    a_pooled = mean_pool(a)
    b_pooled = mean_pool(b)
    return F.cosine_similarity(a_pooled.unsqueeze(0), b_pooled.unsqueeze(0)).item()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Linear CKA (Centered Kernel Alignment).

    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    X and Y are centered activation matrices. Handles different token counts natively.

    Args:
        x: Tensor of shape (n, hidden_size) - n tokens from prompt A.
        y: Tensor of shape (m, hidden_size) - m tokens from prompt B.

    Returns:
        CKA similarity value in [0, 1].
    """
    # Center the representations
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Compute cross-covariance and self-covariances
    xtx = x.T @ x  # (hidden, hidden)
    yty = y.T @ y  # (hidden, hidden)
    xty = x.T @ y  # (hidden, hidden)

    # Frobenius norms
    numerator = torch.norm(xty, p="fro") ** 2
    denominator = torch.norm(xtx, p="fro") * torch.norm(yty, p="fro")

    if denominator < 1e-10:
        return 0.0

    return (numerator / denominator).item()


def compute_layer_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    layer_index: int,
    compute_cosine: bool = True,
    compute_l2: bool = True,
    compute_cka: bool = True,
    token_level: bool = True,
    segment_level: bool = True,
) -> LayerMetrics:
    """Compute all requested metrics for a single layer.

    Args:
        a, b: Activation tensors. Same shape (num_tokens, hidden) for token-level,
               possibly different first dim for segment-level only.
        layer_index: Which layer these activations come from.
    """
    metrics = LayerMetrics(layer_index=layer_index)

    if token_level and a.shape[0] == b.shape[0] and a.shape[0] > 0:
        if compute_cosine:
            cos = cosine_similarity_paired(a, b)
            metrics.cosine_per_token = cos
            metrics.cosine_mean = cos.mean().item()
            metrics.cosine_std = cos.std().item() if cos.numel() > 1 else 0.0
        if compute_l2:
            l2 = l2_distance_paired(a, b)
            metrics.l2_per_token = l2
            metrics.l2_mean = l2.mean().item()
            metrics.l2_std = l2.std().item() if l2.numel() > 1 else 0.0

    if segment_level and a.shape[0] > 0 and b.shape[0] > 0:
        if compute_cosine:
            metrics.cosine_segment = cosine_similarity_segment(a, b)
        if compute_cka:
            metrics.cka = linear_cka(a, b)

    return metrics


def compute_all_metrics(
    original_activations: dict[int, torch.Tensor],
    perturbed_activations: dict[int, torch.Tensor],
    compute_cosine: bool = True,
    compute_l2: bool = True,
    compute_cka: bool = True,
    token_level: bool = True,
    segment_level: bool = True,
) -> list[LayerMetrics]:
    """Compute metrics across all extracted layers.

    Args:
        original_activations: dict[layer_index -> (num_tokens, hidden_size)]
        perturbed_activations: dict[layer_index -> (num_tokens, hidden_size)]

    Returns:
        List of LayerMetrics sorted by layer index.
    """
    results = []
    common_layers = sorted(set(original_activations.keys()) & set(perturbed_activations.keys()))

    for layer_idx in common_layers:
        a = original_activations[layer_idx]
        b = perturbed_activations[layer_idx]
        metrics = compute_layer_metrics(
            a, b, layer_idx,
            compute_cosine=compute_cosine,
            compute_l2=compute_l2,
            compute_cka=compute_cka,
            token_level=token_level,
            segment_level=segment_level,
        )
        results.append(metrics)

    return results
