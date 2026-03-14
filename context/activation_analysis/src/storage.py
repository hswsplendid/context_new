"""Persistence utilities for saving/loading metrics, activations, and configs."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml

from .config import ExperimentConfig
from .experiment_runner import ExperimentResult
from .metrics import LayerMetrics

logger = logging.getLogger(__name__)


def _metrics_to_row(
    metrics: LayerMetrics,
    experiment_id: str,
    context_length: int,
    perturbation_ratio: float,
    position: str,
    perturbation_type: str,
) -> dict:
    """Convert a LayerMetrics to a flat dict for CSV storage."""
    return {
        "experiment_id": experiment_id,
        "context_length": context_length,
        "perturbation_ratio": perturbation_ratio,
        "position": position,
        "perturbation_type": perturbation_type,
        "layer_index": metrics.layer_index,
        "cosine_mean": metrics.cosine_mean,
        "cosine_std": metrics.cosine_std,
        "l2_mean": metrics.l2_mean,
        "l2_std": metrics.l2_std,
        "cosine_segment": metrics.cosine_segment,
        "cka": metrics.cka,
    }


def save_metrics(result: ExperimentResult, path: str | Path) -> None:
    """Save experiment metrics to CSV (append mode).

    Creates the file with headers if it doesn't exist, otherwise appends.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for lm in result.layer_metrics:
        rows.append(
            _metrics_to_row(
                lm,
                result.experiment_id,
                result.context_length,
                result.perturbation_ratio,
                result.position,
                result.perturbation_type,
            )
        )

    if not rows:
        logger.warning(f"No metrics to save for experiment {result.experiment_id}")
        return

    df = pd.DataFrame(rows)

    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)

    logger.info(f"Saved {len(rows)} metric rows to {path}")


def save_activations(
    activations: dict[int, torch.Tensor],
    path: str | Path,
    prefix: str = "activations",
) -> None:
    """Save raw activation tensors as .pt files.

    Each layer saved separately: {path}/{prefix}_layer{i}.pt
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for layer_idx, tensor in activations.items():
        filepath = path / f"{prefix}_layer{layer_idx}.pt"
        torch.save(tensor, filepath)

    logger.info(f"Saved {len(activations)} activation tensors to {path}")


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save experiment config as YAML for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    import dataclasses
    config_dict = dataclasses.asdict(config)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Saved config to {path}")


def load_metrics(path: str | Path) -> pd.DataFrame:
    """Load metrics CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} metric rows from {path}")
    return df


def load_activations(
    path: str | Path,
    prefix: str = "activations",
    layer_indices: Optional[list[int]] = None,
) -> dict[int, torch.Tensor]:
    """Load saved activation tensors.

    Args:
        path: Directory containing .pt files.
        prefix: Filename prefix.
        layer_indices: Specific layers to load. None loads all.

    Returns:
        dict[layer_index -> tensor]
    """
    path = Path(path)
    activations = {}

    for pt_file in sorted(path.glob(f"{prefix}_layer*.pt")):
        # Extract layer index from filename
        layer_str = pt_file.stem.replace(f"{prefix}_layer", "")
        try:
            layer_idx = int(layer_str)
        except ValueError:
            continue

        if layer_indices is not None and layer_idx not in layer_indices:
            continue

        activations[layer_idx] = torch.load(pt_file, weights_only=True)

    logger.info(f"Loaded {len(activations)} activation tensors from {path}")
    return activations
