"""Configuration schema and YAML loading for experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    model_path: str = "Qwen/Qwen3-30B-A3B"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_gpu_count: Optional[int] = None  # deprecated, kept for compat
    gpu_ids: Optional[list[int]] = None
    min_free_gpu_mb: int = 2048
    num_layers: int = 49


@dataclass
class ExtractionConfig:
    layer_indices: list[int] = field(
        default_factory=lambda: list(range(0, 49, 4))  # [0,4,8,...,48], 13 layers
    )
    save_full_hidden: bool = False
    use_hooks: bool = True


@dataclass
class PerturbationConfig:
    type: str = "type1"  # "type1" or "type2"
    context_lengths: list[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    ratios: list[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50]
    )
    positions: list[str] = field(
        default_factory=lambda: ["beginning", "middle", "end"]
    )


@dataclass
class MetricsConfig:
    cosine: bool = True
    cka: bool = True
    l2: bool = True
    granularity: list[str] = field(
        default_factory=lambda: ["token", "segment"]
    )


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output_dir: str = "results"
    seed: int = 42
    template_path: str = "prompts/agent_templates/tool_use_agent.yaml"
    replacements_path: str = "prompts/perturbations/type1_replacements.yaml"


def _merge_dict_into_dataclass(dc_class, data: dict):
    """Recursively merge a dict into a dataclass, respecting nested dataclasses."""
    if data is None:
        return dc_class()
    field_types = {f.name: f.type for f in dc_class.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        # Check if the field type is itself a dataclass
        if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__"):
            kwargs[key] = _merge_dict_into_dataclass(ft, value)
        else:
            kwargs[key] = value
    return dc_class(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Missing fields use defaults from the dataclass definitions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _merge_dict_into_dataclass(ExperimentConfig, raw)

    # Validate
    if config.perturbation.type not in ("type1", "type2"):
        raise ValueError(
            f"perturbation.type must be 'type1' or 'type2', got '{config.perturbation.type}'"
        )
    for pos in config.perturbation.positions:
        if pos not in ("beginning", "middle", "end"):
            raise ValueError(f"Invalid position: '{pos}'")
    for ratio in config.perturbation.ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"Perturbation ratio must be in (0, 1), got {ratio}")

    return config
