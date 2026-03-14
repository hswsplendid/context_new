"""Model and tokenizer loading utilities with explicit GPU allocation."""

import logging
import subprocess

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def probe_gpus(
    candidate_gpus: list[int] | None = None,
    min_free_mb: int = 2048,
) -> list[int]:
    """Return candidate GPUs that have at least ``min_free_mb`` MB free.

    Uses ``torch.cuda.mem_get_info`` when available, falls back to
    ``nvidia-smi`` parsing otherwise.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — returning empty GPU list")
        return []

    num_devices = torch.cuda.device_count()
    if candidate_gpus is None:
        candidate_gpus = list(range(num_devices))

    available: list[int] = []

    for gpu_id in candidate_gpus:
        if gpu_id >= num_devices:
            logger.warning(
                f"GPU {gpu_id} not visible (only {num_devices} devices) — skipping"
            )
            continue

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
        except Exception:
            # Fallback: parse nvidia-smi
            free_mb, total_mb = _nvidia_smi_free_mb(gpu_id)

        if free_mb >= min_free_mb:
            logger.info(
                f"GPU {gpu_id}: {free_mb:.0f}/{total_mb:.0f} MB free — accepted"
            )
            available.append(gpu_id)
        else:
            logger.warning(
                f"GPU {gpu_id}: {free_mb:.0f}/{total_mb:.0f} MB free "
                f"(< {min_free_mb} MB) — rejected"
            )

    return available


def _nvidia_smi_free_mb(gpu_id: int) -> tuple[float, float]:
    """Parse nvidia-smi to get (free_mb, total_mb) for a single GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        free_str, total_str = out.strip().split(",")
        return float(free_str.strip()), float(total_str.strip())
    except Exception as exc:
        logger.warning(f"nvidia-smi fallback failed for GPU {gpu_id}: {exc}")
        return 0.0, 0.0


def build_device_map(gpu_ids: list[int], num_layers: int = 49) -> dict:
    """Build an explicit device map that pins model layers to physical GPUs.

    Distributes ``num_layers`` decoder layers evenly across ``gpu_ids``.
    ``model.embed_tokens`` goes on the first GPU; ``model.norm`` and
    ``lm_head`` go on the last GPU.
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list")

    device_map: dict[str, int] = {}
    first_gpu = gpu_ids[0]
    last_gpu = gpu_ids[-1]

    # Embedding on first GPU
    device_map["model.embed_tokens"] = first_gpu
    # Rotary embedding (if present) on first GPU
    device_map["model.rotary_emb"] = first_gpu

    # Distribute decoder layers
    n_gpus = len(gpu_ids)
    layers_per_gpu = num_layers // n_gpus
    remainder = num_layers % n_gpus

    layer_idx = 0
    for i, gpu in enumerate(gpu_ids):
        # First `remainder` GPUs get one extra layer
        count = layers_per_gpu + (1 if i < remainder else 0)
        for _ in range(count):
            device_map[f"model.layers.{layer_idx}"] = gpu
            layer_idx += 1

    # Final norm + lm_head on last GPU
    device_map["model.norm"] = last_gpu
    device_map["lm_head"] = last_gpu

    logger.info(f"Built device map across GPUs {gpu_ids} for {num_layers} layers")
    for gpu in gpu_ids:
        assigned = [k for k, v in device_map.items() if v == gpu]
        layer_keys = [k for k in assigned if k.startswith("model.layers.")]
        other_keys = [k for k in assigned if not k.startswith("model.layers.")]
        parts = []
        if layer_keys:
            indices = sorted(int(k.split(".")[-1]) for k in layer_keys)
            parts.append(f"layers {indices[0]}-{indices[-1]}")
        if other_keys:
            parts.append(", ".join(other_keys))
        logger.info(f"  GPU {gpu}: {' + '.join(parts)}")

    return device_map


def load_model_and_tokenizer(
    config: ModelConfig,
    gpu_ids: list[int] | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer based on config.

    If ``gpu_ids`` is provided, builds an explicit device map that pins
    model layers to the specified physical GPUs.  Otherwise falls back
    to ``config.device_map`` (default ``"auto"``).
    """
    dtype = DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

    # Determine effective gpu_ids: explicit arg > config field
    effective_gpu_ids = gpu_ids if gpu_ids is not None else config.gpu_ids

    # Build device map
    if effective_gpu_ids is not None and len(effective_gpu_ids) > 0:
        # Auto-detect layer count from the model's config.json
        model_config = AutoConfig.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        num_layers = getattr(model_config, "num_hidden_layers", config.num_layers)
        logger.info(f"Detected {num_layers} layers from model config")
        device_map = build_device_map(effective_gpu_ids, num_layers)
    else:
        device_map = config.device_map

    logger.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dm_desc = (
        f"explicit ({len(effective_gpu_ids)} GPUs)"
        if isinstance(device_map, dict)
        else device_map
    )
    logger.info(
        f"Loading model from {config.model_path} "
        f"(dtype={config.torch_dtype}, device_map={dm_desc})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params / 1e9:.1f}B parameters")

    return model, tokenizer
