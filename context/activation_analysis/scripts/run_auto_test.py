#!/usr/bin/env python3
"""
Auto-test script with multi-GPU optimization for activation analysis experiments.

Orchestrates the full experiment pipeline:
  Phase 1: Sanity checks (identity + minimal perturbation)
  Phase 2: Type 1 main experiments (content replacement)
  Phase 3: Type 2 main experiments (semantic rewriting)
  Phase 4: Aggregation, analysis, and visualization

Usage:
    python scripts/run_auto_test.py \
        --gpus 0,3,4,5 \
        --cards-per-model 2 \
        --model-path Qwen/Qwen3-30B-A3B \
        --output-dir results/full_experiment \
        --phases 1,2,3,4 \
        --seed 42

最终指令：
 python scripts/run_auto_test.py       --gpus 0,1,2,3       --cards-per-model 2       --model-path /root/share/models/Qwen3-30B-A3B       --phases 1,2,3,4       --seed 42
 gpu最好是2的倍数 每两个一对
"""

import argparse
import logging
import os
import sys
import time
import uuid
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig, ExtractionConfig, PerturbationConfig, MetricsConfig, ExperimentConfig
from src.model_loader import load_model_and_tokenizer
from src.prompt_builder import load_template, load_replacements
from src.paraphrase_generator import batch_generate_paraphrases
from src.experiment_runner import run_single_experiment
from src.storage import save_metrics, load_metrics
from src.visualization import generate_all_plots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-test script with multi-GPU optimization"
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="Comma-separated GPU IDs, e.g. '0,3,4,5'"
    )
    parser.add_argument(
        "--cards-per-model", type=int, default=2,
        help="Number of GPUs per model instance (default: 2)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override model path from config"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "auto_test.yaml"),
        help="Path to auto_test.yaml config"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/auto_test_<timestamp>)"
    )
    parser.add_argument(
        "--phases", type=str, default="1,2,3,4",
        help="Comma-separated phase numbers to run, e.g. '1,2,3,4'"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: Phase 1 + reduced Phase 2 grid"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print experiment grid and GPU allocation, then exit"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output-dir, skip completed experiments"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_auto_config(path: str) -> dict:
    """Load the auto_test.yaml configuration as raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Experiment grid construction
# ---------------------------------------------------------------------------

def build_experiment_grid(
    raw_config: dict,
    templates: list[dict],
    phase: int,
    quick: bool = False,
) -> list[dict]:
    """Build the list of experiment parameter dicts for a given phase.

    Each entry contains: template_name, template_path, replacements_path,
    pair_index, context_length, ratio, position, perturbation_type.
    """
    if quick:
        perturb = raw_config.get("quick_perturbation", {})
    else:
        perturb = raw_config.get("perturbation", {})

    context_lengths = perturb.get("context_lengths", [1024])
    ratios = perturb.get("ratios", [0.25])
    positions = perturb.get("positions", ["middle"])

    ptype = "type1" if phase == 2 else "type2"

    grid = []
    for tmpl in templates:
        replacements = load_replacements(str(PROJECT_ROOT / tmpl["replacements_path"]))
        num_pairs = len(replacements)

        for ctx_len, ratio, position, pair_idx in product(
            context_lengths, ratios, positions, range(num_pairs)
        ):
            grid.append({
                "template_name": tmpl["name"],
                "template_path": str(PROJECT_ROOT / tmpl["template_path"]),
                "replacements_path": str(PROJECT_ROOT / tmpl["replacements_path"]),
                "pair_index": pair_idx,
                "context_length": ctx_len,
                "ratio": ratio,
                "position": position,
                "perturbation_type": ptype,
            })

    return grid


# ---------------------------------------------------------------------------
# GPU allocation
# ---------------------------------------------------------------------------

def allocate_gpus(gpu_list: list[int], cards_per_model: int) -> list[list[int]]:
    """Split GPU list into worker groups.

    Returns list of GPU-id lists, one per worker.
    """
    num_workers = len(gpu_list) // cards_per_model
    if num_workers == 0:
        raise ValueError(
            f"Need at least {cards_per_model} GPUs but got {len(gpu_list)}"
        )
    groups = []
    for i in range(num_workers):
        start = i * cards_per_model
        groups.append(gpu_list[start : start + cards_per_model])
    leftover = len(gpu_list) - num_workers * cards_per_model
    if leftover > 0:
        logger.warning(
            f"{leftover} GPU(s) unused: {gpu_list[num_workers * cards_per_model:]}"
        )
    return groups


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def worker_fn(
    worker_id: int,
    gpu_ids: list[int],
    experiments: list[dict],
    raw_config: dict,
    output_dir: str,
    seed: int,
    completed_ids: set,
):
    """Worker process: loads model on assigned GPUs and runs experiment subset.

    Writes results to worker-specific CSV to avoid file-lock contention.
    """
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    torch.manual_seed(seed)

    worker_csv = Path(output_dir) / f"metrics_worker{worker_id}.csv"

    # Build model config
    model_cfg = ModelConfig(
        model_path=raw_config["model"]["model_path"],
        torch_dtype=raw_config["model"].get("torch_dtype", "bfloat16"),
        device_map=raw_config["model"].get("device_map", "auto"),
        max_gpu_count=None,  # We control via CUDA_VISIBLE_DEVICES
    )

    extraction_cfg = ExtractionConfig(
        layer_indices=raw_config["extraction"]["layer_indices"],
    )

    logger.info(
        f"[Worker {worker_id}] Loading model on GPUs {gpu_ids} "
        f"({len(experiments)} experiments)"
    )

    try:
        model, tokenizer = load_model_and_tokenizer(model_cfg)
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Failed to load model: {e}")
        return

    # Paraphrase cache for Type 2 (per-worker, but shared cache file)
    paraphrase_cache_dir = Path(output_dir) / "paraphrase_cache"
    paraphrase_cache_dir.mkdir(parents=True, exist_ok=True)

    # Template cache
    template_cache = {}

    succeeded = 0
    failed = 0

    pbar = tqdm(
        experiments,
        desc=f"Worker {worker_id} (GPUs {gpu_ids})",
        position=worker_id,
    )

    for exp in pbar:
        exp_id = _make_experiment_id(exp)

        if exp_id in completed_ids:
            pbar.set_postfix_str(f"skip {exp_id[:16]}")
            continue

        ptype = exp["perturbation_type"]

        # Determine metrics config based on perturbation type
        if ptype == "type1":
            metrics_section = raw_config.get("metrics_type1", {})
        else:
            metrics_section = raw_config.get("metrics_type2", {})

        config = ExperimentConfig(
            model=model_cfg,
            extraction=extraction_cfg,
            perturbation=PerturbationConfig(
                type=ptype,
                context_lengths=[exp["context_length"]],
                ratios=[exp["ratio"]],
                positions=[exp["position"]],
            ),
            metrics=MetricsConfig(
                cosine=metrics_section.get("cosine", True),
                cka=metrics_section.get("cka", True),
                l2=metrics_section.get("l2", ptype == "type1"),
                granularity=metrics_section.get(
                    "granularity", ["token", "segment"] if ptype == "type1" else ["segment"]
                ),
            ),
            output_dir=output_dir,
            seed=seed,
            template_path=exp["template_path"],
            replacements_path=exp["replacements_path"],
        )

        # Load template (cached)
        tpath = exp["template_path"]
        if tpath not in template_cache:
            template_cache[tpath] = load_template(tpath)
        template = template_cache[tpath]

        # Load replacement pair
        replacements = load_replacements(exp["replacements_path"])
        pair = replacements[exp["pair_index"]]
        orig_seg = pair["original"]
        repl_seg = pair["replacement"]

        # For Type 2, generate paraphrase as the replacement
        if ptype == "type2":
            cache_file = (
                paraphrase_cache_dir
                / f"{exp['template_name']}_pair{exp['pair_index']}.yaml"
            )
            paraphrases = batch_generate_paraphrases(
                model, tokenizer, [orig_seg], cache_path=cache_file
            )
            repl_seg = paraphrases[0]

        try:
            result = run_single_experiment(
                model=model,
                tokenizer=tokenizer,
                config=config,
                template=template,
                context_length=exp["context_length"],
                perturbation_ratio=exp["ratio"],
                position=exp["position"],
                original_segment=orig_seg,
                replacement_segment=repl_seg,
                experiment_id=exp_id,
            )

            if result.error:
                logger.warning(f"[Worker {worker_id}] {exp_id}: {result.error}")
                failed += 1
            else:
                save_metrics(result, worker_csv)
                succeeded += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(f"[Worker {worker_id}] OOM on {exp_id}, skipping")
            failed += 1
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error on {exp_id}: {e}")
            failed += 1

        pbar.set_postfix_str(f"ok={succeeded} err={failed}")

    logger.info(
        f"[Worker {worker_id}] Done: {succeeded} succeeded, {failed} failed"
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()


def _make_experiment_id(exp: dict) -> str:
    """Create a deterministic experiment ID from parameters."""
    return (
        f"{exp['template_name']}_{exp['perturbation_type']}_"
        f"L{exp['context_length']}_R{exp['ratio']}_"
        f"P{exp['position']}_pair{exp['pair_index']}"
    )


# ---------------------------------------------------------------------------
# Phase 1: Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    raw_config: dict,
    output_dir: Path,
    gpu_ids: list[int],
    seed: int,
) -> bool:
    """Phase 1: Identity and minimal perturbation sanity checks.

    Returns True if all checks pass, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Sanity Checks")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    torch.manual_seed(seed)

    model_cfg = ModelConfig(
        model_path=raw_config["model"]["model_path"],
        torch_dtype=raw_config["model"].get("torch_dtype", "bfloat16"),
        device_map=raw_config["model"].get("device_map", "auto"),
    )
    extraction_cfg = ExtractionConfig(
        layer_indices=raw_config["extraction"]["layer_indices"],
    )

    model, tokenizer = load_model_and_tokenizer(model_cfg)

    sanity_cfg = raw_config.get("sanity", {})
    identity_min = sanity_cfg.get("identity_cosine_min", 0.999)
    minimal_shallow_min = sanity_cfg.get("minimal_perturbation_shallow_cosine_min", 0.95)
    shallow_threshold = sanity_cfg.get("shallow_layer_threshold", 12)

    templates_cfg = raw_config.get("templates", [])
    if not templates_cfg:
        logger.error("No templates configured")
        return False

    # Use first template for sanity checks
    tmpl_cfg = templates_cfg[0]
    template = load_template(str(PROJECT_ROOT / tmpl_cfg["template_path"]))
    replacements = load_replacements(str(PROJECT_ROOT / tmpl_cfg["replacements_path"]))

    all_passed = True
    sanity_dir = output_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    # --- Identity test: same prompt vs same prompt ---
    logger.info("--- Identity Test ---")
    orig_seg = replacements[0]["original"]

    config = ExperimentConfig(
        model=model_cfg,
        extraction=extraction_cfg,
        perturbation=PerturbationConfig(type="type1"),
        metrics=MetricsConfig(cosine=True, l2=False, cka=False, granularity=["token", "segment"]),
        seed=seed,
        template_path=str(PROJECT_ROOT / tmpl_cfg["template_path"]),
        replacements_path=str(PROJECT_ROOT / tmpl_cfg["replacements_path"]),
    )

    result = run_single_experiment(
        model=model,
        tokenizer=tokenizer,
        config=config,
        template=template,
        context_length=1024,
        perturbation_ratio=0.25,
        position="middle",
        original_segment=orig_seg,
        replacement_segment=orig_seg,  # Same text = identity
        experiment_id="sanity_identity",
    )

    if result.error:
        logger.error(f"Identity test failed: {result.error}")
        all_passed = False
    else:
        identity_cosines = [
            lm.cosine_mean for lm in result.layer_metrics
            if lm.cosine_mean is not None
        ]
        min_cos = min(identity_cosines) if identity_cosines else 0.0
        logger.info(f"Identity test cosines: min={min_cos:.6f}")

        if all(c >= identity_min for c in identity_cosines):
            logger.info(f"  PASS: All cosines >= {identity_min}")
        else:
            logger.error(
                f"  FAIL: Some cosines < {identity_min}. "
                f"Values: {[f'{c:.6f}' for c in identity_cosines]}"
            )
            all_passed = False

        save_metrics(result, sanity_dir / "identity_metrics.csv")

    # --- Minimal perturbation test ---
    logger.info("--- Minimal Perturbation Test ---")
    repl_seg = replacements[0]["replacement"]
    # Use very low ratio to simulate minimal perturbation
    result = run_single_experiment(
        model=model,
        tokenizer=tokenizer,
        config=config,
        template=template,
        context_length=1024,
        perturbation_ratio=0.05,  # Minimal perturbation
        position="middle",
        original_segment=orig_seg,
        replacement_segment=repl_seg,
        experiment_id="sanity_minimal",
    )

    if result.error:
        logger.error(f"Minimal perturbation test failed: {result.error}")
        all_passed = False
    else:
        shallow_cosines = [
            lm.cosine_mean for lm in result.layer_metrics
            if lm.cosine_mean is not None and lm.layer_index <= shallow_threshold
        ]
        if shallow_cosines:
            min_shallow = min(shallow_cosines)
            logger.info(f"Minimal perturbation shallow cosines: min={min_shallow:.6f}")

            if min_shallow >= minimal_shallow_min:
                logger.info(f"  PASS: All shallow cosines >= {minimal_shallow_min}")
            else:
                logger.warning(
                    f"  WARN: Shallow cosine {min_shallow:.6f} < {minimal_shallow_min}. "
                    f"Values: {[f'{c:.6f}' for c in shallow_cosines]}"
                )
                # Warning only, not a hard failure
        else:
            logger.warning("  No shallow layer cosines computed")

        save_metrics(result, sanity_dir / "minimal_metrics.csv")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    if all_passed:
        logger.info("Phase 1: ALL SANITY CHECKS PASSED")
    else:
        logger.error("Phase 1: SOME SANITY CHECKS FAILED")

    return all_passed


# ---------------------------------------------------------------------------
# Phase 2 & 3: Main experiments (multi-worker)
# ---------------------------------------------------------------------------

def run_main_experiments(
    phase: int,
    raw_config: dict,
    output_dir: Path,
    gpu_groups: list[list[int]],
    seed: int,
    quick: bool = False,
    resume: bool = False,
):
    """Run Phase 2 (Type 1) or Phase 3 (Type 2) experiments with multi-GPU workers."""
    phase_name = "Type 1 (Content Replacement)" if phase == 2 else "Type 2 (Semantic Rewriting)"
    logger.info("=" * 60)
    logger.info(f"Phase {phase}: {phase_name}")
    logger.info("=" * 60)

    templates = raw_config.get("templates", [])
    grid = build_experiment_grid(raw_config, templates, phase=phase, quick=quick)

    if not grid:
        logger.warning(f"Phase {phase}: Empty experiment grid")
        return

    phase_dir = output_dir / f"phase{phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Check for completed experiments (resume support)
    completed_ids = set()
    if resume:
        completed_ids = _load_completed_ids(phase_dir)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} experiments already completed")

    # Split grid across workers
    num_workers = len(gpu_groups)
    chunks = _split_grid(grid, num_workers)

    logger.info(
        f"Grid: {len(grid)} experiments, {num_workers} workers, "
        f"~{len(grid) // max(num_workers, 1)} per worker"
    )

    if num_workers == 1:
        # Single worker: run in main process (simpler debugging)
        worker_fn(
            worker_id=0,
            gpu_ids=gpu_groups[0],
            experiments=chunks[0],
            raw_config=raw_config,
            output_dir=str(phase_dir),
            seed=seed,
            completed_ids=completed_ids,
        )
    else:
        # Multi-worker: use multiprocessing
        mp.set_start_method("spawn", force=True)
        processes = []
        for i, (gpus, chunk) in enumerate(zip(gpu_groups, chunks)):
            p = mp.Process(
                target=worker_fn,
                args=(i, gpus, chunk, raw_config, str(phase_dir), seed, completed_ids),
            )
            p.start()
            processes.append(p)

        # Wait for all workers
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker exited with code {p.exitcode}")

    # Merge worker CSVs
    merged = merge_results(phase_dir)
    if merged is not None:
        logger.info(f"Phase {phase} complete: {len(merged)} metric rows")
    else:
        logger.warning(f"Phase {phase}: No results produced")


def _split_grid(grid: list[dict], num_workers: int) -> list[list[dict]]:
    """Split experiment grid into roughly equal chunks for workers."""
    chunks = [[] for _ in range(num_workers)]
    for i, exp in enumerate(grid):
        chunks[i % num_workers].append(exp)
    return chunks


def _load_completed_ids(phase_dir: Path) -> set:
    """Load experiment IDs from existing worker CSVs for resume support."""
    completed = set()
    for csv_file in phase_dir.glob("metrics_worker*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if "experiment_id" in df.columns:
                completed.update(df["experiment_id"].unique())
        except Exception:
            pass
    return completed


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------

def merge_results(phase_dir: Path) -> pd.DataFrame | None:
    """Merge all worker CSV files into a single metrics.csv."""
    csv_files = sorted(phase_dir.glob("metrics_worker*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged.drop_duplicates(subset=["experiment_id", "layer_index"], inplace=True)
    merged.to_csv(phase_dir / "metrics.csv", index=False)
    logger.info(f"Merged {len(csv_files)} worker files → {phase_dir / 'metrics.csv'}")
    return merged


# ---------------------------------------------------------------------------
# Phase 4: Analysis and visualization
# ---------------------------------------------------------------------------

def generate_analysis_report(output_dir: Path, raw_config: dict):
    """Phase 4: Merge all results, generate plots, and produce summary statistics."""
    logger.info("=" * 60)
    logger.info("Phase 4: Analysis & Visualization")
    logger.info("=" * 60)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all phase metrics
    all_dfs = {}
    for phase_num in [2, 3]:
        metrics_file = output_dir / f"phase{phase_num}" / "metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            all_dfs[f"phase{phase_num}"] = df
            logger.info(f"Loaded phase {phase_num}: {len(df)} rows")

    if not all_dfs:
        logger.warning("No metrics found for analysis")
        return

    # Generate plots for each phase
    for phase_key, df in all_dfs.items():
        phase_plot_dir = analysis_dir / phase_key
        phase_plot_dir.mkdir(parents=True, exist_ok=True)

        # Determine which metrics to plot
        ptype = "type1" if "phase2" in phase_key else "type2"
        if ptype == "type1":
            for metric in ["cosine_mean", "cosine_segment", "l2_mean", "cka"]:
                if metric in df.columns and df[metric].notna().any():
                    generate_all_plots(df, str(phase_plot_dir), metric=metric)
        else:
            for metric in ["cosine_segment", "cka"]:
                if metric in df.columns and df[metric].notna().any():
                    generate_all_plots(df, str(phase_plot_dir), metric=metric)

    # Generate summary statistics
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("EXPERIMENT SUMMARY REPORT")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    for phase_key, df in all_dfs.items():
        summary_lines.append(f"--- {phase_key.upper()} ---")
        summary_lines.append(f"Total metric rows: {len(df)}")
        summary_lines.append(
            f"Unique experiments: {df['experiment_id'].nunique()}"
        )
        summary_lines.append("")

        # Per-metric summaries
        numeric_cols = ["cosine_mean", "cosine_std", "l2_mean", "l2_std", "cosine_segment", "cka"]
        for col in numeric_cols:
            if col in df.columns and df[col].notna().any():
                vals = df[col].dropna()
                summary_lines.append(
                    f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                    f"min={vals.min():.4f}, max={vals.max():.4f}"
                )
        summary_lines.append("")

        # Hypothesis testing: shallow vs deep layer divergence
        sanity_cfg = raw_config.get("sanity", {})
        shallow_threshold = sanity_cfg.get("shallow_layer_threshold", 12)

        shallow = df[df["layer_index"] <= shallow_threshold]
        deep = df[df["layer_index"] > shallow_threshold]

        if "cosine_mean" in df.columns:
            s_cos = shallow["cosine_mean"].dropna()
            d_cos = deep["cosine_mean"].dropna()
            if len(s_cos) > 0 and len(d_cos) > 0:
                summary_lines.append(
                    f"  Shallow layers (<=L{shallow_threshold}) cosine: "
                    f"mean={s_cos.mean():.4f}"
                )
                summary_lines.append(
                    f"  Deep layers (>L{shallow_threshold}) cosine:    "
                    f"mean={d_cos.mean():.4f}"
                )
                delta = s_cos.mean() - d_cos.mean()
                summary_lines.append(
                    f"  Delta (shallow - deep): {delta:.4f}"
                )
                summary_lines.append("")

        # Per-ratio breakdown
        if "perturbation_ratio" in df.columns and "cosine_mean" in df.columns:
            summary_lines.append("  Per-ratio cosine_mean:")
            for ratio, grp in df.groupby("perturbation_ratio"):
                vals = grp["cosine_mean"].dropna()
                if len(vals) > 0:
                    summary_lines.append(
                        f"    ratio={ratio}: mean={vals.mean():.4f}, std={vals.std():.4f}"
                    )
            summary_lines.append("")

        # Per-position breakdown
        if "position" in df.columns and "cosine_mean" in df.columns:
            summary_lines.append("  Per-position cosine_mean:")
            for pos, grp in df.groupby("position"):
                vals = grp["cosine_mean"].dropna()
                if len(vals) > 0:
                    summary_lines.append(
                        f"    {pos}: mean={vals.mean():.4f}, std={vals.std():.4f}"
                    )
            summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_path = analysis_dir / "summary_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(f"Summary report saved to {summary_path}")
    print("\n" + summary_text)


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------

def print_dry_run(args, raw_config, gpu_groups):
    """Print experiment grid and GPU allocation without executing."""
    templates = raw_config.get("templates", [])

    print("\n" + "=" * 60)
    print("DRY RUN — Experiment Grid & GPU Allocation")
    print("=" * 60)

    print(f"\nModel: {raw_config['model']['model_path']}")
    print(f"GPUs: {args.gpus}")
    print(f"Cards per model: {args.cards_per_model}")
    print(f"Workers: {len(gpu_groups)}")
    for i, group in enumerate(gpu_groups):
        print(f"  Worker {i}: GPUs {group}")

    phases = [int(p) for p in args.phases.split(",")]

    for phase in phases:
        if phase == 1:
            print(f"\nPhase 1 (Sanity): 2 experiments on worker 0")
            continue
        if phase == 4:
            print(f"\nPhase 4 (Analysis): aggregation + visualization")
            continue

        grid = build_experiment_grid(
            raw_config, templates, phase=phase, quick=args.quick
        )
        print(f"\nPhase {phase} ({'Type 1' if phase == 2 else 'Type 2'}):")
        print(f"  Total experiments: {len(grid)}")
        print(f"  Per worker: ~{len(grid) // max(len(gpu_groups), 1)}")

        # Show grid dimensions
        ctx_lens = sorted(set(e["context_length"] for e in grid))
        ratios = sorted(set(e["ratio"] for e in grid))
        positions = sorted(set(e["position"] for e in grid))
        tmpls = sorted(set(e["template_name"] for e in grid))
        pairs = sorted(set(e["pair_index"] for e in grid))

        print(f"  Templates: {tmpls}")
        print(f"  Context lengths: {ctx_lens}")
        print(f"  Ratios: {ratios}")
        print(f"  Positions: {positions}")
        print(f"  Pair indices: {list(pairs)}")
        print(f"  Grid = {len(tmpls)} × {len(pairs)} × {len(ctx_lens)} × {len(ratios)} × {len(positions)} = {len(grid)}")

    print()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    raw_config = load_auto_config(args.config)

    # Override model path if specified
    if args.model_path:
        raw_config["model"]["model_path"] = args.model_path

    # Parse GPU list
    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
    gpu_groups = allocate_gpus(gpu_list, args.cards_per_model)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"auto_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    run_info = {
        "gpus": args.gpus,
        "cards_per_model": args.cards_per_model,
        "num_workers": len(gpu_groups),
        "gpu_groups": gpu_groups,
        "phases": args.phases,
        "quick": args.quick,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "config_path": args.config,
    }
    with open(output_dir / "run_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(run_info, f, default_flow_style=False)

    # Parse phases
    phases = [int(p) for p in args.phases.split(",")]

    # Dry run
    if args.dry_run:
        print_dry_run(args, raw_config, gpu_groups)
        return

    # Quick mode adjustments
    if args.quick:
        logger.info("Quick mode: reduced experiment grid")
        phases = [p for p in phases if p in (1, 2)]  # Only Phase 1 + 2

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Phases to run: {phases}")
    logger.info(f"Workers: {len(gpu_groups)}, GPU groups: {gpu_groups}")

    # Execute phases
    seed = args.seed

    if 1 in phases:
        passed = run_sanity_checks(
            raw_config=raw_config,
            output_dir=output_dir,
            gpu_ids=gpu_groups[0],  # Sanity runs on first worker's GPUs
            seed=seed,
        )
        if not passed:
            logger.error("Sanity checks failed! Continuing anyway...")

    if 2 in phases:
        run_main_experiments(
            phase=2,
            raw_config=raw_config,
            output_dir=output_dir,
            gpu_groups=gpu_groups,
            seed=seed,
            quick=args.quick,
            resume=args.resume,
        )

    if 3 in phases:
        run_main_experiments(
            phase=3,
            raw_config=raw_config,
            output_dir=output_dir,
            gpu_groups=gpu_groups,
            seed=seed,
            quick=args.quick,
            resume=args.resume,
        )

    if 4 in phases:
        generate_analysis_report(output_dir, raw_config)

    logger.info("All phases complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
