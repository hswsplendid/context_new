"""
Experiment sweep orchestration.

Generates the full Cartesian product of experimental conditions, creates
jobs, and dispatches them through the GPU scheduler.
"""

from __future__ import annotations

import hashlib
import logging
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from .experiment import run_prompt_diff_experiment, run_single_experiment, run_triplet_experiment
from .prompts import PromptGenerator
from .scheduler import ExperimentJob, ExperimentScheduler, GPUInfo, detect_gpus
from .storage import ResultStore

logger = logging.getLogger(__name__)


def _make_experiment_id(params: Dict[str, Any]) -> str:
    """Deterministic experiment id from the parameter dict."""
    key = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_sweep_jobs(
    config: Dict[str, Any],
    prompt_token_counts: Optional[List[int]] = None,
) -> List[ExperimentJob]:
    """Build the full list of experiment jobs from config.

    Parameters
    ----------
    prompt_token_counts:
        When using ``source='jsonfile'``, pass a list of actual token counts
        for each prompt.  The sweep will iterate over these real lengths
        instead of the ``context_lengths`` list in config.
    """
    sweep = config["sweep"]
    jobs: List[ExperimentJob] = []

    # Determine context lengths to sweep over.
    if prompt_token_counts is not None:
        # Each prompt is a separate "context length".
        context_lengths = prompt_token_counts
    else:
        context_lengths = sweep["context_lengths"]

    for ctx_len, pos_frac, span_len, pert_type in product(
        context_lengths,
        sweep["perturbation_positions"],
        sweep["perturbation_span_lengths"],
        sweep["perturbation_types"],
    ):
        # Skip impossible combinations:
        # span can't be larger than what remains after the perturbation position.
        max_possible_span = int(ctx_len * (1.0 - pos_frac))
        if span_len > max_possible_span:
            continue

        params = {
            "context_length": ctx_len,
            "perturbation_position_frac": pos_frac,
            "perturbation_span_length": span_len,
            "perturbation_type": pert_type,
        }
        job_id = _make_experiment_id(params)
        jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info("Built %d experiment jobs.", len(jobs))
    return jobs


def build_prompt_diff_jobs(
    prompt_token_counts: List[int],
) -> List[ExperimentJob]:
    """Build jobs for consecutive prompt-pair comparisons.

    For N prompts, produces N-1 jobs — one per consecutive pair
    ``(prompt[i], prompt[i+1])``.
    """
    jobs: List[ExperimentJob] = []
    for i in range(len(prompt_token_counts) - 1):
        params = {
            "perturbation_type": "prompt_diff",
            "prompt_pair_index": i,
            "prev_prompt_index": i,
            "curr_prompt_index": i + 1,
            "prev_token_count": prompt_token_counts[i],
            "curr_token_count": prompt_token_counts[i + 1],
        }
        job_id = _make_experiment_id(params)
        jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info("Built %d prompt-diff jobs.", len(jobs))
    return jobs


def build_triplet_jobs(
    chain_groups: Dict[str, List[int]],
) -> List[ExperimentJob]:
    """Build triplet comparison jobs from prefix-chain groups.

    Parameters
    ----------
    chain_groups:
        Mapping from ``chain_id`` to list of prompt indices (into the
        flat prompt list) that form a prefix chain, sorted by position.

    For a chain of length N, produces N-2 triplet jobs:
    ``(prompt[i], prompt[i+1], prompt[i+2])`` for i in 0..N-3.
    """
    jobs: List[ExperimentJob] = []
    for chain_id, indices in sorted(chain_groups.items()):
        if len(indices) < 3:
            continue
        for i in range(len(indices) - 2):
            params = {
                "perturbation_type": "prompt_diff_triplet",
                "chain_id": chain_id,
                "triplet_index": i,
                "p_k_index": indices[i],
                "p_k1_index": indices[i + 1],
                "p_k2_index": indices[i + 2],
            }
            job_id = _make_experiment_id(params)
            jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info("Built %d triplet jobs from %d chains.", len(jobs), len(chain_groups))
    return jobs


def run_sweep(config: Dict[str, Any]):
    """Entry point: configure resources, generate jobs, and run."""
    # --- GPU detection ------------------------------------------------
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    max_parallel = gpu_cfg.get("max_parallel_experiments", 4)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs meet the free-memory threshold. Will use CPU.")

    # --- Storage ------------------------------------------------------
    store_cfg = config.get("storage", {})
    store = ResultStore(
        output_dir=store_cfg.get("output_dir", "./results"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 10),
    )

    # --- Model loading (done once, shared across sequential jobs) ----
    from .model import load_model_and_tokenizer

    model_cfg = config["model"]

    # Build max_memory map from CUDA-visible devices (not pynvml indices).
    import torch
    max_mem_gb = model_cfg.get("max_memory_per_gpu_gb", 70)
    num_visible = torch.cuda.device_count()
    if num_visible > 0:
        max_memory = {i: f"{max_mem_gb}GiB" for i in range(num_visible)}
    else:
        max_memory = None

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_cfg["path"],
        dtype=model_cfg.get("dtype", "float16"),
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # --- Prompt generator ---------------------------------------------
    prompt_cfg = config.get("prompts", {})
    prompt_gen = PromptGenerator(
        tokenizer=tokenizer,
        source=prompt_cfg.get("source", "dataset"),
        dataset_name=prompt_cfg.get("dataset_name", "wikitext"),
        dataset_config=prompt_cfg.get("dataset_config", "wikitext-103-raw-v1"),
        dataset_split=prompt_cfg.get("dataset_split", "train"),
        num_candidates=prompt_cfg.get("num_prompt_candidates", 50),
        seed=prompt_cfg.get("seed", 42),
        jsonfile_path=prompt_cfg.get("jsonfile_path"),
    )

    # Pre-tokenize a replacement pool for semantic_change strategy.
    _replacement_pool = tokenizer.encode(
        prompt_gen.generate(target_length=2048, index=99),
        add_special_tokens=False,
    )

    # --- Build jobs ---------------------------------------------------
    is_jsonfile = prompt_cfg.get("source") == "jsonfile"

    # When using jsonfile source, each prompt has its own natural token count.
    # Build a mapping: token_count → prompt_index for lookup.
    prompt_token_counts = None
    _ctx_to_prompt_idx: Dict[int, int] = {}
    if is_jsonfile:
        prompt_token_counts = []
        for i, passage in enumerate(prompt_gen._passages):
            ntok = len(tokenizer.encode(passage, add_special_tokens=False))
            prompt_token_counts.append(ntok)
            _ctx_to_prompt_idx[ntok] = i
        logger.info(
            "jsonfile mode: %d prompts, token counts: %s",
            len(prompt_token_counts),
            prompt_token_counts,
        )

    # --- Detect prompt_diff mode --------------------------------------
    sweep = config["sweep"]
    is_prompt_diff = (
        sweep["perturbation_types"] == ["prompt_diff"] and is_jsonfile
    )

    # Check if the JSON records carry chain metadata (for triplet mode).
    _chain_groups: Dict[str, List[int]] = {}
    is_triplet = False
    if is_prompt_diff and prompt_gen._json_records:
        for i, rec in enumerate(prompt_gen._json_records):
            cid = rec.get("chain_id")
            if cid is not None:
                _chain_groups.setdefault(cid, []).append(i)
        if _chain_groups:
            is_triplet = True
            logger.info(
                "Triplet mode: %d chains, %d total prompts.",
                len(_chain_groups),
                sum(len(v) for v in _chain_groups.values()),
            )

    if is_triplet:
        all_jobs = build_triplet_jobs(_chain_groups)
    elif is_prompt_diff:
        all_jobs = build_prompt_diff_jobs(prompt_token_counts)
    else:
        all_jobs = build_sweep_jobs(config, prompt_token_counts=prompt_token_counts)

    measurement_cfg = config.get("measurement", {})
    model_id = model_cfg["path"]

    # --- Execute (sequential; the scheduler is available for parallel
    #     runs when the model fits on 1 GPU and multiple copies can
    #     coexist, but the default safe mode is sequential) ------------
    total = len(all_jobs)
    logger.info("Starting sweep: %d experiments.", total)
    t0 = time.time()

    for idx, job in enumerate(all_jobs):
        if store.is_completed(job.job_id):
            logger.info("[%d/%d] Skipping completed %s", idx + 1, total, job.job_id)
            continue

        params = job.params

        if params.get("perturbation_type") == "prompt_diff_triplet":
            # --- Triplet mode: 3 consecutive prefix-chain prompts ---
            triplet_idx = params["triplet_index"]
            pk_idx = params["p_k_index"]
            pk1_idx = params["p_k1_index"]
            pk2_idx = params["p_k2_index"]
            chain_id = params["chain_id"]
            logger.info(
                "[%d/%d] Running triplet %s — chain=%s triplet=%d "
                "(prompts %d→%d→%d)",
                idx + 1, total, job.job_id, chain_id,
                triplet_idx, pk_idx, pk1_idx, pk2_idx,
            )

            p_k_text = prompt_gen.generate(target_length=0, index=pk_idx)
            p_k1_text = prompt_gen.generate(target_length=0, index=pk1_idx)
            p_k2_text = prompt_gen.generate(target_length=0, index=pk2_idx)

            try:
                records = run_triplet_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    p_k_text=p_k_text,
                    p_k1_text=p_k1_text,
                    p_k2_text=p_k2_text,
                    measurement_cfg=measurement_cfg,
                    triplet_index=triplet_idx,
                )
            except Exception:
                logger.exception("Triplet experiment %s failed.", job.job_id)
                continue

            # Compute metadata: D1 is the perturbation.
            pk_ids = tokenizer.encode(p_k_text, add_special_tokens=False)
            pk1_ids = tokenizer.encode(p_k1_text, add_special_tokens=False)
            pk2_ids = tokenizer.encode(p_k2_text, add_special_tokens=False)
            prefix_len = len(pk_ids)
            d1_len = len(pk1_ids) - prefix_len
            d2_len = len(pk2_ids) - len(pk1_ids)

            metadata = {
                "model_id": model_id,
                "context_length": len(pk2_ids),
                "perturbation_type": "prompt_diff_triplet",
                "perturbation_position_frac": (
                    prefix_len / len(pk2_ids) if len(pk2_ids) > 0 else 0.0
                ),
                "perturbation_position_abs": prefix_len,
                "perturbation_span_length": d1_len,
                "prompt_pair_index": triplet_idx,
                "common_prefix_length": prefix_len,
            }
            store.add_records(job.job_id, records, metadata=metadata)

        elif params.get("perturbation_type") == "prompt_diff":
            # --- Prompt-diff mode: consecutive pair comparison ---
            pair_idx = params["prompt_pair_index"]
            prev_idx = params["prev_prompt_index"]
            curr_idx = params["curr_prompt_index"]
            logger.info(
                "[%d/%d] Running prompt_diff %s — pair=%d (prompt %d→%d)",
                idx + 1, total, job.job_id, pair_idx, prev_idx, curr_idx,
            )

            prev_prompt = prompt_gen.generate(target_length=0, index=prev_idx)
            curr_prompt = prompt_gen.generate(target_length=0, index=curr_idx)

            try:
                records = run_prompt_diff_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    prev_prompt=prev_prompt,
                    curr_prompt=curr_prompt,
                    measurement_cfg=measurement_cfg,
                    prompt_pair_index=pair_idx,
                )
            except Exception:
                logger.exception("Prompt-diff experiment %s failed.", job.job_id)
                continue

            # Compute actual diff position and length from token counts.
            prev_ids = tokenizer.encode(prev_prompt, add_special_tokens=False)
            curr_ids = tokenizer.encode(curr_prompt, add_special_tokens=False)
            lcp = 0
            for a, b in zip(prev_ids, curr_ids):
                if a != b:
                    break
                lcp += 1
            else:
                lcp = min(len(prev_ids), len(curr_ids))

            metadata = {
                "model_id": model_id,
                "context_length": len(curr_ids),
                "perturbation_type": "prompt_diff",
                "perturbation_position_frac": lcp / len(curr_ids) if len(curr_ids) > 0 else 0.0,
                "perturbation_position_abs": lcp,
                "perturbation_span_length": len(curr_ids) - lcp,
                "prompt_pair_index": pair_idx,
                "common_prefix_length": lcp,
            }
            store.add_records(job.job_id, records, metadata=metadata)
        else:
            # --- Standard perturbation mode ---
            logger.info(
                "[%d/%d] Running %s — ctx=%d pos=%.2f span=%d type=%s",
                idx + 1,
                total,
                job.job_id,
                params["context_length"],
                params["perturbation_position_frac"],
                params["perturbation_span_length"],
                params["perturbation_type"],
            )

            if is_jsonfile:
                # Use the specific prompt whose token count matches this job.
                pidx = _ctx_to_prompt_idx[params["context_length"]]
                prompt_text = prompt_gen.generate(target_length=0, index=pidx)
            else:
                prompt_text = prompt_gen.generate(
                    target_length=params["context_length"], index=idx
                )

            try:
                records = run_single_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    perturbation_type=params["perturbation_type"],
                    perturbation_position_frac=params["perturbation_position_frac"],
                    perturbation_span_length=params["perturbation_span_length"],
                    measurement_cfg=measurement_cfg,
                    seed=prompt_cfg.get("seed", 42),
                    replacement_pool=_replacement_pool,
                )
            except Exception:
                logger.exception("Experiment %s failed.", job.job_id)
                continue

            metadata = {
                "model_id": model_id,
                "context_length": params["context_length"],
                "perturbation_type": params["perturbation_type"],
                "perturbation_position_frac": params["perturbation_position_frac"],
                "perturbation_position_abs": int(
                    params["perturbation_position_frac"] * params["context_length"]
                ),
                "perturbation_span_length": params["perturbation_span_length"],
            }
            store.add_records(job.job_id, records, metadata=metadata)

    store.flush()
    elapsed = time.time() - t0
    logger.info("Sweep complete. %d experiments in %.1f s.", total, elapsed)


# -------------------------------------------------------------------
# Parallel variant (uses scheduler for multi-copy model serving)
# -------------------------------------------------------------------
def run_sweep_parallel(config: Dict[str, Any]):
    """Run the sweep using the GPU scheduler for parallelism.

    This variant loads a *separate model copy per GPU group* and runs
    experiments concurrently.  Suitable when the model fits on a single
    GPU with room to spare.
    """
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    max_parallel = gpu_cfg.get("max_parallel_experiments", 4)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs available; falling back to sequential sweep.")
        return run_sweep(config)

    scheduler = ExperimentScheduler(
        available_gpus=gpus,
        max_parallel=max_parallel,
        gpus_per_experiment=1,
    )

    store_cfg = config.get("storage", {})
    store = ResultStore(
        output_dir=store_cfg.get("output_dir", "./results"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 10),
    )

    all_jobs = build_sweep_jobs(config)
    # Filter already completed.
    pending_jobs = [j for j in all_jobs if not store.is_completed(j.job_id)]
    logger.info(
        "%d total jobs, %d already completed, %d pending.",
        len(all_jobs),
        len(all_jobs) - len(pending_jobs),
        len(pending_jobs),
    )

    def worker_fn(job: ExperimentJob, gpu_indices: List[int]):
        """Worker that loads a model copy and runs the experiment."""
        import os
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))

        from .model import load_model_and_tokenizer
        from .prompts import PromptGenerator

        model_cfg = config["model"]
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_cfg["path"],
            dtype=model_cfg.get("dtype", "float16"),
            device_map="auto",
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )

        prompt_cfg = config.get("prompts", {})
        prompt_gen = PromptGenerator(
            tokenizer=tokenizer,
            source=prompt_cfg.get("source", "dataset"),
            dataset_name=prompt_cfg.get("dataset_name", "wikitext"),
            dataset_config=prompt_cfg.get("dataset_config", "wikitext-103-raw-v1"),
            dataset_split=prompt_cfg.get("dataset_split", "train"),
            num_candidates=prompt_cfg.get("num_prompt_candidates", 50),
            seed=prompt_cfg.get("seed", 42),
            jsonfile_path=prompt_cfg.get("jsonfile_path"),
        )

        params = job.params
        prompt_text = prompt_gen.generate(target_length=params["context_length"])

        replacement_pool = tokenizer.encode(
            prompt_gen.generate(target_length=2048, index=99),
            add_special_tokens=False,
        )

        measurement_cfg = config.get("measurement", {})
        records = run_single_experiment(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            perturbation_type=params["perturbation_type"],
            perturbation_position_frac=params["perturbation_position_frac"],
            perturbation_span_length=params["perturbation_span_length"],
            measurement_cfg=measurement_cfg,
            seed=prompt_cfg.get("seed", 42),
            replacement_pool=replacement_pool,
        )

        # Clean up.
        del model
        torch.cuda.empty_cache()

        return {
            "records": records,
            "metadata": {
                "model_id": model_cfg["path"],
                "context_length": params["context_length"],
                "perturbation_type": params["perturbation_type"],
                "perturbation_position_frac": params["perturbation_position_frac"],
                "perturbation_position_abs": int(
                    params["perturbation_position_frac"] * params["context_length"]
                ),
                "perturbation_span_length": params["perturbation_span_length"],
            },
        }

    results = scheduler.run(pending_jobs, worker_fn)

    for job_id, result in results:
        if result and "records" in result:
            store.add_records(job_id, result["records"], metadata=result["metadata"])

    store.flush()
    logger.info("Parallel sweep complete. %d jobs processed.", len(results))
