# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework for analyzing how LLMs (Qwen3-30B-A3B) handle prompt perturbations by extracting and comparing intermediate activation patterns (hidden states) across transformer layers. Studies activation propagation in Agent scenarios (tool-calling, multi-turn conversations).

## Commands

```bash
# Install dependencies
pip install -r activation_analysis/requirements.txt

# Run a single experiment
python activation_analysis/scripts/run_experiment.py \
    --config activation_analysis/configs/default.yaml \
    --context-length 1024 --ratio 0.25 --position middle --pair-index 0

# Run full parameter sweep
python activation_analysis/scripts/run_sweep.py --config activation_analysis/configs/default.yaml

# Run sweep without plots
python activation_analysis/scripts/run_sweep.py --config activation_analysis/configs/default.yaml --no-plots

# Regenerate visualization from saved metrics
python activation_analysis/scripts/visualize_results.py results/sweep_xxxx/metrics.csv

# Run all tests (CPU-only, no GPU required)
pytest activation_analysis/tests/ -v

# Run a single test file
pytest activation_analysis/tests/test_metrics.py -v

# Run a specific test
pytest activation_analysis/tests/test_metrics.py::TestCosineSimilarityPaired::test_identical_vectors -v
```

## Architecture

### Pipeline Flow

`Load model → Build prompts (original + perturbed) → Extract activations via hooks → Align tokens → Compute metrics → Store CSV → Visualize`

### Core Modules (`activation_analysis/src/`)

- **config.py** — Dataclass-based config schema (ModelConfig, ExtractionConfig, PerturbationConfig, MetricsConfig) with YAML loading
- **model_loader.py** — Model/tokenizer loading with multi-GPU support (`device_map="auto"`, CUDA_VISIBLE_DEVICES)
- **prompt_builder.py** — Prompt template construction with segment boundary tracking. All prompts have three sections: `[Prefix] + [Segment (perturbed)] + [Subsequent]`
- **token_aligner.py** — Token alignment between original/perturbed prompts, handles BPE tokenization boundary issues
- **activation_extractor.py** — PyTorch hook-based selective hidden state extraction (memory-efficient vs `output_hidden_states`). Only extracts target tokens/layers.
- **metrics.py** — Similarity metrics: cosine similarity, L2 distance, CKA (Centered Kernel Alignment), mean pooling for segment comparison
- **paraphrase_generator.py** — LLM-based paraphrasing for Type 2 perturbations with YAML caching
- **experiment_runner.py** — Single experiment orchestration
- **batch_runner.py** — Parameter grid sweep across `context_lengths × ratios × positions × pairs`
- **storage.py** — CSV append-mode persistence for metrics
- **visualization.py** — Matplotlib (Agg backend) / seaborn plotting (heatmaps, line plots, bar charts)

### Experiment Types

- **Type 1 (Content Replacement)**: Replace specific text segments (e.g., tool descriptions) and measure downstream activation changes
- **Type 2 (Semantic Paraphrasing)**: Generate semantically equivalent rewrites and compare segment-level activations using mean pooling/CKA

### Configuration

YAML-based configs in `activation_analysis/configs/`. Four experiment variables: `context_lengths`, `ratios`, `positions`, `layer_indices`. Default extracts 13 layers (every 4th from 0-48).

### Output Structure

Results go to `results/sweep_{uuid}/` containing `config.yaml`, `metrics.csv`, and `plots/` directory with auto-generated visualizations.

### Metrics CSV Columns

`experiment_id, context_length, perturbation_ratio, position, perturbation_type, layer_index, cosine_mean, cosine_std, l2_mean, l2_std, cosine_segment, cka`

## Key Design Decisions

- **Hook-based extraction** over `output_hidden_states` for memory efficiency (~53MB for 13 layers × 1000 tokens vs full model states)
- **BPE boundary awareness** in token alignment — trims mismatched boundary tokens between original/perturbed sequences
- **Python 3.10+** with type hints throughout, dataclass configs with recursive YAML merging
- **All tests are CPU-only** using mock models/tokenizers — no GPU needed
