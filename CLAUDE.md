# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

URECA research project: investigating whether post-training an instruct LLM on natural conversations (Reddit/Discord) improves its alignment with human humour preferences when used as a judge. Evaluation uses pairwise comparison tasks (which caption/joke is funnier?) on the New Yorker Caption Contest (primary) and Jester joke datasets.

The key insight: training data is deliberately out-of-domain from evaluation benchmarks to show improvements are not dataset-specific.

## Infrastructure

- **GPU machine**: 4090 at `192.168.0.85`, running Ollama (`http://192.168.0.85:11434`)
- Models are served via Ollama. Fine-tuned models should be exported to GGUF and loaded into Ollama for evaluation.
- Eval pipeline communicates with Ollama via `scripts/interact.py`

## Running Evaluations

```bash
cd scripts
python main.py
```

Configure what to run at the top of `main.py` (models, datasets, folds, limits). Results are saved to `results/newyorker/` and `results/jester/`, with per-run CSVs and a running `accuracy_summary.csv`.

## Architecture

**Two distinct phases — different tools for each:**

### Phase 1: Training (notebooks)
Jupyter notebooks in `training/` handle the exploratory work:
- Loading Reddit/Discord datasets from HuggingFace
- Formatting into SFT or DPO training format
- Running fine-tuning on the 4090 (via unsloth or HF Trainer)
- Exporting GGUF → loading into Ollama

### Phase 2: Evaluation (scripts)
`scripts/main.py` is the batch evaluation runner. Data flows:
1. `dataset_helpers.py` loads pairwise examples from CSV
2. `prompt_helpers.py` builds the judge prompt
3. `interact.py` sends to Ollama and returns `{content, thinking}`
4. Results parsed, saved via `result_io.py`

### Datasets
- `datasets/newyorker/ranking_fold{0-4}_{train,validation,test}.csv` — pre-split NYCC pairwise comparisons, each row has `joke_a_text`, `joke_b_text`, `expected` (A or B), `from_description` (pre-built prompt)
- `datasets/jester/cleaned_ranked_dataset{3,4}.csv` — Jester jokes ranked by avg rating; pairs are constructed at load time by offsetting indices
- Raw data in `datasets/newyorker/raw/` (HuggingFace arrow format)

To rebuild datasets: `scripts/prepare_datasets/prepare_newyorker_dataset.py` and `prepare_jester_dataset.py`

## Baseline Results (NYCC validation, "Reply with either <A> or <B>.")

All Qwen3 models cluster 54–61% across folds (random = 50%). Scaling barely helps — 32B is not clearly better than 8B. These are the numbers the fine-tuned model needs to beat (or compare against).

Jester accuracy is higher (61–73%) but noisier; use as secondary sanity check only.

## Key Config in main.py

- `MODELS` — Ollama model names (e.g. `"qwen3:14b"`)
- `DATASETS` — `"newyorker_default"` uses the pre-built `from_description` prompt; `"newyorker"` builds prompts from image descriptions
- `NYCC_SPLIT` / `NYCC_FOLDS` / `NYCC_LIMIT` — control what gets evaluated
- `THINK` — enables Ollama thinking mode
