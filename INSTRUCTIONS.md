# Reproduction Instructions

This document explains how to reproduce every file that is excluded from the repository via `.gitignore`.

---

## 0. Prerequisites

- NVIDIA GPU with CUDA (scripts target `cuda:0` / `cuda:1`)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

---

## 1. Clone & Set Up the Environment

```bash
git clone <repo-url>
cd local-dllm-exploration

# Create and activate conda env
conda create -n dllm python=3.11 -y
conda activate dllm

# Install PyTorch with CUDA 12.4 first (must match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt

# NLTK data required by analysis scripts
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

---

## 2. Download Source Datasets

### ApacheCM (full corpus)

The full dataset is sourced from the [ApacheCM](https://github.com/ApacheCM/ApacheCM) project.
Download and place the file at `datasets/ApacheCM/full.jsonl`.

```bash
# Example using Hugging Face datasets (adjust to actual source):
# python datasets/ApacheCM/explore_dataset.py datasets/ApacheCM/full.jsonl datasets/ApacheCM/exploration.json
```

`datasets/ApacheCM/test.jsonl` (the evaluation split) **is** tracked in git — do not regenerate it, as it defines the fixed test set.

### MCMD+ datasets

Download the per-language JSONL files and place them in `datasets/MCMD+/`:

```
datasets/MCMD+/cpp.jsonl
datasets/MCMD+/cs.jsonl
datasets/MCMD+/go.jsonl
datasets/MCMD+/java.jsonl
datasets/MCMD+/js.jsonl
datasets/MCMD+/php.jsonl
datasets/MCMD+/py.jsonl
datasets/MCMD+/rust.jsonl
```

---

## 3. Build Evaluation Tasks

Generates `build_tasks/tasks_tags.jsonl`, `tasks_no_tags.jsonl`, `tasks_3to10files.jsonl`, `tasks_4to12files.jsonl`, `labels.jsonl`, and `task_stats.json` from the ApacheCM full corpus (excluding the test split).

```bash
python build_tasks/build_tasks.py \
    --input datasets/ApacheCM/full.jsonl \
    --test  datasets/ApacheCM/test.jsonl \
    --output build_tasks/
```

---

## 4. (Optional) Download / Convert the dLLM Model

The `Efficient-Large-Model/Fast_dLLM_v2_1.5B` weights are downloaded from Hugging Face automatically at inference time. To pre-download and patch them into a local directory (e.g. `gemma4/`):

```bash
python training/60_convert_qwen_to_fast_dllm.py \
    --source Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output gemma4/ \
    --bd-size 32
```

---

## 5. Run dLLM Inference (Ablation Grid Search)

Produces `outputs/fresh_ablation_results/` (and optionally `outputs/new_ablation_64_32/`).

```bash
# Standard ablation (matches the paper configs)
python inference/10_new_eval.py \
    -i build_tasks/tasks_tags.jsonl \
    -o outputs/fresh_ablation_results/ \
    --thresholds 0.2,0.4,0.6,0.8,1.0 \
    --block-sizes 16,32,64 \
    --small-block-sizes 4,8,16,32 \
    --batch-sizes 1,4

# 64/32 variant
python inference/10_new_eval.py \
    -i build_tasks/tasks_tags.jsonl \
    -o outputs/new_ablation_64_32/ \
    --block-sizes 64 --small-block-sizes 32 \
    --thresholds 0.2,0.4,0.6,0.8,1.0
```

---

## 6. Run LLM Baseline Inference

Produces `outputs/results_llm_baseline/`.

```bash
python inference/20_new_eval_llm.py \
    --input build_tasks/tasks_tags.jsonl \
    --output outputs/results_llm_baseline/
```

---

## 7. (Optional) Run Two-Step Pipeline Inference

Produces `outputs/results_two_step_dllm/` and `outputs/results_two_step_llm/`.

```bash
# dLLM summarise → LLM CMG
python inference/25_eval_dllm_summary.py \
    -i build_tasks/tasks_tags.jsonl \
    -o outputs/results_two_step_dllm/

# LLM summarise → LLM CMG
python inference/26_eval_llm_summary.py \
    -i build_tasks/tasks_tags.jsonl \
    -o outputs/results_two_step_llm/
```

---

## 8. Compute Quality Metrics

Produces `outputs/quality_metrics/`.

```bash
python analysis/40_quality_eval.py \
    -i outputs/fresh_ablation_results/ \
       outputs/new_ablation_64_32/ \
       outputs/results_llm_baseline/ \
    -t build_tasks/tasks_tags.jsonl \
    -o outputs/quality_metrics/
```

---

## 9. Run Ablation Analysis & Generate Plots

Produces `outputs/analysis_output/` (plots, CSV, Markdown report, `all_metrics.json`).

```bash
python analysis/50_analyze_ablation.py
```

---

## 10. (Optional) Run Two-Step Analysis

```bash
python analysis/51_analyze_two_step.py \
    --tasks build_tasks/tasks_tags.jsonl
```

---

## 11. Build the Interactive Results Viewer

Produces `outputs/analysis_output/results_viewer.html`.

```bash
python analysis/80_build_results_viewer.py
```

---

## 12. (Optional) Build Training Data & Fine-Tune

```bash
# Build training split (writes build_tasks/train_tasks.jsonl)
python training/65_build_training_data.py

# Fine-tune (single GPU)
python training/70_train_block_diffusion.py \
    --dataset build_tasks/train_tasks.jsonl

# Fine-tune (multi-GPU with DeepSpeed)
deepspeed training/70_train_block_diffusion.py \
    --deepspeed training/ds_config_zero2.json \
    --dataset build_tasks/train_tasks.jsonl
```

---

## 13. (Optional) VRAM Profiling

Produces `outputs/vram_profile.json`.

```bash
python profiling/30_vram_profile.py \
    -i build_tasks/tasks_tags.jsonl
```

---

## Summary of What Each Script Generates

| Script | Outputs |
|---|---|
| `build_tasks/build_tasks.py` | `build_tasks/tasks_*.jsonl`, `labels.jsonl`, `task_stats.json` |
| `training/60_convert_qwen_to_fast_dllm.py` | `gemma4/` (model weights) |
| `training/65_build_training_data.py` | `build_tasks/train_tasks.jsonl` |
| `inference/10_new_eval.py` | `outputs/fresh_ablation_results/`, `outputs/new_ablation_*/` |
| `inference/20_new_eval_llm.py` | `outputs/results_llm_baseline/` |
| `inference/25_eval_dllm_summary.py` | `outputs/results_two_step_dllm/` |
| `inference/26_eval_llm_summary.py` | `outputs/results_two_step_llm/` |
| `analysis/40_quality_eval.py` | `outputs/quality_metrics/` |
| `analysis/50_analyze_ablation.py` | `outputs/analysis_output/` (plots, CSV, report, JSON) |
| `analysis/51_analyze_two_step.py` | `outputs/analysis_output/` (two-step plots/report) |
| `analysis/80_build_results_viewer.py` | `outputs/analysis_output/results_viewer.html` |
| `profiling/30_vram_profile.py` | `outputs/vram_profile.json` |
| `datasets/ApacheCM/explore_dataset.py` | `datasets/ApacheCM/exploration.json` |
