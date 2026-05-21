# Agent Prompt: Build the Unified CMG Evaluation Pipeline

## Context & Goal

You are building a single **master evaluation script** (`pipeline/run_pipeline.py`) for a Commit Message Generation (CMG) research experiment. The experiment compares four pipeline configurations across a shared task set, producing a unified results JSON that can feed directly into quality analysis.

The codebase already contains working implementations that **you must reuse**. Do NOT reinvent core logic — import it, call it, or adapt the patterns directly from the reference files listed below.

### Reference files (read ALL of these before writing any code)

| File | What to reuse |
|------|--------------|
| `lib/generation_functions.py` | `Fast_dLLM_QwenForCausalLM.batch_sample` engine, `FAST_DLLM_MASK_ID`, `FAST_DLLM_STOP_TOKEN`. **Import directly.** |
| `lib/diff_utils.py` | `get_per_file_diffs()`, `build_summary_messages()`, `build_cmg_messages()`. **Import directly.** |
| `inference/25_eval_dllm_summary.py` | **Primary reference** for the `dllm_llm` mode. Copy its: dLLM model loading (bound `mdm_sample` method via `types.MethodType`), `run_dllm_batch()` function, sort-by-token-length batching, 2-GPU detection (dLLM on cuda:0, LLM on cuda:1), `_MODEL_LOAD_LOCK` pattern, and `low_cpu_mem_usage=False` for dLLM. |
| `inference/26_eval_llm_summary.py` | **Primary reference** for the `llm_llm` mode. Copy its: sequential per-file LLM summary loop, single-GPU model reuse (`cmg_model = summary_model`), 2-GPU model separation. |
| `inference/10_new_eval.py` | **Primary reference** for the `dllm_only` mode. Copy its: `load_model()` + `gpu_worker()` pattern, `detect_devices()`, `ThreadPoolExecutor` multi-GPU dispatch, OOM-fallback-to-batch-1, per-task resume via output file check. |
| `inference/20_new_eval_llm.py` | **Primary reference** for the `llm_only` mode. Copy its: LLM `load_model()` pattern (`torch_dtype="auto"`, `device_map={"": device}`), `do_sample=False` greedy decoding, detailed timing stats (`prefill_seconds`, `decode_seconds`, `time_to_first_token_ms`). |
| `profiling/bench_dllm_batch.py` | **Copy verbatim**: `resolve_diff_cap()` function (handles `"none"`, `"600"`, `"tok:80"`, `"adaptive"`, `"adaptive:75"`), batch padding with `MASK_ID`, sort-by-length logic. |

**Do NOT rewrite `lib/` files.** Import from them.  
**Do NOT rewrite inference logic from scratch** — adapt the existing, tested implementations.

---

## Step 1: Design the Experiment (do this before writing any code)

Before writing any code, document your design decisions as comments at the top of `run_pipeline.py`. Specifically answer:

1. **What are the four pipeline modes?**
   - `dllm_only` — dLLM generates the commit message directly from the full diff (no summarisation step). Reference: `inference/10_new_eval.py`.
   - `llm_only` — LLM generates the commit message directly from the full diff. Reference: `inference/20_new_eval_llm.py`.
   - `dllm_llm` — dLLM summarises each file's diff in batches → LLM generates the commit message. Reference: `inference/25_eval_dllm_summary.py`.
   - `llm_llm` — LLM summarises each file's diff sequentially → LLM generates the commit message. Reference: `inference/26_eval_llm_summary.py`.

2. **What models are supported?**
   - dLLM: `Efficient-Large-Model/Fast_dLLM_v2_1.5B` (default), `Efficient-Large-Model/Fast_dLLM_v2_7B`
   - LLM generator: `Qwen/Qwen2.5-1.5B-Instruct` (default), `Qwen/Qwen2.5-7B-Instruct`
   - LLM summariser (for `llm_llm`): independently configurable, defaults to same as generator
   - All model names are configurable strings — not hardcoded enums

3. **What are the GPU scenarios?**

   **1 GPU** (`"devices": ["cuda:0"]`):
   - All models loaded onto the same device.
   - For two-step modes (`dllm_llm`, `llm_llm`): if summariser and generator are the same model, share one model instance (same pattern as `26_eval_llm_summary.py` single-GPU path: `cmg_model = summary_model`).
   - Tasks processed sequentially.

   **2 GPUs** (`"devices": ["cuda:0", "cuda:1"]`):
   - **Model-parallel, not data-parallel.** This follows the existing pattern in `25_eval_dllm_summary.py` and `26_eval_llm_summary.py`.
   - Summariser models (dLLM or LLM) load on `cuda:0` (the "summary device").
   - Generator models (LLM or dLLM) load on `cuda:1` (the "generation device").
   - For `*_only` modes, only one GPU is used (the one hosting the relevant model).
   - For two-step modes, both GPUs are active sequentially per task: Step 1 runs on the summary device, then Step 2 on the generation device. No pipeline parallelism — this ensures clean, uncontaminated timing measurements per step, which is critical for reliable experiment results.
   - Use `threading.Lock` around `from_pretrained` calls (same `_MODEL_LOAD_LOCK` pattern as `25_eval_dllm_summary.py`).

   **Why model-parallel instead of data-parallel for 2-GPU?**
   - Clean timing: summary time measured on GPU:0 with no resource contention, generation time measured on GPU:1 with no resource contention. No shared memory bandwidth, no cache eviction between models.
   - Memory headroom: each T4 (16GB) gets a single model — easily fits 1.5B in bf16 (~3GB) or 7B in int4 (~5GB) with ample KV-cache room.
   - Reproducibility: same model always runs on same hardware → consistent benchmarks.
   - Data-parallel (splitting tasks across GPUs) requires loading both summariser AND generator on each GPU, which would (a) halve available KV-cache memory, (b) introduce timing noise from memory contention, and (c) fail entirely for 7B models on 16GB GPUs.

4. **What is the output schema per task result?**
   Each result entry must contain:
   - `task_id`, `pipeline_mode`, `label` (ground truth)
   - `generated` (final commit message text)
   - `file_summaries` (list of `{filename, summary}` — empty list for `*_only` modes)
   - `timing`: `{summary_wall_s, cmg_wall_s, total_wall_s}`
   - `token_counts`: `{summary_tokens_per_file: list[int], cmg_tokens: int}`
   - `dllm_stats` (if applicable): `{total_steps, tokens_per_step, batch_sizes_used: list[int]}`
   - `llm_stats` (if applicable): `{prompt_tokens, generated_tokens, tokens_per_second}`
   - `config_snapshot`: copy of the resolved config dict used for this task (for reproducibility)

---

## Step 2: Build `pipeline/run_pipeline.py`

### Configuration

All parameters are loaded from a JSON config file (default: `pipeline/config.json`) and can be selectively overridden via CLI flags. The config file takes precedence over hardcoded defaults; CLI flags take precedence over config file.

**Every hyperparameter must be configurable** — no magic numbers buried in code. Defaults are the "sweet spots" from the existing implementations.

**Config schema** (all fields optional, defaults shown):

```json
{
  "input": "build_tasks/tasks_tags.jsonl",
  "output_file": "pipeline/results.json",
  "resume": true,

  "modes": ["dllm_only", "llm_only", "dllm_llm", "llm_llm"],

  "devices": ["cuda:0"],

  "dllm": {
    "model": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    "quantization": null,
    "block_size": 32,
    "small_block_size": 8,
    "threshold": 0.8,
    "use_block_cache": true,
    "temperature": 0.0,
    "top_p": 0.95,
    "compile": false,
    "low_cpu_mem_usage": false
  },

  "llm_generator": {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "quantization": null,
    "torch_dtype": "auto",
    "do_sample": false,
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.0
  },

  "llm_summariser": {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "quantization": null,
    "torch_dtype": "auto",
    "do_sample": false,
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.0
  },

  "summary": {
    "max_new_tokens": 128,
    "batch_size": 4,
    "diff_cap": "600",
    "sort_by_length": true
  },

  "cmg": {
    "max_new_tokens": 128
  },

  "sample": null,
  "task_ids": null,

  "no_summaries_in_output": false
}
```

**Default rationale** (document these in the README):
- `dllm.threshold: 0.8` — best quality/speed tradeoff from ablation (avoid 1.0 which degrades to autoregressive)
- `dllm.block_size: 32`, `dllm.small_block_size: 8` — optimal pair from ablation results
- `dllm.temperature: 0.0` — deterministic generation for reproducible experiments
- `dllm.use_block_cache: true` — KV-cache reuse for speed
- `dllm.low_cpu_mem_usage: false` — required so `lm_head.weight` is materialized (not meta tensor); same as `25_eval_dllm_summary.py`
- `dllm.compile: false` — `torch.compile` optional; adds startup latency but speeds up large runs
- `llm_*.do_sample: false` — greedy decoding for deterministic, reproducible results
- `llm_*.temperature: 1.0`, `llm_*.top_p: 0.95` — only used when `do_sample: true`; defaults from HF
- `summary.batch_size: 4` — sweet spot for dLLM batching on T4 (matches `25_eval_dllm_summary.py`)
- `summary.diff_cap: "600"` — 600-char cap; good balance between context and padding efficiency
- `summary.sort_by_length: true` — sort files by tokenized prompt length before batching to minimize padding
- `summary.max_new_tokens: 128`, `cmg.max_new_tokens: 128` — sufficient for summaries and commit messages

**`diff_cap` spec** — copy the `resolve_diff_cap()` function from `profiling/bench_dllm_batch.py` verbatim:
- `"none"` — no cap (full diff used)
- `"600"` — fixed 600-char cap
- `"tok:80"` — fixed 80 diff-token cap
- `"adaptive"` — p50 adaptive (median-based)
- `"adaptive:75"` — adaptive at 75th percentile

**`quantization`** field (applies to both dLLM and LLM models):
- `null` — no quantization (bfloat16 for dLLM, auto-detected dtype for LLM)
- `"int8"` — `load_in_8bit=True` via bitsandbytes
- `"int4"` — `load_in_4bit=True` via bitsandbytes (primarily for 7B on a single 16GB GPU)

### CLI flags

```
python pipeline/run_pipeline.py [--config pipeline/config.json]
  [--input PATH]           override config.input
  [--output-file PATH]     override config.output_file
  [--modes dllm_only,...]  comma-separated, overrides config.modes
  [--devices cuda:0,...]   comma-separated, overrides config.devices
  [--sample N]             run on first N tasks only
  [--task-ids id1,id2,...] run on specific task IDs only
  [--no-resume]            ignore existing output_file and start fresh
  [--no-summaries]         suppress per-file summaries from stdout (still saved to file)
```

### Model loading

**Reuse the loading patterns from the reference files:**

- **dLLM loading** — follow `inference/25_eval_dllm_summary.py` exactly:
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
  ).to(device)
  model.mdm_sample = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)
  if compile: model = torch.compile(model, mode="reduce-overhead")
  ```

- **LLM loading** — follow `inference/20_new_eval_llm.py` exactly:
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      model_name, torch_dtype="auto", device_map={"": device}
  )
  ```

- **Quantized loading** (for both dLLM and LLM):
  ```python
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(load_in_8bit=True)   # or load_in_4bit=True
  model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, device_map={"": device})
  ```

Load each distinct model exactly once per device. Determine which models are needed from the resolved mode list:

- `dllm_only` or `dllm_llm` → need dLLM
- `llm_only` or `dllm_llm` → need LLM generator
- `llm_llm` → need both LLM summariser and LLM generator
  - If same model name on same device → share one instance (same as `26_eval_llm_summary.py` single-GPU path)
  - If different devices (2-GPU) → load separate instances

Do NOT use `device_map="auto"` — always pin to explicit device (`device_map={"": device}`).

### Task processing

**Reuse the per-mode processing logic from the reference implementations:**

- **`dllm_only`**: adapt `inference/10_new_eval.py`'s `run_batch()` → process each task's full diff through dLLM `batch_sample()`. Include the OOM-fallback-to-batch-1 pattern.
- **`llm_only`**: adapt `inference/20_new_eval_llm.py`'s per-task flow → `model.generate()` with greedy decoding. Capture `prefill_seconds`, `decode_seconds`, `tokens_per_second`.
- **`dllm_llm`**: adapt `inference/25_eval_dllm_summary.py`'s two-step flow:
  1. `get_per_file_diffs()` → `build_summary_messages()` per file → sort by token length → batch with `MASK_ID` padding → `model.mdm_sample()` → collect summaries
  2. `build_cmg_messages(file_summaries)` → LLM `model.generate()` → final commit message
- **`llm_llm`**: adapt `inference/26_eval_llm_summary.py`'s two-step flow:
  1. Sequential per-file: `build_summary_messages()` → LLM `model.generate()` → collect summaries
  2. `build_cmg_messages(file_summaries)` → LLM `model.generate()` → final commit message

For each `(task, mode)` combination:

1. Check resume: if `task_id + "_" + mode` already exists in loaded results, skip.
2. Wrap execution in `try/except Exception` — on failure, write an error record:
   `{"task_id": ..., "pipeline_mode": ..., "error": str(e), "traceback": ...}` and continue.
3. On each successful result, immediately **append to a checkpoint file** (`output_file + ".ckpt.jsonl"`) so progress is never lost.

### Progress saving

- On startup: load existing `output_file` and `.ckpt.jsonl` to build the resume set.
- On each result: append to `.ckpt.jsonl` (one JSON line).
- On completion (or SIGINT): merge all results, write final `output_file` as a JSON array, delete `.ckpt.jsonl`.
- Install a `signal.signal(signal.SIGINT, ...)` handler to trigger the final save on Ctrl+C.

### Logging

Use Python `logging` with `%(asctime)s [%(levelname)s]` format. Log:
- Model load events with device and model name
- Each task start/finish with wall time
- Batch sizes used in dLLM summarisation
- Any truncation applied by diff cap
- Errors with full traceback at WARNING level

### Sort-by-length for dLLM batches

When `summary.sort_by_length` is true (default), sort files within each task by post-truncation prompt token length ascending before batching — this minimises padding waste. Use the actual tokenized length after `build_summary_messages()` + `apply_chat_template()`, not raw diff char length. This is the same approach used in `inference/25_eval_dllm_summary.py`.

---

## Step 3: Provide config files

### `pipeline/config.json` — default config (1.5B, 1 GPU, all 4 modes)

Provide the default config exactly matching the schema above.

### `pipeline/config_7b.json` — partial override: all 7B models

Partial override for running **all models** at 7B scale with int4 quantization on a single 16GB GPU:

```json
{
  "dllm": {
    "model": "Efficient-Large-Model/Fast_dLLM_v2_7B",
    "quantization": "int4"
  },
  "llm_generator": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "quantization": "int4"
  },
  "llm_summariser": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "quantization": "int4"
  }
}
```

### `pipeline/config_2gpu.json` — partial override: 2 GPUs (model-parallel)

```json
{
  "devices": ["cuda:0", "cuda:1"]
}
```

Note: all config override files are **partial** — `run_pipeline.py` must deep-merge them onto the defaults from `config.json`, so only the keys present in the override file are changed.

---

## Step 4: Output file schema

The final `results.json` is a JSON array. Each element:

```json
{
  "task_id": "apache_spark_a3feffd",
  "pipeline_mode": "dllm_llm",
  "label": "ground truth commit message",
  "generated": "model-generated commit message",
  "file_summaries": [
    {"filename": "path/to/File.scala", "summary": "..."}
  ],
  "timing": {
    "summary_wall_s": 12.4,
    "cmg_wall_s": 3.1,
    "total_wall_s": 15.5
  },
  "token_counts": {
    "summary_tokens_per_file": [112, 98, 107],
    "cmg_tokens": 87
  },
  "dllm_stats": {
    "total_steps": 142,
    "tokens_per_step": 6.74,
    "batch_sizes_used": [4, 4, 3]
  },
  "llm_stats": {
    "prompt_tokens": 412,
    "generated_tokens": 87,
    "tokens_per_second": 28.4
  },
  "config_snapshot": { "...resolved config dict..." },
  "error": null
}
```

For `*_only` modes, `file_summaries` is `[]`.  
For modes without a dLLM step, `dllm_stats` is `null`.  
For modes without an LLM step, `llm_stats` is `null`.

---

## Step 5: Quality analysis hook

At the end of `run_pipeline.py`, after saving, print a quick summary table:

```
Pipeline results summary
────────────────────────────────────────────────────────
Mode         Tasks   Errors   Avg total_wall_s   Avg cmg_tokens
dllm_only      9911       0             10.6s              112
llm_only       9911       0              8.3s               98
dllm_llm       9911       0             14.1s              103
llm_llm        9911       0             19.4s              105
────────────────────────────────────────────────────────
Results saved → pipeline/results.json
```

---

## Constraints & Requirements

- **Reuse existing code**: The four pipeline mode implementations must be adapted from their reference files (`10_new_eval.py`, `20_new_eval_llm.py`, `25_eval_dllm_summary.py`, `26_eval_llm_summary.py`). Import `lib/` modules. Copy `resolve_diff_cap()` from `profiling/bench_dllm_batch.py`. Do NOT rewrite inference logic from scratch.
- **Everything configurable**: Every hyperparameter (model names, block sizes, thresholds, temperatures, batch sizes, token limits, diff caps, quantization, compilation, etc.) must be in the config JSON and overridable. No magic numbers in code.
- **No global model state**: do not store models as module-level globals. Pass them as arguments.
- **Thread safety**: `_MODEL_LOAD_LOCK` around `from_pretrained` calls (same pattern as `25_eval_dllm_summary.py`). The `.ckpt.jsonl` file is written with a lock.
- **No `device_map="auto"`**: always pin to explicit device (`device_map={"": device}`).
- **No silent truncation**: when diff cap truncates a file's diff, log it at DEBUG level with the original and truncated lengths.
- **Fail loudly on bad config**: validate all config keys on startup; raise `ValueError` with a clear message for unknown keys or incompatible combinations (e.g., `dllm_only` mode requested but no CUDA device available).
- **No hardcoded paths**: all paths come from config or CLI.
- **Python 3.10+**: use `match/case` where it improves readability; use `X | Y` union types.
- **Dependency on `lib/`**: use `sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))` — same pattern as all existing inference scripts.

---

## Files to create

```
pipeline/
    run_pipeline.py        ← master script
    config.json            ← default config (1.5B, 1 GPU, all 4 modes)
    config_7b.json         ← partial override: all 7B int4
    config_2gpu.json       ← partial override: 2 GPUs (model-parallel)
    README.md              ← one-page usage guide with example commands
```

Do not modify any existing files outside `pipeline/`.

---

## Example invocations to include in the README

```bash
# Run all 4 modes on full dataset with defaults (1.5B, 1 GPU)
python pipeline/run_pipeline.py

# Quick 50-task smoke test
python pipeline/run_pipeline.py --sample 50

# dLLM+LLM only, save to custom file
python pipeline/run_pipeline.py --modes dllm_llm --output-file pipeline/dllm_llm_results.json

# All 7B models in int4 (override config)
python pipeline/run_pipeline.py --config pipeline/config_7b.json

# 2-GPU model-parallel run
python pipeline/run_pipeline.py --config pipeline/config_2gpu.json

# Combine overrides: 7B models on 2 GPUs
python pipeline/run_pipeline.py --config pipeline/config_7b.json --devices cuda:0,cuda:1

# Run on specific tasks (for debugging / spot-check)
python pipeline/run_pipeline.py --task-ids apache_spark_a3feffd,apache_doris_c3253b4,apache_echarts_444bc08

# Resume an interrupted run (default behavior — no flag needed)
python pipeline/run_pipeline.py

# Fresh run, ignoring existing checkpoint
python pipeline/run_pipeline.py --no-resume
```
