# Agent Prompt: Build the Unified CMG Evaluation Pipeline

## Context & Goal

You are building a single **master evaluation script** (`pipeline/run_pipeline.py`) for a Commit Message Generation (CMG) research experiment. The experiment compares four pipeline configurations across a shared task set, producing a unified results JSON that can feed directly into quality analysis.

The codebase already contains:
- `lib/generation_functions.py` — Fast-dLLM v2 `batch_sample` engine (NVIDIA, Apache 2.0)
- `lib/diff_utils.py` — `get_per_file_diffs()`, `build_summary_messages()`, `build_cmg_messages()`
- `inference/25_eval_dllm_summary.py` — working dLLM→LLM two-step pipeline (reference implementation)
- `inference/26_eval_llm_summary.py` — working LLM→LLM two-step pipeline (reference implementation)
- `inference/10_new_eval.py` — dLLM direct pipeline with multi-GPU ThreadPoolExecutor pattern
- `inference/20_new_eval_llm.py` — LLM direct pipeline with multi-GPU ThreadPoolExecutor pattern
- `profiling/bench_dllm_batch.py` — diff-cap resolution logic (`resolve_diff_cap`), batch padding, sort-by-length — **copy this logic verbatim**
- `build_tasks/build_tasks.py` — produces `tasks_tags.jsonl` and `labels.jsonl`

**Do NOT rewrite `lib/` files.** Import from them.

---

## Step 1: Design the Experiment (do this before writing any code)

Before writing any code, document your design decisions as comments at the top of `run_pipeline.py`. Specifically answer:

1. **What are the four pipeline modes?**
   - `dllm_only` — dLLM generates the commit message directly from the full diff (no summarisation step)
   - `llm_only` — LLM generates the commit message directly from the full diff
   - `dllm_llm` — dLLM summarises each file's diff in batches → LLM generates the commit message
   - `llm_llm` — LLM summarises each file's diff sequentially → LLM generates the commit message

2. **What models are supported?**
   - dLLM: `Efficient-Large-Model/Fast_dLLM_v2_1.5B` (fixed — only 1.5B exists)
   - LLM generator: `Qwen/Qwen2.5-1.5B-Instruct` (default), `Qwen/Qwen2.5-7B-Instruct` (large)
   - LLM summariser (for `llm_llm`): same model as generator, or independently configurable

3. **What are the GPU scenarios?**
   - 1 GPU: all models share `cuda:0`, sequential task processing
   - 2 GPUs: tasks are split into two halves; each half runs on a worker thread with its own model instance loaded on `cuda:0` and `cuda:1` respectively. Use `threading.Lock` around `from_pretrained` calls (see `10_new_eval.py`). Both workers write to a shared thread-safe result list protected by a `threading.Lock`.

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
    "block_size": 32,
    "small_block_size": 8,
    "threshold": 0.8,
    "use_block_cache": true,
    "temperature": 0.0,
    "top_p": 0.95,
    "compile": false
  },

  "llm_generator": {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "quantization": null,
    "do_sample": false,
    "temperature": 1.0,
    "top_p": 0.95
  },

  "llm_summariser": {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "quantization": null,
    "do_sample": false,
    "temperature": 1.0,
    "top_p": 0.95
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

**`diff_cap` spec** — copy the `resolve_diff_cap()` logic from `profiling/bench_dllm_batch.py` exactly:
- `"none"` — no cap
- `"600"` — fixed 600-char cap
- `"tok:80"` — fixed 80 diff-token cap
- `"adaptive"` — p50 adaptive
- `"adaptive:75"` — adaptive at 75th percentile

**`quantization`** field for LLM models:
- `null` — no quantization (bfloat16)
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

Load each distinct model exactly once per device. Determine which models are needed from the resolved mode list:

- `dllm_only` or `dllm_llm` → need dLLM
- `llm_only` or `llm_llm` or `dllm_llm` → need LLM generator
- `llm_llm` → need LLM summariser (may be the same object as LLM generator if same model+device)

For **quantization**, use:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_8bit=True)   # or load_in_4bit=True
model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
```
Do NOT use `device_map="auto"` for quantized models on single-GPU; use `device_map={"": device}`.

For 2-GPU scenario: load separate model instances per thread (same pattern as `10_new_eval.py`). If the same model is needed as both summariser and generator in `llm_llm` mode, load two independent instances (one per GPU).

### Task processing

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

## Step 3: Provide `pipeline/config.json`

Provide the default config as `pipeline/config.json` exactly matching the schema above, with sensible defaults. Add a second file `pipeline/config_7b.json` preconfigured for running the LLM with the 7B model in int4 quantization on 1 GPU:

```json
{
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

Note: `config_7b.json` is a **partial override** — `run_pipeline.py` should deep-merge it onto `config.json` defaults, so only the keys present in `config_7b.json` are overridden.

Also provide `pipeline/config_2gpu.json` for the 2-GPU scenario:
```json
{
  "devices": ["cuda:0", "cuda:1"]
}
```

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

- **No global model state**: do not store models as module-level globals. Pass them as arguments.
- **Thread safety**: the 2-GPU worker threads each have their own model objects. The shared result list is protected by `threading.Lock`. The `.ckpt.jsonl` file is written with a lock.
- **No `device_map="auto"`**: always pin to explicit device (`device_map={"": device}`), except for quantized 7B where bitsandbytes may require `device_map="auto"` — in that case document it explicitly.
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
    config_7b.json         ← partial override: 7B int4
    config_2gpu.json       ← partial override: 2 GPUs
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

# dLLM+LLM only, custom diff cap, save to custom file
python pipeline/run_pipeline.py --modes dllm_llm --output-file pipeline/dllm_llm_results.json

# 7B model in int4 (override config)
python pipeline/run_pipeline.py --config pipeline/config_7b.json

# 2-GPU run
python pipeline/run_pipeline.py --config pipeline/config_2gpu.json

# Run on specific tasks (for debugging / spot-check)
python pipeline/run_pipeline.py --task-ids apache_spark_a3feffd,apache_doris_c3253b4,apache_echarts_444bc08

# Resume an interrupted run (default behavior — no flag needed)
python pipeline/run_pipeline.py

# Fresh run, ignoring existing checkpoint
python pipeline/run_pipeline.py --no-resume
```
