# Dependencies for Kaggle

When copying this experiment to Kaggle, you need the following files/folders alongside `main_experiment/`.

## Required files

```
main_experiment/                  ← this folder (upload as-is)
    100_main_experiment.py
    config.json
    config_7b.json                (optional — only if running 7B models)
    config_2gpu.json              (optional — only if using 2 GPUs)

lib/                              ← must be one level up from main_experiment/
    generation_functions.py       — Fast-dLLM v2 batch_sample engine (NVIDIA)
    diff_utils.py                 — get_per_file_diffs(), build_summary_messages(), build_cmg_messages()

build_tasks/
    tasks_tags.jsonl              — evaluation task set (input data)
```

## Folder structure on Kaggle

The script expects `lib/` to be a sibling of `main_experiment/` (i.e. at the project root). It adds `lib/` to `sys.path` automatically via:

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "lib"))
```

If your Kaggle working directory is `/kaggle/working/`, the layout should be:

```
/kaggle/working/
    lib/
        generation_functions.py
        diff_utils.py
    main_experiment/
        100_main_experiment.py
        config.json
        ...
    build_tasks/
        tasks_tags.jsonl
```

## Python packages required

These should already be available on Kaggle GPU instances. If not, install them:

```bash
pip install torch transformers accelerate bitsandbytes
```

`bitsandbytes` is only needed if using `"quantization": "int4"` or `"int8"` in the config.

## How to run on Kaggle

```python
# In a notebook cell:
!cd /kaggle/working && python main_experiment/100_main_experiment.py --sample 50

# Full run, all 4 modes:
!cd /kaggle/working && python main_experiment/100_main_experiment.py

# Only two-step modes:
!cd /kaggle/working && python main_experiment/100_main_experiment.py --modes dllm_llm,llm_llm

# 7B models with int4 quantization:
!cd /kaggle/working && python main_experiment/100_main_experiment.py --config main_experiment/config_7b.json

# 2-GPU (if Kaggle instance has 2 GPUs):
!cd /kaggle/working && python main_experiment/100_main_experiment.py --devices cuda:0,cuda:1
```

## Config defaults rationale

| Parameter | Default | Why |
|---|---|---|
| `dllm.threshold` | 0.8 | Best quality/speed from ablation; 1.0 degrades to autoregressive |
| `dllm.block_size` | 32 | Optimal from ablation results |
| `dllm.small_block_size` | 8 | Optimal from ablation results |
| `dllm.temperature` | 0.0 | Deterministic generation for reproducibility |
| `dllm.use_block_cache` | true | KV-cache reuse for speed |
| `dllm.low_cpu_mem_usage` | false | Required: ensures lm_head.weight is materialized |
| `llm_*.do_sample` | false | Greedy decoding for reproducibility |
| `summary.batch_size` | 4 | Sweet spot for dLLM batching on T4 |
| `summary.diff_cap` | "600" | 600-char cap; balances context and padding efficiency |
| `summary.sort_by_length` | true | Minimizes padding waste in dLLM batches |
| `summary.max_new_tokens` | 128 | Sufficient for file-level summaries |
| `cmg.max_new_tokens` | 128 | Sufficient for commit messages |

## 2-GPU strategy

When using 2 GPUs, the script uses **model-parallel** (not data-parallel):
- `cuda:0` — summariser models (dLLM or LLM)
- `cuda:1` — generator models (LLM or dLLM)

This gives each GPU full memory for one model and produces clean, uncontaminated timing per step.
