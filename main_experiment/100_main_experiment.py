#!/usr/bin/env python3
"""
Unified CMG Evaluation Pipeline — Master Experiment Script
============================================================

Compares four pipeline configurations for Commit Message Generation (CMG):

  dllm_only  — dLLM generates commit message directly from the full diff
  llm_only   — LLM generates commit message directly from the full diff
  dllm_llm   — dLLM summarises per-file diffs → LLM generates commit message
  llm_llm    — LLM summarises per-file diffs → LLM generates commit message

Design decisions:
  1. GPU strategy (2-GPU): model-parallel, NOT data-parallel.
     - Summariser on cuda:0, generator on cuda:1.
     - Clean timing: no memory contention between models.
     - Each T4 (16 GB) gets full memory for one model.
  2. All hyperparameters are configurable via JSON config + CLI overrides.
  3. Core inference logic is adapted from existing reference implementations:
     - dllm_only  ← inference/10_new_eval.py
     - llm_only   ← inference/20_new_eval_llm.py
     - dllm_llm   ← inference/25_eval_dllm_summary.py
     - llm_llm    ← inference/26_eval_llm_summary.py
  4. diff_cap resolution copied from profiling/bench_dllm_batch.py
  5. lib/diff_utils.py and lib/generation_functions.py imported directly.

Usage:
    python main_experiment/100_main_experiment.py
    python main_experiment/100_main_experiment.py --sample          # smoke-test: 1 task per mode
    python main_experiment/100_main_experiment.py --sample 50       # first 50 tasks per mode
    python main_experiment/100_main_experiment.py --config main_experiment/config_7b.json
    python main_experiment/100_main_experiment.py --modes dllm_llm,llm_llm
"""

import argparse
import copy
import itertools
import json
import logging
import signal
import sys
import time
import traceback
import threading
import types
from pathlib import Path

import torch

# ─── Path setup ───────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "lib"))

from generation_functions import (
    Fast_dLLM_QwenForCausalLM,
    FAST_DLLM_MASK_ID,
    FAST_DLLM_STOP_TOKEN,
)
from diff_utils import get_per_file_diffs, build_summary_messages, build_cmg_messages

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Thread safety ────────────────────────────────────────────────────────────

_MODEL_LOAD_LOCK = threading.Lock()
_CKPT_LOCK = threading.Lock()

# ─── Default config ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "input": "build_tasks/tasks_tags.jsonl",
    "output_file": "main_experiment/results.json",
    "resume": True,

    "modes": ["dllm_only", "llm_only", "dllm_llm", "llm_llm"],

    "devices": ["cuda:0"],

    "dllm": {
        "model": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        "quantization": None,
        "block_size": 32,
        "small_block_size": 8,
        "threshold": 0.8,
        "use_block_cache": True,
        "temperature": 0.0,
        "top_p": 0.95,
        "compile": False,
        "low_cpu_mem_usage": False,
    },

    "llm_generator": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization": None,
        "torch_dtype": "auto",
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    },

    "llm_summariser": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization": None,
        "torch_dtype": "auto",
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    },

    "summary": {
        "max_new_tokens": 256,
        "batch_size": 4,
        "diff_cap": "600",
        "sort_by_length": True,
    },

    "cmg": {
        "max_new_tokens": 128,
    },

    "sample": None,
    "task_ids": None,
    "no_summaries_in_output": False,
}

VALID_MODES = {"dllm_only", "llm_only", "dllm_llm", "llm_llm"}


# ─── Config utilities ────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | None) -> dict:
    """Load config from JSON, deep-merged onto defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(p, encoding="utf-8") as f:
            override = json.load(f)
        cfg = deep_merge(cfg, override)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI flags on top of the loaded config."""
    if args.input is not None:
        cfg["input"] = args.input
    if args.output_file is not None:
        cfg["output_file"] = args.output_file
    if args.modes is not None:
        cfg["modes"] = [m.strip() for m in args.modes.split(",")]
    if args.devices is not None:
        cfg["devices"] = [d.strip() for d in args.devices.split(",")]
    if args.sample is not None:
        cfg["sample"] = args.sample  # None → not set; 0/1/N from nargs='?'
    if args.task_ids is not None:
        cfg["task_ids"] = [t.strip() for t in args.task_ids.split(",")]
    if args.no_resume:
        cfg["resume"] = False
    if args.no_summaries:
        cfg["no_summaries_in_output"] = True
    return cfg


def validate_config(cfg: dict):
    """Validate config and raise ValueError on problems."""
    modes = cfg["modes"]
    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Unknown mode '{m}'. Valid: {VALID_MODES}")

    devices = cfg["devices"]
    if not devices:
        raise ValueError("At least one device must be specified.")

    needs_cuda = any(d.startswith("cuda") for d in devices)
    if needs_cuda and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA devices requested ({devices}) but CUDA is not available."
        )

    for d in devices:
        if d.startswith("cuda:"):
            idx = int(d.split(":")[1])
            if idx >= torch.cuda.device_count():
                raise ValueError(
                    f"Device '{d}' requested but only {torch.cuda.device_count()} GPU(s) available."
                )

    input_path = Path(cfg["input"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg['input']}")


# ─── Grid search utilities ────────────────────────────────────────────────────

# Keys that should NEVER be treated as grid-search axes even if they are lists
_GRID_SEARCH_SKIP_KEYS = frozenset({
    "modes", "devices", "task_ids",
    "batch_sizes_used",  # runtime field, not config
})


def _collect_grid_axes(cfg: dict, prefix: str = "") -> list[tuple[str, list]]:
    """
    Walk the config tree and find every key whose value is a list of scalars
    (i.e. a grid-search axis).  Returns [(dotted_key, [values]), ...].

    Lists of dicts or deeply nested lists are left alone.
    """
    axes = []
    for key, value in cfg.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if dotted in _GRID_SEARCH_SKIP_KEYS or key in _GRID_SEARCH_SKIP_KEYS:
            continue
        if isinstance(value, dict):
            axes.extend(_collect_grid_axes(value, dotted))
        elif isinstance(value, list) and value and all(
            isinstance(v, (int, float, str, bool, type(None))) for v in value
        ):
            axes.append((dotted, value))
    return axes


def _set_nested(cfg: dict, dotted_key: str, value):
    """Set a nested config value by dotted key (e.g. 'dllm.threshold')."""
    parts = dotted_key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def _get_nested(cfg: dict, dotted_key: str):
    """Get a nested config value by dotted key."""
    parts = dotted_key.split(".")
    d = cfg
    for p in parts:
        d = d[p]
    return d


def generate_grid_configs(cfg: dict) -> list[tuple[dict, str]]:
    """
    Given a config that may contain list-valued parameters (grid search axes),
    expand into all combinations.

    Returns [(cfg_copy, label_string), ...] where label_string encodes the
    parameter values for this combination (used in output file naming).
    """
    axes = _collect_grid_axes(cfg)
    if not axes:
        return [(cfg, "single")]

    keys = [k for k, _ in axes]
    value_lists = [v for _, v in axes]

    log.info(f"Grid search: {len(axes)} axis/axes, "
             f"{len(list(itertools.product(*value_lists)))} combinations")
    for k, v in axes:
        log.info(f"  {k}: {v}")

    configs = []
    for combo in itertools.product(*value_lists):
        c = copy.deepcopy(cfg)
        label_parts = []
        for k, v in zip(keys, combo):
            _set_nested(c, k, v)
            # Short label: last segment of dotted key + value
            short_key = k.split(".")[-1]
            label_parts.append(f"{short_key}={v}")
        label = "__".join(label_parts)
        configs.append((c, label))

    return configs


def grid_search_output_path(base_output: str, label: str) -> str:
    """Generate an output file path for a grid search combination."""
    p = Path(base_output)
    stem = p.stem
    # Remove .json extension if checkpoint-style name
    if stem.endswith(".json.ckpt"):
        stem = stem[:-10]
    elif stem.endswith(".json"):
        stem = stem[:-5]
    return str(p.parent / f"{stem}__{label}.json")


# ─── resolve_diff_cap (from profiling/bench_dllm_batch.py) ────────────────────

def resolve_diff_cap(spec, file_diffs, tokenizer):
    """
    Parse a diff-cap spec string and return (max_diff_chars, max_diff_tokens, label).

    Exactly one of max_diff_chars / max_diff_tokens is non-None (or both None).

    Spec syntax:
      "none"           → no cap (full diff)
      "<N>"            → fixed N-character cap (e.g. "600")
      "tok:<N>"        → fixed N diff-token cap (e.g. "tok:80")
      "adaptive"       → adaptive p50 (median prompt length target), char-based
      "adaptive:<pct>" → adaptive at the given percentile (0-100), char-based
    """
    spec = spec.strip().lower()
    if spec == "none":
        return None, None, "none (full diff used)"

    if spec.startswith("tok:"):
        n = int(spec[4:])
        return None, n, f"{n} diff tok [fixed token cap]"

    if spec.startswith("adaptive"):
        pct = 50.0
        if ":" in spec:
            pct = float(spec.split(":", 1)[1])
        pct = max(0.0, min(100.0, pct))

        raw_messages = [build_summary_messages(fn, fd) for fn, fd in file_diffs]
        raw_lens = []
        for msgs in raw_messages:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            raw_lens.append(
                len(tokenizer([text], return_tensors="pt")["input_ids"][0])
            )

        sorted_lens = sorted(raw_lens)
        n_raw = len(sorted_lens)
        idx_f = pct / 100.0 * (n_raw - 1)
        lo, hi = int(idx_f), min(int(idx_f) + 1, n_raw - 1)
        target_len = int(
            sorted_lens[lo] + (idx_f - lo) * (sorted_lens[hi] - sorted_lens[lo])
        )

        min_idx = raw_lens.index(min(raw_lens))
        diff_chars_min = len(file_diffs[min_idx][1])
        overhead_tokens = max(0, raw_lens[min_idx] - diff_chars_min)

        caps = []
        for raw_len, (fn, fd) in zip(raw_lens, file_diffs):
            diff_tok_count = max(1, raw_len - overhead_tokens)
            chars_per_tok = len(fd) / diff_tok_count
            diff_tok_budget = max(1, target_len - overhead_tokens)
            caps.append(int(diff_tok_budget * chars_per_tok))

        max_diff_chars = max(max(caps), 200)
        label = (
            f"{max_diff_chars} chars  "
            f"[adaptive p{pct:.0f} → target {target_len} tok | "
            f"raw prompt tok: min={min(raw_lens)}  max={max(raw_lens)}]"
        )
        return max_diff_chars, None, label

    try:
        n = int(spec)
        return n, None, f"{n} chars [fixed]"
    except ValueError:
        raise ValueError(
            f"Unknown diff_cap spec '{spec}'. "
            f"Valid: 'none', '<N>' (chars), 'tok:<N>' (tokens), "
            f"'adaptive', 'adaptive:<pct>'"
        )


# ─── Model loading ────────────────────────────────────────────────────────────

def _get_bnb_config(quantization: str | None):
    """Return a BitsAndBytesConfig or None."""
    if quantization is None:
        return None
    from transformers import BitsAndBytesConfig
    match quantization:
        case "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        case "int4":
            return BitsAndBytesConfig(load_in_4bit=True)
        case _:
            raise ValueError(f"Unknown quantization '{quantization}'. Use null, 'int8', or 'int4'.")


def load_dllm(cfg_dllm: dict, device: str):
    """
    Load a dLLM model and bind batch_sample.
    Adapted from inference/25_eval_dllm_summary.py load_dllm().
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg_dllm["model"]
    log.info(f"[dLLM/{device}] Loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    log.info(f"[dLLM/{device}] Loading model {model_name} ...")
    bnb = _get_bnb_config(cfg_dllm.get("quantization"))
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=cfg_dllm.get("low_cpu_mem_usage", False),
    )
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
        load_kwargs["device_map"] = {"": device}

    with _MODEL_LOAD_LOCK:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if bnb is None:
        model = model.to(device)
    model.eval()

    # Bind batch_sample method (same as 25_eval_dllm_summary.py)
    model.mdm_sample = types.MethodType(
        Fast_dLLM_QwenForCausalLM.batch_sample, model
    )

    if cfg_dllm.get("compile", False):
        log.info(f"[dLLM/{device}] Compiling with torch.compile (reduce-overhead) ...")
        model = torch.compile(model, mode="reduce-overhead")

    log.info(f"[dLLM/{device}] Ready.")
    return tokenizer, model


def load_llm(cfg_llm: dict, device: str):
    """
    Load an LLM model for generation or summarisation.
    Adapted from inference/20_new_eval_llm.py load_model().
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg_llm["model"]
    log.info(f"[LLM/{device}] Loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info(f"[LLM/{device}] Loading model {model_name} ...")
    bnb = _get_bnb_config(cfg_llm.get("quantization"))
    torch_dtype = cfg_llm.get("torch_dtype", "auto")
    if torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype == "float16":
        torch_dtype = torch.float16

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map={"": device},
    )
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb

    with _MODEL_LOAD_LOCK:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.eval()
    log.info(f"[LLM/{device}] Ready.")
    return tokenizer, model


# ─── Device assignment ────────────────────────────────────────────────────────

def resolve_devices(cfg: dict) -> dict:
    """
    Return a dict mapping role → device.

    Roles: 'dllm', 'llm_summariser', 'llm_generator'

    2-GPU model-parallel strategy:
      - cuda:0 = summary device (dLLM summaries or LLM summaries)
      - cuda:1 = generation device (LLM CMG or dLLM direct)
    1-GPU: everything on devices[0]
    """
    devices = cfg["devices"]
    if len(devices) >= 2:
        return {
            "dllm": devices[0],
            "llm_summariser": devices[0],
            "llm_generator": devices[1],
        }
    else:
        d = devices[0]
        return {"dllm": d, "llm_summariser": d, "llm_generator": d}


# ─── Model manager ───────────────────────────────────────────────────────────

class ModelManager:
    """
    Loads required models lazily, exactly once per (model_name, device) pair.
    On single-GPU, LLM summariser and generator share one instance if same model.
    """

    def __init__(self, cfg: dict, device_map: dict):
        self.cfg = cfg
        self.device_map = device_map
        self._cache: dict[str, tuple] = {}  # key → (tokenizer, model)

    def _cache_key(self, model_name: str, device: str) -> str:
        return f"{model_name}@{device}"

    def get_dllm(self) -> tuple:
        device = self.device_map["dllm"]
        key = self._cache_key(self.cfg["dllm"]["model"], device)
        if key not in self._cache:
            self._cache[key] = load_dllm(self.cfg["dllm"], device)
        return self._cache[key]

    def get_llm_generator(self) -> tuple:
        device = self.device_map["llm_generator"]
        key = self._cache_key(self.cfg["llm_generator"]["model"], device)
        if key not in self._cache:
            self._cache[key] = load_llm(self.cfg["llm_generator"], device)
        return self._cache[key]

    def get_llm_summariser(self) -> tuple:
        device = self.device_map["llm_summariser"]
        cfg_sum = self.cfg["llm_summariser"]
        key = self._cache_key(cfg_sum["model"], device)
        if key not in self._cache:
            # If same model+device as generator, reuse
            gen_device = self.device_map["llm_generator"]
            gen_key = self._cache_key(self.cfg["llm_generator"]["model"], gen_device)
            if key == gen_key and gen_key in self._cache:
                self._cache[key] = self._cache[gen_key]
            else:
                self._cache[key] = load_llm(cfg_sum, device)
        return self._cache[key]


# ─── dLLM batch inference (from 25_eval_dllm_summary.py) ─────────────────────

@torch.no_grad()
def run_dllm_batch(
    messages_list: list[list[dict]],
    dllm_model,
    dllm_tokenizer,
    device: str,
    gen_kwargs: dict,
    max_new_tokens: int,
) -> tuple[list[str], list[int], float, int, int]:
    """
    Run dLLM batch_sample for a list of prompt message sequences.
    Returns: (generated_texts, token_counts, wall_seconds, total_tokens, total_steps)
    """
    input_ids_list = []
    seq_lens = []
    for messages in messages_list:
        text = dllm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = dllm_tokenizer([text], return_tensors="pt")["input_ids"][0]
        input_ids_list.append(ids)
        seq_lens.append(len(ids))

    max_len = max(seq_lens)
    min_len = min(seq_lens)

    padded = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), FAST_DLLM_MASK_ID, dtype=torch.long)]
            )
        padded.append(ids.unsqueeze(0))
    batched_ids = torch.cat(padded, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, device=device)

    if device != "cpu":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    generated_ids, total_steps = dllm_model.mdm_sample(
        batched_ids,
        tokenizer=dllm_tokenizer,
        block_size=gen_kwargs["block_size"],
        small_block_size=gen_kwargs["small_block_size"],
        max_new_tokens=max_new_tokens,
        mask_id=FAST_DLLM_MASK_ID,
        min_len=min_len,
        seq_len=seq_len_tensor,
        use_block_cache=gen_kwargs.get("use_block_cache", True),
        threshold=gen_kwargs["threshold"],
        temperature=gen_kwargs.get("temperature", 0.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        stop_token=FAST_DLLM_STOP_TOKEN,
    )

    if device != "cpu":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t0

    texts = []
    counts = []
    total_tokens = 0
    for i, sl in enumerate(seq_lens):
        gen_part = generated_ids[i][sl:]
        n_tok = int((gen_part != FAST_DLLM_MASK_ID).sum().item())
        total_tokens += n_tok
        counts.append(n_tok)
        texts.append(dllm_tokenizer.decode(gen_part, skip_special_tokens=True))

    return texts, counts, wall_seconds, total_tokens, total_steps


# ─── LLM single inference (from 25/26_eval) ──────────────────────────────────

@torch.no_grad()
def run_llm_single(
    messages: list[dict],
    llm_model,
    llm_tokenizer,
    device: str,
    gen_kwargs: dict,
    max_new_tokens: int,
) -> tuple[str, int, float, int]:
    """
    Run a single LLM generate call.
    Returns: (generated_text, generated_token_count, wall_seconds, prompt_tokens)
    """
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if device != "cpu":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=gen_kwargs.get("do_sample", False),
        temperature=gen_kwargs.get("temperature", 1.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
        pad_token_id=llm_tokenizer.eos_token_id,
    )

    if device != "cpu":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t0

    gen_ids = outputs[0][prompt_len:]
    generated_text = llm_tokenizer.decode(gen_ids, skip_special_tokens=True)
    return generated_text, len(gen_ids), wall_seconds, prompt_len


# ─── dLLM direct inference (from 10_new_eval.py) ─────────────────────────────

@torch.no_grad()
def run_dllm_direct(
    task: dict,
    dllm_model,
    dllm_tokenizer,
    device: str,
    gen_kwargs: dict,
    max_new_tokens: int,
) -> tuple[str, int, float, int, int, int]:
    """
    Run dLLM direct inference for a single task (no summarisation).
    Returns: (generated_text, gen_tokens, wall_seconds, prompt_tokens, total_steps, total_tokens)
    """
    messages = task.get("messages", [])
    if not messages:
        raise ValueError(f"Task '{task.get('task_id', '?')}' has no 'messages' field.")

    text = dllm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = dllm_tokenizer([text], return_tensors="pt")["input_ids"][0]
    prompt_len = len(ids)

    # Pad to prompt_len + max_new_tokens with MASK_ID
    padded = torch.cat(
        [ids, torch.full((max_new_tokens,), FAST_DLLM_MASK_ID, dtype=torch.long)]
    ).unsqueeze(0).to(device)
    seq_len_tensor = torch.tensor([prompt_len], device=device)

    if device != "cpu":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    generated_ids, total_steps = dllm_model.mdm_sample(
        padded,
        tokenizer=dllm_tokenizer,
        block_size=gen_kwargs["block_size"],
        small_block_size=gen_kwargs["small_block_size"],
        max_new_tokens=max_new_tokens,
        mask_id=FAST_DLLM_MASK_ID,
        min_len=prompt_len,
        seq_len=seq_len_tensor,
        use_block_cache=gen_kwargs.get("use_block_cache", True),
        threshold=gen_kwargs["threshold"],
        temperature=gen_kwargs.get("temperature", 0.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        stop_token=FAST_DLLM_STOP_TOKEN,
    )

    if device != "cpu":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t0

    gen_part = generated_ids[0][prompt_len:]
    n_tok = int((gen_part != FAST_DLLM_MASK_ID).sum().item())
    generated_text = dllm_tokenizer.decode(gen_part, skip_special_tokens=True)

    return generated_text, n_tok, wall_seconds, prompt_len, total_steps, n_tok


# ─── Per-mode task processors ────────────────────────────────────────────────

def process_dllm_only(task: dict, models: ModelManager, cfg: dict) -> dict:
    """dLLM generates commit message directly from the full diff."""
    dllm_tokenizer, dllm_model = models.get_dllm()
    device = models.device_map["dllm"]
    dllm_cfg = cfg["dllm"]

    gen_kwargs = {
        "block_size": dllm_cfg["block_size"],
        "small_block_size": dllm_cfg["small_block_size"],
        "threshold": dllm_cfg["threshold"],
        "use_block_cache": dllm_cfg["use_block_cache"],
        "temperature": dllm_cfg["temperature"],
        "top_p": dllm_cfg["top_p"],
    }

    text, gen_tokens, wall_s, prompt_tokens, total_steps, total_tok = run_dllm_direct(
        task, dllm_model, dllm_tokenizer, device, gen_kwargs,
        cfg["cmg"]["max_new_tokens"],
    )

    eps = 1e-9
    tokens_per_step = total_tok / max(total_steps, 1)

    return {
        "task_id": task.get("task_id", "unknown"),
        "pipeline_mode": "dllm_only",
        "label": task.get("label", ""),
        "generated": text,
        "file_summaries": [],
        "timing": {
            "summary_wall_s": 0.0,
            "cmg_wall_s": round(wall_s, 4),
            "total_wall_s": round(wall_s, 4),
        },
        "token_counts": {
            "summary_tokens_per_file": [],
            "cmg_tokens": gen_tokens,
        },
        "dllm_stats": {
            "prompt_tokens": prompt_tokens,
            "total_steps": total_steps,
            "tokens_per_step": round(tokens_per_step, 3),
            "batch_sizes_used": [1],
        },
        "llm_stats": None,
        "config_snapshot": _make_config_snapshot(cfg, "dllm_only"),
        "error": None,
    }


def process_llm_only(task: dict, models: ModelManager, cfg: dict) -> dict:
    """LLM generates commit message directly from the full diff."""
    llm_tokenizer, llm_model = models.get_llm_generator()
    device = models.device_map["llm_generator"]
    llm_cfg = cfg["llm_generator"]

    gen_kwargs = {
        "do_sample": llm_cfg["do_sample"],
        "temperature": llm_cfg["temperature"],
        "top_p": llm_cfg["top_p"],
        "repetition_penalty": llm_cfg["repetition_penalty"],
    }

    messages = task.get("messages", [])
    if not messages:
        raise ValueError(f"Task '{task.get('task_id', '?')}' has no 'messages' field.")

    text, gen_tokens, wall_s, prompt_tokens = run_llm_single(
        messages, llm_model, llm_tokenizer, device, gen_kwargs,
        cfg["cmg"]["max_new_tokens"],
    )

    eps = 1e-9
    tps = gen_tokens / (wall_s + eps)

    return {
        "task_id": task.get("task_id", "unknown"),
        "pipeline_mode": "llm_only",
        "label": task.get("label", ""),
        "generated": text,
        "file_summaries": [],
        "timing": {
            "summary_wall_s": 0.0,
            "cmg_wall_s": round(wall_s, 4),
            "total_wall_s": round(wall_s, 4),
        },
        "token_counts": {
            "summary_tokens_per_file": [],
            "cmg_tokens": gen_tokens,
        },
        "dllm_stats": None,
        "llm_stats": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "tokens_per_second": round(tps, 2),
        },
        "config_snapshot": _make_config_snapshot(cfg, "llm_only"),
        "error": None,
    }


def process_dllm_llm(task: dict, models: ModelManager, cfg: dict) -> dict:
    """dLLM summarises per-file diffs, then LLM generates commit message."""
    dllm_tokenizer, dllm_model = models.get_dllm()
    llm_tokenizer, llm_model = models.get_llm_generator()
    dllm_device = models.device_map["dllm"]
    llm_device = models.device_map["llm_generator"]
    dllm_cfg = cfg["dllm"]
    summary_cfg = cfg["summary"]
    llm_gen_cfg = cfg["llm_generator"]

    dllm_gen_kwargs = {
        "block_size": dllm_cfg["block_size"],
        "small_block_size": dllm_cfg["small_block_size"],
        "threshold": dllm_cfg["threshold"],
        "use_block_cache": dllm_cfg["use_block_cache"],
        "temperature": dllm_cfg["temperature"],
        "top_p": dllm_cfg["top_p"],
    }
    llm_gen_kwargs = {
        "do_sample": llm_gen_cfg["do_sample"],
        "temperature": llm_gen_cfg["temperature"],
        "top_p": llm_gen_cfg["top_p"],
        "repetition_penalty": llm_gen_cfg["repetition_penalty"],
    }

    task_id = task.get("task_id", "unknown")
    label = task.get("label", "")

    # ── Step 0: per-file diffs ───────────────────────────────────────────
    file_diffs = get_per_file_diffs(task)

    # Resolve diff cap
    max_diff_chars, max_diff_tokens, cap_label = resolve_diff_cap(
        summary_cfg["diff_cap"], file_diffs, dllm_tokenizer,
    )
    log.debug(f"[{task_id}] diff_cap resolved: {cap_label}")

    # Apply token-level truncation if needed
    processed_diffs = []
    for fn, fd in file_diffs:
        if max_diff_tokens is not None:
            raw_ids = dllm_tokenizer(fd, return_tensors="pt")["input_ids"][0]
            if len(raw_ids) > max_diff_tokens:
                log.debug(
                    f"[{task_id}] Truncating {fn}: {len(raw_ids)} → {max_diff_tokens} tokens"
                )
                fd = dllm_tokenizer.decode(
                    raw_ids[:max_diff_tokens], skip_special_tokens=True
                )
        processed_diffs.append((fn, fd))

    # Build messages (char-cap applied inside build_summary_messages)
    char_cap = None if max_diff_tokens is not None else max_diff_chars
    all_summary_messages = [
        build_summary_messages(fn, fd, max_diff_chars=char_cap)
        for fn, fd in processed_diffs
    ]

    # ── Sort by prompt token length (from 25_eval_dllm_summary.py) ───────
    if summary_cfg["sort_by_length"]:
        prompt_tok_lens = [
            len(dllm_tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True
            ))
            for msgs in all_summary_messages
        ]
        sort_order = sorted(range(len(processed_diffs)), key=lambda i: prompt_tok_lens[i])
        processed_diffs = [processed_diffs[i] for i in sort_order]
        all_summary_messages = [all_summary_messages[i] for i in sort_order]

    # ── Step 1: dLLM file summarisation ──────────────────────────────────
    file_summaries = []
    summary_token_counts = []
    t_summary_total = 0.0
    summary_total_tokens = 0
    summary_total_steps = 0
    batch_sizes_used = []
    batch_size = summary_cfg["batch_size"]

    for batch_start in range(0, len(all_summary_messages), batch_size):
        batch_msgs = all_summary_messages[batch_start:batch_start + batch_size]
        batch_filenames = [
            fd[0] for fd in processed_diffs[batch_start:batch_start + batch_size]
        ]
        actual_bs = len(batch_msgs)
        batch_sizes_used.append(actual_bs)

        texts, counts, wall_s, total_tok, total_steps = run_dllm_batch(
            batch_msgs, dllm_model, dllm_tokenizer, dllm_device,
            dllm_gen_kwargs, summary_cfg["max_new_tokens"],
        )

        t_summary_total += wall_s
        summary_total_tokens += total_tok
        summary_total_steps += total_steps

        for filename, summary_text, n_tok in zip(batch_filenames, texts, counts):
            file_summaries.append((filename, summary_text))
            summary_token_counts.append(n_tok)

    log.debug(f"[{task_id}] dLLM batches: {batch_sizes_used}")

    # ── Step 2: LLM commit message generation ────────────────────────────
    cmg_messages = build_cmg_messages(file_summaries)
    cmg_text, cmg_tokens, t_cmg, cmg_prompt_tokens = run_llm_single(
        cmg_messages, llm_model, llm_tokenizer, llm_device,
        llm_gen_kwargs, cfg["cmg"]["max_new_tokens"],
    )

    t_total = t_summary_total + t_cmg
    eps = 1e-9
    tps_cmg = cmg_tokens / (t_cmg + eps)
    tokens_per_step = summary_total_tokens / max(summary_total_steps, 1)

    eps = 1e-9
    summary_tps = summary_total_tokens / (t_summary_total + eps)

    return {
        "task_id": task_id,
        "pipeline_mode": "dllm_llm",
        "label": label,
        "generated": cmg_text,
        "n_files": len(file_diffs),
        "file_summaries": [
            {"filename": fn, "summary": s}
            for fn, s in file_summaries
        ],
        "timing": {
            "summary_wall_s": round(t_summary_total, 4),
            "cmg_wall_s": round(t_cmg, 4),
            "total_wall_s": round(t_total, 4),
        },
        "token_counts": {
            "summary_tokens_per_file": summary_token_counts,
            "cmg_tokens": cmg_tokens,
        },
        "dllm_stats": {
            "total_steps": summary_total_steps,
            "tokens_per_step": round(tokens_per_step, 3),
            "tokens_per_second": round(summary_tps, 2),
            "batch_sizes_used": batch_sizes_used,
        },
        "llm_stats": {
            "prompt_tokens": cmg_prompt_tokens,
            "generated_tokens": cmg_tokens,
            "tokens_per_second": round(tps_cmg, 2),
        },
        "config_snapshot": _make_config_snapshot(cfg, "dllm_llm"),
        "error": None,
    }


def process_llm_llm(task: dict, models: ModelManager, cfg: dict) -> dict:
    """LLM summarises per-file diffs, then LLM generates commit message."""
    sum_tokenizer, sum_model = models.get_llm_summariser()
    gen_tokenizer, gen_model = models.get_llm_generator()
    sum_device = models.device_map["llm_summariser"]
    gen_device = models.device_map["llm_generator"]
    sum_cfg = cfg["llm_summariser"]
    gen_cfg = cfg["llm_generator"]
    summary_cfg = cfg["summary"]

    sum_gen_kwargs = {
        "do_sample": sum_cfg["do_sample"],
        "temperature": sum_cfg["temperature"],
        "top_p": sum_cfg["top_p"],
        "repetition_penalty": sum_cfg["repetition_penalty"],
    }
    gen_gen_kwargs = {
        "do_sample": gen_cfg["do_sample"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg["top_p"],
        "repetition_penalty": gen_cfg["repetition_penalty"],
    }

    task_id = task.get("task_id", "unknown")
    label = task.get("label", "")

    # ── Step 0: per-file diffs ───────────────────────────────────────────
    file_diffs = get_per_file_diffs(task)

    # Resolve diff cap (use summariser tokenizer for adaptive)
    max_diff_chars, max_diff_tokens, cap_label = resolve_diff_cap(
        summary_cfg["diff_cap"], file_diffs, sum_tokenizer,
    )
    log.debug(f"[{task_id}] diff_cap resolved: {cap_label}")

    # ── Step 1: LLM file summarisation (sequential, from 26_eval) ────────
    file_summaries = []
    summary_token_counts = []
    t_summary_total = 0.0
    summary_total_tokens = 0

    for fn, fd in file_diffs:
        # Token-level truncation
        if max_diff_tokens is not None:
            raw_ids = sum_tokenizer(fd, return_tensors="pt")["input_ids"][0]
            if len(raw_ids) > max_diff_tokens:
                log.debug(
                    f"[{task_id}] Truncating {fn}: {len(raw_ids)} → {max_diff_tokens} tokens"
                )
                fd = sum_tokenizer.decode(
                    raw_ids[:max_diff_tokens], skip_special_tokens=True
                )

        char_cap = None if max_diff_tokens is not None else max_diff_chars
        messages = build_summary_messages(fn, fd, max_diff_chars=char_cap)
        summary_text, n_tok, wall_s, _ = run_llm_single(
            messages, sum_model, sum_tokenizer, sum_device,
            sum_gen_kwargs, summary_cfg["max_new_tokens"],
        )
        file_summaries.append((fn, summary_text))
        summary_token_counts.append(n_tok)
        t_summary_total += wall_s
        summary_total_tokens += n_tok

    # ── Step 2: LLM commit message generation ────────────────────────────
    cmg_messages = build_cmg_messages(file_summaries)
    cmg_text, cmg_tokens, t_cmg, cmg_prompt_tokens = run_llm_single(
        cmg_messages, gen_model, gen_tokenizer, gen_device,
        gen_gen_kwargs, cfg["cmg"]["max_new_tokens"],
    )

    t_total = t_summary_total + t_cmg
    eps = 1e-9
    tps_cmg = cmg_tokens / (t_cmg + eps)
    summary_tps = summary_total_tokens / (t_summary_total + eps)

    return {
        "task_id": task_id,
        "pipeline_mode": "llm_llm",
        "label": label,
        "generated": cmg_text,
        "n_files": len(file_diffs),
        "file_summaries": [
            {"filename": fn, "summary": s}
            for fn, s in file_summaries
        ],
        "timing": {
            "summary_wall_s": round(t_summary_total, 4),
            "cmg_wall_s": round(t_cmg, 4),
            "total_wall_s": round(t_total, 4),
        },
        "token_counts": {
            "summary_tokens_per_file": summary_token_counts,
            "cmg_tokens": cmg_tokens,
        },
        "dllm_stats": None,
        "llm_summary_stats": {
            "total_tokens": summary_total_tokens,
            "tokens_per_second": round(summary_tps, 2),
        },
        "llm_stats": {
            "prompt_tokens": cmg_prompt_tokens,
            "generated_tokens": cmg_tokens,
            "tokens_per_second": round(tps_cmg, 2),
        },
        "config_snapshot": _make_config_snapshot(cfg, "llm_llm"),
        "error": None,
    }


# ─── Config snapshot helper ──────────────────────────────────────────────────

def _make_config_snapshot(cfg: dict, mode: str) -> dict:
    """Return a minimal config snapshot relevant to the given mode."""
    snap = {"mode": mode, "devices": cfg["devices"]}
    match mode:
        case "dllm_only":
            snap["dllm"] = cfg["dllm"]
            snap["cmg"] = cfg["cmg"]
        case "llm_only":
            snap["llm_generator"] = cfg["llm_generator"]
            snap["cmg"] = cfg["cmg"]
        case "dllm_llm":
            snap["dllm"] = cfg["dllm"]
            snap["llm_generator"] = cfg["llm_generator"]
            snap["summary"] = cfg["summary"]
            snap["cmg"] = cfg["cmg"]
        case "llm_llm":
            snap["llm_summariser"] = cfg["llm_summariser"]
            snap["llm_generator"] = cfg["llm_generator"]
            snap["summary"] = cfg["summary"]
            snap["cmg"] = cfg["cmg"]
    return snap


# ─── Mode dispatcher ─────────────────────────────────────────────────────────

MODE_PROCESSORS = {
    "dllm_only": process_dllm_only,
    "llm_only": process_llm_only,
    "dllm_llm": process_dllm_llm,
    "llm_llm": process_llm_llm,
}


# ─── Checkpoint / resume ─────────────────────────────────────────────────────

def _result_key(task_id: str, mode: str) -> str:
    return f"{task_id}_{mode}"


def load_existing_results(output_file: str, resume: bool) -> tuple[list[dict], set[str]]:
    """Load existing results from output_file and checkpoint. Return (results, done_keys)."""
    results = []
    done_keys = set()

    if not resume:
        return results, done_keys

    # Load final output file
    out_path = Path(output_file)
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    key = _result_key(r.get("task_id", ""), r.get("pipeline_mode", ""))
                    if key not in done_keys:
                        results.append(r)
                        done_keys.add(key)
                log.info(f"Resumed {len(results)} results from {output_file}")
        except Exception as e:
            log.warning(f"Could not load {output_file}: {e}")

    # Load checkpoint file
    ckpt_path = Path(output_file + ".ckpt.jsonl")
    if ckpt_path.exists():
        n_ckpt = 0
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    key = _result_key(r.get("task_id", ""), r.get("pipeline_mode", ""))
                    if key not in done_keys:
                        results.append(r)
                        done_keys.add(key)
                        n_ckpt += 1
            if n_ckpt:
                log.info(f"Resumed {n_ckpt} additional results from checkpoint")
        except Exception as e:
            log.warning(f"Could not load checkpoint: {e}")

    return results, done_keys


def append_checkpoint(result: dict, ckpt_path: str):
    """Append a single result to the checkpoint file (thread-safe)."""
    with _CKPT_LOCK:
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def save_final_results(results: list[dict], output_file: str):
    """Write final results JSON array and remove checkpoint."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved → {output_file}")

    ckpt_path = Path(output_file + ".ckpt.jsonl")
    if ckpt_path.exists():
        ckpt_path.unlink()
        log.info("Checkpoint file cleaned up.")


# ─── GPU warm-up ─────────────────────────────────────────────────────────────

_WARMUP_MESSAGES = [
    {"role": "system", "content": "You are a developer."},
    {"role": "user", "content": "Write a commit message for: renamed variable x to y."},
]


def warmup_models(models: ModelManager, modes: list[str], cfg: dict, n_rounds: int = 3):
    """
    Run a few throwaway inference passes to warm up the GPU.

    This ensures CUDA kernels are compiled/cached, memory allocators are
    primed, and timing measurements on actual tasks are not contaminated by
    first-run overhead (cuDNN autotuner, torch.compile tracing, etc.).
    """
    log.info(f"Running {n_rounds} warm-up round(s) per model ...")

    needs_dllm = any(m in modes for m in ("dllm_only", "dllm_llm"))
    needs_llm_gen = any(m in modes for m in ("llm_only", "dllm_llm", "llm_llm"))
    needs_llm_sum = "llm_llm" in modes

    for r in range(n_rounds):
        # Warm up dLLM
        if needs_dllm:
            dllm_tok, dllm_model = models.get_dllm()
            device = models.device_map["dllm"]
            dllm_cfg = cfg["dllm"]
            gen_kwargs = {
                "block_size": dllm_cfg["block_size"],
                "small_block_size": dllm_cfg["small_block_size"],
                "threshold": dllm_cfg["threshold"],
                "use_block_cache": dllm_cfg["use_block_cache"],
                "temperature": dllm_cfg["temperature"],
                "top_p": dllm_cfg["top_p"],
            }
            try:
                run_dllm_batch(
                    [_WARMUP_MESSAGES], dllm_model, dllm_tok, device,
                    gen_kwargs, max_new_tokens=cfg["summary"]["max_new_tokens"],
                )
            except Exception as e:
                log.warning(f"dLLM warm-up round {r+1} failed: {e}")

        # Warm up LLM generator
        if needs_llm_gen:
            llm_tok, llm_model = models.get_llm_generator()
            device = models.device_map["llm_generator"]
            llm_cfg = cfg["llm_generator"]
            gen_kwargs = {
                "do_sample": llm_cfg["do_sample"],
                "temperature": llm_cfg["temperature"],
                "top_p": llm_cfg["top_p"],
                "repetition_penalty": llm_cfg["repetition_penalty"],
            }
            try:
                run_llm_single(
                    _WARMUP_MESSAGES, llm_model, llm_tok, device,
                    gen_kwargs, max_new_tokens=cfg["cmg"]["max_new_tokens"],
                )
            except Exception as e:
                log.warning(f"LLM generator warm-up round {r+1} failed: {e}")

        # Warm up LLM summariser
        if needs_llm_sum:
            sum_tok, sum_model = models.get_llm_summariser()
            device = models.device_map["llm_summariser"]
            sum_cfg = cfg["llm_summariser"]
            gen_kwargs = {
                "do_sample": sum_cfg["do_sample"],
                "temperature": sum_cfg["temperature"],
                "top_p": sum_cfg["top_p"],
                "repetition_penalty": sum_cfg["repetition_penalty"],
            }
            try:
                run_llm_single(
                    _WARMUP_MESSAGES, sum_model, sum_tok, device,
                    gen_kwargs, max_new_tokens=cfg["summary"]["max_new_tokens"],
                )
            except Exception as e:
                log.warning(f"LLM summariser warm-up round {r+1} failed: {e}")

    # Clear any memory used by warm-up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Warm-up complete.")


# ─── Smoke-test output ──────────────────────────────────────────────────────

def _print_sample_result(result: dict):
    """Print a clearly formatted output block for visual smoke-test inspection."""
    mode = result.get("pipeline_mode", "?")
    task_id = result.get("task_id", "?")
    label = result.get("label", "").strip()
    generated = result.get("generated", "").strip()
    timing = result.get("timing", {})
    cmg_tok = result.get("token_counts", {}).get("cmg_tokens", 0)
    t_total = timing.get("total_wall_s", 0.0)
    t_sum = timing.get("summary_wall_s", 0.0)
    t_cmg = timing.get("cmg_wall_s", 0.0)
    error = result.get("error")

    sep = "═" * 60
    print()
    print(sep)
    print(f"  SMOKE TEST  │  mode={mode}  │  task={task_id}")
    print(sep)
    if error:
        print(f"  ✗  ERROR: {error}")
    else:
        print(f"  LABEL     : {label}")
        print(f"  GENERATED : {generated}")
        print()
        print(f"  timing    : total={t_total:.2f}s  "
              f"(summary={t_sum:.2f}s + cmg={t_cmg:.2f}s)  "
              f"cmg_tokens={cmg_tok}")

        # Show dLLM stats if present
        dstats = result.get("dllm_stats")
        if dstats:
            print(f"  dllm      : steps={dstats['total_steps']}  "
                  f"tok/step={dstats['tokens_per_step']:.2f}  "
                  f"batches={dstats['batch_sizes_used']}")

        # Show LLM stats if present
        lstats = result.get("llm_stats")
        if lstats:
            print(f"  llm       : prompt_tok={lstats['prompt_tokens']}  "
                  f"gen_tok={lstats['generated_tokens']}  "
                  f"tok/s={lstats['tokens_per_second']:.1f}")

        # Show per-file summaries if present
        summaries = result.get("file_summaries", [])
        if summaries:
            print(f"  summaries ({len(summaries)} file(s)):")
            for i, s in enumerate(summaries[:3], 1):  # cap at 3 to avoid flooding
                snippet = s.get("summary", "").replace("\n", " ")[:120]
                print(f"    [{i}] {s.get('filename', '?')}: {snippet}")
            if len(summaries) > 3:
                print(f"    ... and {len(summaries) - 3} more")
    print(sep)
    print()


# ─── Summary table ───────────────────────────────────────────────────────────

def print_summary_table(results: list[dict], output_file: str):
    """Print a quick summary table at the end of the run."""
    from collections import defaultdict

    by_mode = defaultdict(lambda: {"tasks": 0, "errors": 0, "total_wall": 0.0, "cmg_tokens": 0})

    for r in results:
        mode = r.get("pipeline_mode", "unknown")
        entry = by_mode[mode]
        entry["tasks"] += 1
        if r.get("error"):
            entry["errors"] += 1
        else:
            entry["total_wall"] += r.get("timing", {}).get("total_wall_s", 0.0)
            entry["cmg_tokens"] += r.get("token_counts", {}).get("cmg_tokens", 0)

    print()
    print("Pipeline results summary")
    print("─" * 60)
    print(f"{'Mode':<14} {'Tasks':>6} {'Errors':>7} {'Avg total_wall_s':>17} {'Avg cmg_tokens':>15}")

    for mode in ["dllm_only", "llm_only", "dllm_llm", "llm_llm"]:
        if mode not in by_mode:
            continue
        e = by_mode[mode]
        n_ok = e["tasks"] - e["errors"]
        avg_wall = e["total_wall"] / max(n_ok, 1)
        avg_tok = e["cmg_tokens"] / max(n_ok, 1)
        print(
            f"{mode:<14} {e['tasks']:>6} {e['errors']:>7} "
            f"{avg_wall:>16.1f}s {avg_tok:>14.0f}"
        )

    print("─" * 60)
    print(f"Results saved → {output_file}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified CMG Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to JSON config file. Deep-merged onto defaults.")
    p.add_argument("--input", default=None, help="Override config.input")
    p.add_argument("--output-file", default=None, help="Override config.output_file")
    p.add_argument("--modes", default=None,
                   help="Comma-separated modes (e.g. dllm_llm,llm_llm)")
    p.add_argument("--devices", default=None,
                   help="Comma-separated devices (e.g. cuda:0,cuda:1)")
    p.add_argument("--sample", nargs="?", const=1, type=int, default=None,
                   help="Smoke-test mode. '--sample' alone runs 1 task per mode "
                        "and prints generated output for visual inspection. "
                        "'--sample N' runs the first N tasks. Disables resume so "
                        "results are always regenerated fresh.")
    p.add_argument("--task-ids", default=None,
                   help="Comma-separated task IDs to run")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing output and start fresh")
    p.add_argument("--no-summaries", action="store_true",
                   help="Suppress per-file summaries from stdout")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip GPU warm-up rounds (not recommended for benchmarks)")
    p.add_argument("--warmup-rounds", type=int, default=3,
                   help="Number of warm-up inference passes per model (default: 3)")
    p.add_argument("--grid-search", action="store_true",
                   help="Enable grid search mode. Any config parameter that is a list "
                        "of scalars will be treated as a search axis. All combinations "
                        "are enumerated and run sequentially. Each combination writes "
                        "to a separate output file suffixed with the parameter values.")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load & validate config ───────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # ── Grid search mode ─────────────────────────────────────────────────
    if args.grid_search:
        _run_grid_search(cfg, args)
        return

    # ── Standard single-config run ───────────────────────────────────────
    _run_single(cfg, args)


def _run_grid_search(base_cfg: dict, args: argparse.Namespace):
    """Run all grid-search combinations sequentially, sharing loaded models."""
    grid_configs = generate_grid_configs(base_cfg)
    if len(grid_configs) <= 1:
        log.warning("Grid search requested but no list-valued parameters found in config. "
                     "Running single configuration.")
        _run_single(base_cfg, args)
        return

    base_output = base_cfg["output_file"]
    log.info("=" * 60)
    log.info(f"GRID SEARCH: {len(grid_configs)} configurations")
    log.info("=" * 60)

    # Validate first config to fail fast
    first_cfg, _ = grid_configs[0]
    validate_config(first_cfg)

    # Load models once (they are shared across grid configs since model names
    # don't change — only generation hyperparameters do)
    modes = first_cfg["modes"]
    device_map = resolve_devices(first_cfg)
    models = ModelManager(first_cfg, device_map)

    needs_dllm = any(m in modes for m in ("dllm_only", "dllm_llm"))
    needs_llm_gen = any(m in modes for m in ("llm_only", "dllm_llm", "llm_llm"))
    needs_llm_sum = "llm_llm" in modes

    if needs_dllm:
        models.get_dllm()
    if needs_llm_gen:
        models.get_llm_generator()
    if needs_llm_sum:
        models.get_llm_summariser()

    if not args.no_warmup:
        warmup_models(models, modes, first_cfg, n_rounds=args.warmup_rounds)

    # Load tasks once
    tasks = _load_tasks_from_config(first_cfg)

    # SIGINT handler
    _interrupted = threading.Event()
    def _sigint_handler(sig, frame):
        log.warning("SIGINT received — saving current grid config and exiting ...")
        _interrupted.set()
    signal.signal(signal.SIGINT, _sigint_handler)

    # Track grid-level summary
    grid_summary = []

    for gi, (grid_cfg, label) in enumerate(grid_configs, 1):
        if _interrupted.is_set():
            break

        output_file = grid_search_output_path(base_output, label)
        grid_cfg["output_file"] = output_file
        ckpt_path = output_file + ".ckpt.jsonl"

        log.info(f"\n{'═' * 60}")
        log.info(f"Grid config {gi}/{len(grid_configs)}: {label}")
        log.info(f"  Output → {output_file}")
        log.info(f"{'═' * 60}")

        # Update the ModelManager's config reference so processors see new hyperparams
        models.cfg = grid_cfg

        results, done_keys = load_existing_results(output_file, grid_cfg.get("resume", True))

        n_done, n_err, n_skipped = _process_tasks(
            tasks, modes, grid_cfg, models, results, done_keys,
            output_file, ckpt_path, _interrupted,
            is_sample_run=False,
        )

        save_final_results(results, output_file)

        grid_summary.append({
            "label": label,
            "output_file": output_file,
            "n_done": n_done,
            "n_err": n_err,
            "n_skipped": n_skipped,
        })

        log.info(f"Grid config {gi}: {n_done} done, {n_err} errors, {n_skipped} skipped")

    # Write grid summary
    summary_path = str(Path(base_output).parent / "grid_search_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(grid_summary, f, indent=2)
    log.info(f"\nGrid search complete. Summary → {summary_path}")

    # Print grid summary table
    print(f"\n{'═' * 60}")
    print(f"  GRID SEARCH COMPLETE — {len(grid_summary)} configurations")
    print(f"{'═' * 60}")
    for entry in grid_summary:
        print(f"  {entry['label']}: {entry['n_done']} done, {entry['n_err']} err → {entry['output_file']}")
    print()


def _run_single(cfg: dict, args: argparse.Namespace):
    """Standard single-configuration run (original main logic)."""
    validate_config(cfg)

    modes = cfg["modes"]
    output_file = cfg["output_file"]
    ckpt_path = output_file + ".ckpt.jsonl"

    # --sample forces no-resume so results are always generated fresh
    is_sample_run = cfg.get("sample") is not None
    if is_sample_run:
        cfg["resume"] = False
        log.info(
            f"Sample mode: {cfg['sample']} task(s) per mode. "
            "Resume disabled — results will be regenerated."
        )

    log.info("=" * 60)
    log.info("Unified CMG Evaluation Pipeline")
    log.info(f"  modes   : {modes}")
    log.info(f"  devices : {cfg['devices']}")
    log.info(f"  input   : {cfg['input']}")
    log.info(f"  output  : {output_file}")
    log.info(f"  resume  : {cfg['resume']}")
    if is_sample_run:
        log.info(f"  sample  : {cfg['sample']} task(s) per mode (smoke-test)")
    log.info("=" * 60)

    # ── Load tasks ───────────────────────────────────────────────────────
    tasks = _load_tasks_from_config(cfg)

    if not tasks:
        log.warning("No tasks to process. Exiting.")
        return

    # ── Resume ───────────────────────────────────────────────────────────
    results, done_keys = load_existing_results(output_file, cfg["resume"])

    # ── Resolve devices & load models ────────────────────────────────────
    device_map = resolve_devices(cfg)
    log.info(f"Device map: {device_map}")

    models = ModelManager(cfg, device_map)

    # Pre-load models that will be needed (so we fail fast on load errors)
    needs_dllm = any(m in modes for m in ("dllm_only", "dllm_llm"))
    needs_llm_gen = any(m in modes for m in ("llm_only", "dllm_llm", "llm_llm"))
    needs_llm_sum = "llm_llm" in modes

    if needs_dllm:
        models.get_dllm()
    if needs_llm_gen:
        models.get_llm_generator()
    if needs_llm_sum:
        models.get_llm_summariser()

    # ── GPU warm-up ──────────────────────────────────────────────────────
    if not args.no_warmup:
        warmup_models(models, modes, cfg, n_rounds=args.warmup_rounds)
    else:
        log.info("Warm-up skipped (--no-warmup).")

    # ── SIGINT handler for graceful shutdown ──────────────────────────────
    _interrupted = threading.Event()

    def _sigint_handler(sig, frame):
        log.warning("SIGINT received — saving progress and exiting ...")
        _interrupted.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Process tasks ────────────────────────────────────────────────────
    n_done, n_err, n_skipped = _process_tasks(
        tasks, modes, cfg, models, results, done_keys,
        output_file, ckpt_path, _interrupted,
        is_sample_run=is_sample_run,
    )

    # ── Save final results ───────────────────────────────────────────────
    save_final_results(results, output_file)
    print_summary_table(results, output_file)

    log.info(
        f"Complete: {n_done} done, {n_err} errors, {n_skipped} skipped (resumed)."
    )


def _load_tasks_from_config(cfg: dict) -> list[dict]:
    """Load and filter tasks according to config."""
    tasks = []
    with open(cfg["input"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    if cfg.get("task_ids"):
        allowed = set(cfg["task_ids"])
        tasks = [t for t in tasks if t.get("task_id") in allowed]
        log.info(f"Filtered to {len(tasks)} tasks by task_ids")

    if cfg.get("sample"):
        tasks = tasks[:cfg["sample"]]

    log.info(f"Loaded {len(tasks)} tasks from {cfg['input']}")
    return tasks


def _process_tasks(
    tasks: list[dict],
    modes: list[str],
    cfg: dict,
    models: ModelManager,
    results: list[dict],
    done_keys: set[str],
    output_file: str,
    ckpt_path: str,
    interrupted: threading.Event,
    is_sample_run: bool = False,
) -> tuple[int, int, int]:
    """Core task processing loop. Returns (n_done, n_err, n_skipped)."""
    n_skipped = 0
    n_done = 0
    n_err = 0

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if interrupted.is_set():
            break

        processor = MODE_PROCESSORS[mode]
        log.info(f"── Mode: {mode} ({'─' * 40})")

        for task in tasks:
            if interrupted.is_set():
                break

            task_id = task.get("task_id", "unknown")
            key = _result_key(task_id, mode)

            if key in done_keys:
                n_skipped += 1
                continue

            t_start = time.perf_counter()

            try:
                result = processor(task, models, cfg)

                # True end-to-end wall time (includes preprocessing, sorting, etc.)
                wall = time.perf_counter() - t_start
                result["timing"]["task_wall_s"] = round(wall, 4)

                # Optionally strip summaries from output
                if cfg.get("no_summaries_in_output"):
                    result["file_summaries"] = []

                results.append(result)
                done_keys.add(key)
                append_checkpoint(result, ckpt_path)
                n_done += 1

                cmg_tok = result.get("token_counts", {}).get("cmg_tokens", 0)
                log.info(
                    f"[{task_id}/{mode}] {wall:.2f}s | cmg_tok={cmg_tok}"
                )

                if is_sample_run:
                    _print_sample_result(result)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning(f"[{task_id}/{mode}] OOM — skipping.")
                error_result = {
                    "task_id": task_id,
                    "pipeline_mode": mode,
                    "label": task.get("label", ""),
                    "generated": "",
                    "file_summaries": [],
                    "timing": {"summary_wall_s": 0, "cmg_wall_s": 0, "total_wall_s": 0},
                    "token_counts": {"summary_tokens_per_file": [], "cmg_tokens": 0},
                    "dllm_stats": None,
                    "llm_stats": None,
                    "config_snapshot": _make_config_snapshot(cfg, mode),
                    "error": "OutOfMemoryError",
                }
                results.append(error_result)
                done_keys.add(key)
                append_checkpoint(error_result, ckpt_path)
                n_err += 1

            except Exception as e:
                log.warning(
                    f"[{task_id}/{mode}] Error: {e}\n{traceback.format_exc()}"
                )
                error_result = {
                    "task_id": task_id,
                    "pipeline_mode": mode,
                    "label": task.get("label", ""),
                    "generated": "",
                    "file_summaries": [],
                    "timing": {"summary_wall_s": 0, "cmg_wall_s": 0, "total_wall_s": 0},
                    "token_counts": {"summary_tokens_per_file": [], "cmg_tokens": 0},
                    "dllm_stats": None,
                    "llm_stats": None,
                    "config_snapshot": _make_config_snapshot(cfg, mode),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                results.append(error_result)
                done_keys.add(key)
                append_checkpoint(error_result, ckpt_path)
                n_err += 1

    return n_done, n_err, n_skipped


if __name__ == "__main__":
    main()
