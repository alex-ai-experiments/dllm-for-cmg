#!/usr/bin/env python3
"""
MDM Grid-Search Inference Script for Commit Message Generation
Model: Efficient-Large-Model/Fast_dLLM_v2_1.5B

Uses the Fast-dLLM batch_sample function for batched inference.

Supports:
  - Grid search over thresholds, block sizes, small block sizes, cache states, batch sizes
  - Batched inference via model.batch_sample (from Fast-dLLM generation_functions)
  - Multi-GPU parallelism with model reuse across experiments
  - Dynamic output sub-directories per experiment config
  - Experiment summary CSV + JSONL
  - Graceful resumption

Usage:
    python 10_new_eval.py -i tasks.jsonl -o results/ --sample
    python 10_new_eval.py -i tasks.jsonl -o results/ \\
        --thresholds 0.9,0.95 --block-sizes 32,64 --small-block-sizes 4,8 \\
        --batch-sizes 1,4,8 --test-cache-states
"""

import argparse
import csv
import itertools
import json
import logging
import sys
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from generation_functions import (
    Fast_dLLM_QwenForCausalLM,
    FAST_DLLM_MASK_ID,
    FAST_DLLM_STOP_TOKEN,
)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
MASK_ID = FAST_DLLM_MASK_ID
STOP_TOKEN = FAST_DLLM_STOP_TOKEN


# ─── Model Loading ────────────────────────────────────────────────────────────

# Serialise concurrent from_pretrained calls.
# transformers >= 4.45 with accelerate installed uses init_empty_weights() which
# mutates global state; two threads calling it simultaneously corrupt each other.
_MODEL_LOAD_LOCK = __import__("threading").Lock()


def load_model(device: str):
    """
    Load tokenizer + model pinned to a specific device.
    Binds the batch_sample method from generation_functions for batched inference.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"[{device}] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    log.info(f"[{device}] Loading model …")
    # Serialise loading: accelerate's init_empty_weights uses global state that
    # breaks under concurrent threads.  low_cpu_mem_usage=False disables meta
    # tensors entirely so lm_head.weight (uninitialised in checkpoint) is never a
    # meta tensor and .to(device) succeeds.
    with _MODEL_LOAD_LOCK:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
    model = model.to(device)
    model.eval()

    # Bind batch_sample for batched inference (matches NVlabs/Fast-dLLM eval.py)
    model.mdm_sample = types.MethodType(
        Fast_dLLM_QwenForCausalLM.batch_sample, model
    )

    log.info(f"[{device}] Model ready.")
    return tokenizer, model


# ─── Config naming ────────────────────────────────────────────────────────────

def make_config_name(block_size, small_block_size, threshold, use_block_cache, batch_size, max_new_tokens):
    return f"bs{block_size}_sbs{small_block_size}_th{threshold}_cache{use_block_cache}_batch{batch_size}_mnt{max_new_tokens}"


# ─── Batched inference ────────────────────────────────────────────────────────

def run_batch(
    tasks: list[dict],
    model,
    tokenizer,
    device: str,
    gen_kwargs: dict,
    max_new_tokens: int,
) -> tuple[list[dict], float, int, int]:
    """
    Run batch_sample for a batch of tasks.
    Returns (per_task_results, wall_seconds, total_generated_tokens, total_steps).
    """
    # Tokenize
    input_ids_list = []
    seq_lens = []
    for task in tasks:
        text = tokenizer.apply_chat_template(
            task["messages"], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer([text], return_tensors="pt")["input_ids"][0]
        input_ids_list.append(ids)
        seq_lens.append(len(ids))

    max_len = max(seq_lens)
    min_len = min(seq_lens)

    # Pad with MASK_ID to max length in batch (matches Fast-dLLM eval.py)
    padded = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), MASK_ID, dtype=torch.long)])
        padded.append(ids.unsqueeze(0))
    batched_ids = torch.cat(padded, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, device=device)

    # Synchronize for accurate timing
    if device != "cpu":
        torch.cuda.synchronize(device)

    t_start = time.perf_counter()

    generated_ids, total_steps = model.mdm_sample(
        batched_ids,
        tokenizer=tokenizer,
        block_size=gen_kwargs["block_size"],
        small_block_size=gen_kwargs["small_block_size"],
        max_new_tokens=max_new_tokens,
        mask_id=MASK_ID,
        min_len=min_len,
        seq_len=seq_len_tensor,
        use_block_cache=gen_kwargs["use_block_cache"],
        threshold=gen_kwargs["threshold"],
        temperature=gen_kwargs.get("temperature", 0.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        stop_token=STOP_TOKEN,
    )

    if device != "cpu":
        torch.cuda.synchronize(device)

    wall_seconds = time.perf_counter() - t_start

    # Decode results and count tokens (matches Fast-dLLM eval.py: != mask_id)
    results = []
    total_gen_tokens = 0
    for i, task in enumerate(tasks):
        gen_output = generated_ids[i]
        prompt_len = seq_lens[i]
        gen_part = gen_output[prompt_len:]

        gen_tokens = int((gen_part != MASK_ID).sum().item())
        total_gen_tokens += gen_tokens

        generated_text = tokenizer.decode(gen_part, skip_special_tokens=True)

        results.append({
            "task_id": task["task_id"],
            "generated": generated_text,
            "label": task.get("label"),
            "device": device,
            "prompt_tokens": prompt_len,
            "generated_tokens": gen_tokens,
        })

    return results, wall_seconds, total_gen_tokens, total_steps


# ─── Batch result persistence helper ─────────────────────────────────────────

def _save_batch_results(
    batch_results: list[dict],
    wall_seconds: float,
    total_tokens: int,
    total_steps: int,
    actual_batch_size: int,
    gen_kwargs: dict,
    output_dir: Path,
    results: list,
    batch_stats: list,
):
    eps = 1e-9
    tps = total_tokens / (wall_seconds + eps)
    mspt = wall_seconds / (total_tokens + eps) * 1000
    tokens_per_step = total_tokens / max(total_steps, 1)

    batch_stats.append({
        "batch_wall_seconds": round(wall_seconds, 4),
        "batch_total_tokens": total_tokens,
        "batch_total_steps": total_steps,
        "batch_size": actual_batch_size,
        "batch_tokens_per_second": round(tps, 2),
        "batch_ms_per_token": round(mspt, 3),
        "batch_tokens_per_step": round(tokens_per_step, 3),
    })

    for r in batch_results:
        result = {
            "task_id": r["task_id"],
            "generated": r["generated"],
            "label": r["label"],
            "device": r["device"],
            "gen_kwargs": gen_kwargs,
            "stats": {
                "prompt_tokens": r["prompt_tokens"],
                "generated_tokens": r["generated_tokens"],
                "batch_wall_seconds": round(wall_seconds, 4),
                "batch_size": actual_batch_size,
                "batch_total_tokens": total_tokens,
                "batch_total_steps": total_steps,
                "batch_tokens_per_second": round(tps, 2),
                "effective_ms_per_token": round(mspt, 3),
                "tokens_per_step": round(tokens_per_step, 3),
            },
        }

        out_path = output_dir / f"{r['task_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)


# ─── Process tasks in batches with resume ─────────────────────────────────────

def process_tasks(
    tasks: list[dict],
    model,
    tokenizer,
    device: str,
    gen_kwargs: dict,
    max_new_tokens: int,
    batch_size: int,
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Process all tasks in batches with resume support.

    OOM handling:
      1. On torch.cuda.OutOfMemoryError the VRAM cache is cleared immediately
         so subsequent batches are not also poisoned.
      2. If batch_size > 1, the failed batch is retried one task at a time
         so individual results are still saved instead of being lost.
    """
    results = []
    errors = []
    batch_stats = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[batch_start:batch_start + batch_size]

        # Resume: check which are already done
        pending = []
        for task in batch_tasks:
            out_path = output_dir / f"{task['task_id']}.json"
            if out_path.exists():
                try:
                    with open(out_path, encoding="utf-8") as f:
                        results.append(json.load(f))
                    log.info(f"[{task['task_id']}] Already done — skipping.")
                    continue
                except Exception:
                    pass  # re-process if corrupted
            pending.append(task)

        if not pending:
            continue

        try:
            batch_results, wall_seconds, total_tokens, total_steps = run_batch(
                pending, model, tokenizer, device, gen_kwargs, max_new_tokens
            )
            _save_batch_results(
                batch_results, wall_seconds, total_tokens, total_steps, len(pending),
                gen_kwargs, output_dir, results, batch_stats,
            )
            log.info(
                f"[{device}] batch {batch_start // batch_size + 1} | "
                f"{len(pending)} tasks | {wall_seconds:.2f}s | "
                f"{total_tokens} tok | {total_tokens / max(wall_seconds, 1e-9):.1f} tok/s | "
                f"{total_tokens / max(total_steps, 1):.2f} tok/step"
            )

        except torch.cuda.OutOfMemoryError as oom:
            # ── Step 1: release the stuck VRAM allocation immediately ────
            torch.cuda.empty_cache()
            log.warning(
                f"[{device}] OOM on batch of {len(pending)} at "
                f"batch_start={batch_start}. Clearing cache."
            )

            # ── Step 2: retry one-by-one if the original batch was larger ─
            if len(pending) == 1:
                # Single task OOM — nothing we can do; record the error
                task = pending[0]
                log.error(f"[{task['task_id']}] OOM even with batch_size=1; skipping.")
                err = {
                    "task_id": task["task_id"],
                    "device": device,
                    "error": "OutOfMemoryError (batch_size=1)",
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                err_path = output_dir / f"{task['task_id']}.error.json"
                try:
                    with open(err_path, "w") as f:
                        json.dump(err, f, indent=2)
                except Exception:
                    pass
                torch.cuda.empty_cache()
            else:
                log.info(
                    f"[{device}] Falling back to batch_size=1 for "
                    f"{len(pending)} task(s)."
                )
                for task in pending:
                    torch.cuda.empty_cache()
                    try:
                        single_results, wall_seconds, total_tokens, total_steps = run_batch(
                            [task], model, tokenizer, device, gen_kwargs, max_new_tokens
                        )
                        _save_batch_results(
                            single_results, wall_seconds, total_tokens, total_steps, 1,
                            gen_kwargs, output_dir, results, batch_stats,
                        )
                        log.info(
                            f"[{task['task_id']}] fallback OK | "
                            f"{wall_seconds:.2f}s | {total_tokens} tok | "
                            f"{total_tokens / max(total_steps, 1):.2f} tok/step"
                        )
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log.error(
                            f"[{task['task_id']}] OOM even with batch_size=1; skipping."
                        )
                        err = {
                            "task_id": task["task_id"],
                            "device": device,
                            "error": "OutOfMemoryError (batch_size=1 fallback)",
                            "traceback": traceback.format_exc(),
                        }
                        errors.append(err)
                        err_path = output_dir / f"{task['task_id']}.error.json"
                        try:
                            with open(err_path, "w") as f:
                                json.dump(err, f, indent=2)
                        except Exception:
                            pass
                    except Exception as e:
                        log.error(
                            f"[{task['task_id']}] fallback error: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                        err = {
                            "task_id": task["task_id"],
                            "device": device,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        errors.append(err)
                        err_path = output_dir / f"{task['task_id']}.error.json"
                        try:
                            with open(err_path, "w") as f:
                                json.dump(err, f, indent=2)
                        except Exception:
                            pass

        except Exception as e:
            log.error(f"Batch error on {device}: {e}\n{traceback.format_exc()}")
            for task in pending:
                err = {
                    "task_id": task["task_id"],
                    "device": device,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                err_path = output_dir / f"{task['task_id']}.error.json"
                try:
                    with open(err_path, "w") as f:
                        json.dump(err, f, indent=2)
                except Exception:
                    pass

    return results, errors, batch_stats


# ─── GPU Worker ───────────────────────────────────────────────────────────────

def gpu_worker(
    device: str,
    tasks: list[dict],
    configs: list[tuple],
    temperature: float,
    top_p: float,
    output_root: Path,
) -> tuple[dict, dict, dict]:
    """
    Worker for one GPU. Loads model once, runs all experiment configs.
    configs tuples: (block_size, small_block_size, threshold, cache, batch_size, max_new_tokens)
    Returns (all_results, all_errors, all_batch_stats) keyed by config_name.
    """
    try:
        tokenizer, model = load_model(device)
    except Exception as e:
        log.error(f"[{device}] Failed to load model: {e}\n{traceback.format_exc()}")
        err = {"device": device, "task_id": None, "error": str(e)}
        all_errors = {}
        for cfg_tuple in configs:
            cfg = make_config_name(*cfg_tuple)
            all_errors[cfg] = [err]
        return {}, all_errors, {}

    all_results = {}
    all_errors = {}
    all_batch_stats = {}

    for bs, sbs, th, cache, batch_sz, mnt in configs:
        config_name = make_config_name(bs, sbs, th, cache, batch_sz, mnt)
        subdir = output_root / config_name

        gen_kwargs = {
            "block_size": bs,
            "small_block_size": sbs,
            "threshold": th,
            "use_block_cache": cache,
            "temperature": temperature,
            "top_p": top_p,
        }

        log.info(f"[{device}] Config: {config_name} ({len(tasks)} tasks, batch={batch_sz}, mnt={mnt})")

        results, errors, b_stats = process_tasks(
            tasks, model, tokenizer, device, gen_kwargs,
            mnt, batch_sz, subdir,
        )

        all_results[config_name] = results
        all_errors[config_name] = errors
        all_batch_stats[config_name] = b_stats

        log.info(f"[{device}] {config_name}: {len(results)} done, {len(errors)} errors")

    return all_results, all_errors, all_batch_stats


# ─── Summary generation ──────────────────────────────────────────────────────

def rebuild_summary_from_individual_files(directory: Path) -> list[dict]:
    """Rebuild _results_summary.jsonl from individual .json result files in a directory."""
    results = []
    for jf in sorted(directory.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            with open(jf, encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            log.warning(f"Skipping {jf}: {e}")
    return results


def write_per_experiment_summaries(global_results, global_errors, output_root):
    """Write _results_summary.jsonl and _errors_summary.json per experiment subdir.

    Always rebuilds the summary from individual JSON files on disk so that
    resumed runs and partial completions are fully captured.
    """
    for config_name, results in global_results.items():
        subdir = output_root / config_name
        subdir.mkdir(parents=True, exist_ok=True)

        # Rebuild from individual files to capture resumed results
        all_results = rebuild_summary_from_individual_files(subdir)
        if not all_results:
            all_results = results

        summary_path = subdir / "_results_summary.jsonl"
        with open(summary_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(all_results)} results → {summary_path}")

        errors = global_errors.get(config_name, [])
        if errors:
            err_path = subdir / "_errors_summary.json"
            with open(err_path, "w") as f:
                json.dump(errors, f, indent=2)


def write_experiment_summary(global_results, global_errors, global_batch_stats, output_root):
    """
    Write experiments_summary.csv and .jsonl aggregating throughput metrics
    per hyperparameter configuration — for recreating throughput vs accuracy curves.
    """
    rows = []

    for config_name, results in sorted(global_results.items()):
        if not results:
            continue

        all_stats = [r["stats"] for r in results if "stats" in r]
        n = len(all_stats)
        if not n:
            continue

        # Aggregate from per-batch stats (avoids double-counting wall time
        # when multiple tasks share the same batch timing).
        bstats = global_batch_stats.get(config_name, [])
        if bstats:
            total_batch_tokens = sum(bs["batch_total_tokens"] for bs in bstats)
            total_batch_wall = sum(bs["batch_wall_seconds"] for bs in bstats)
            total_batch_steps = sum(bs.get("batch_total_steps", 0) for bs in bstats)
            agg_tps = total_batch_tokens / max(total_batch_wall, 1e-9)
            agg_mspt = total_batch_wall / max(total_batch_tokens, 1e-9) * 1000
            agg_toks_per_step = total_batch_tokens / max(total_batch_steps, 1)
        else:
            agg_tps = sum(s.get("batch_tokens_per_second", 0) for s in all_stats) / n
            agg_mspt = sum(s.get("effective_ms_per_token", 0) for s in all_stats) / n
            agg_toks_per_step = None
            total_batch_steps = None

        total_gen = sum(s["generated_tokens"] for s in all_stats)

        row = {
            "config": config_name,
            "num_tasks": n,
            "num_errors": len(global_errors.get(config_name, [])),
            "tokens_per_second": round(agg_tps, 2),
            "ms_per_token": round(agg_mspt, 3),
            "tokens_per_step": round(agg_toks_per_step, 3) if agg_toks_per_step is not None else None,
            "total_generated_tokens": total_gen,
            "total_diffusion_steps": total_batch_steps,
            "total_wall_seconds": round(total_batch_wall, 4) if bstats else None,
            "avg_generated_tokens": round(total_gen / n, 1),
            "avg_prompt_tokens": round(sum(s["prompt_tokens"] for s in all_stats) / n, 1),
        }
        rows.append(row)

    if not rows:
        log.warning("No experiment results to summarize.")
        return

    # CSV
    csv_path = output_root / "experiments_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Experiment summary CSV  → {csv_path}")

    # JSONL
    jsonl_path = output_root / "experiments_summary.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    log.info(f"Experiment summary JSONL → {jsonl_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MDM grid-search inference for commit message generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", "-i", required=True,
                   help="Input JSONL file (one task per line).")
    p.add_argument("--output", "-o", default="./results",
                   help="Root output directory for experiment sub-directories.")
    p.add_argument("--sample", action="store_true",
                   help="Smoke-test mode: run only the first 5 tasks.")

    # ── Grid-search hyperparameters (comma-separated) ───────────────────
    p.add_argument("--block-sizes", type=str, default="32",
                   help="Comma-separated block sizes to test.")
    p.add_argument("--small-block-sizes", type=str, default="8",
                   help="Comma-separated small block sizes to test.")
    p.add_argument("--thresholds", type=str, default="0.95",
                   help="Comma-separated confidence thresholds to test.")
    p.add_argument("--batch-sizes", type=str, default="1",
                   help="Comma-separated batch sizes (e.g. 1,2,4,8,16,32).")
    p.add_argument("--test-cache-states", action="store_true",
                   help="Test both cache=True and cache=False. "
                        "Otherwise uses --use-block-cache value only.")
    p.add_argument("--use-block-cache", action="store_true",
                   help="Default block cache state (when --test-cache-states is off).")
    p.add_argument("--max-new-tokens", type=str, default="1024",
                   help="Comma-separated max-new-tokens values to test (e.g. 256,512,1024).")

    # ── Fixed generation parameters ─────────────────────────────────────
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy).")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Nucleus sampling p.")

    return p.parse_args()


# ─── Utilities ────────────────────────────────────────────────────────────────

def detect_devices() -> list[str]:
    n = torch.cuda.device_count()
    if n == 0:
        log.warning("No CUDA GPUs detected — falling back to CPU.")
        return ["cpu"]
    devices = [f"cuda:{i}" for i in range(n)]
    log.info(f"Detected {n} GPU(s). Using: {devices}")
    return devices


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed JSON on line {lineno}: {e}")
    return tasks


def split_tasks(tasks: list, n: int) -> list[list]:
    """Round-robin split so each worker gets a balanced mix of tasks."""
    chunks: list[list] = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        chunks[i % n].append(task)
    return chunks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Parse comma-separated hyperparameter lists
    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    small_block_sizes = [int(x) for x in args.small_block_sizes.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    cache_states = [True, False] if args.test_cache_states else [args.use_block_cache]
    max_new_tokens_list = [int(x) for x in args.max_new_tokens.split(",")]

    # Cartesian product → experiment configurations
    # tuple order: (block_size, small_block_size, threshold, cache, batch_size, max_new_tokens)
    configs = list(itertools.product(
        block_sizes, small_block_sizes, thresholds, cache_states, batch_sizes, max_new_tokens_list
    ))

    log.info("=" * 60)
    log.info("MDM Grid-Search Inference — Commit Message Generation")
    log.info(f"  model           : {MODEL_NAME}")
    log.info(f"  input           : {args.input}")
    log.info(f"  output          : {args.output}")
    log.info(f"  max_new_tokens  : {max_new_tokens_list}")
    log.info(f"  temperature     : {args.temperature}")
    log.info(f"  top_p           : {args.top_p}")
    log.info(f"  configurations  : {len(configs)}")
    log.info("=" * 60)
    for c in configs:
        log.info(f"  → {make_config_name(*c)}")

    tasks = load_tasks(args.input)
    if not tasks:
        log.error("No valid tasks found. Exiting.")
        sys.exit(1)

    if args.sample:
        tasks = tasks[:5]
        log.info(f"--sample: using first {len(tasks)} task(s).")
    else:
        log.info(f"Loaded {len(tasks)} task(s).")

    tasks.sort(key=lambda t: t.get("context_length", 0))
    log.info("Tasks sorted by context_length (ascending).")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    devices = detect_devices()
    chunks = split_tasks(tasks, len(devices))
    # Re-sort each chunk so batches within a GPU worker are also length-ordered
    chunks = [sorted(chunk, key=lambda t: t.get("context_length", 0)) for chunk in chunks]

    log.info("Task distribution:")
    for dev, chunk in zip(devices, chunks):
        log.info(f"  {dev}: {len(chunk)} task(s)")

    # ── Run experiments ──────────────────────────────────────────────────
    global_results: dict[str, list] = {}
    global_errors: dict[str, list] = {}
    global_batch_stats: dict[str, list] = {}

    if len(devices) == 1:
        results, errors, batch_stats = gpu_worker(
            devices[0], chunks[0], configs,
            args.temperature, args.top_p,
            output_root,
        )
        global_results = results
        global_errors = errors
        global_batch_stats = batch_stats
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [
                pool.submit(
                    gpu_worker, dev, chunk, configs,
                    args.temperature, args.top_p,
                    output_root,
                )
                for dev, chunk in zip(devices, chunks)
            ]
            for fut in as_completed(futures):
                try:
                    results, errors, batch_stats = fut.result()
                    for cfg, res_list in results.items():
                        global_results.setdefault(cfg, []).extend(res_list)
                    for cfg, err_list in errors.items():
                        global_errors.setdefault(cfg, []).extend(err_list)
                    for cfg, bs_list in batch_stats.items():
                        global_batch_stats.setdefault(cfg, []).extend(bs_list)
                except Exception as e:
                    log.error(f"Worker thread raised: {e}")

    # ── Write summaries ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("All experiments finished. Writing summaries …")

    write_per_experiment_summaries(global_results, global_errors, output_root)
    write_experiment_summary(
        global_results, global_errors, global_batch_stats, output_root
    )

    total_tasks = sum(len(v) for v in global_results.values())
    total_errors = sum(len(v) for v in global_errors.values())
    log.info(
        f"Total: {total_tasks} task results across {len(configs)} config(s), "
        f"{total_errors} error(s)."
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()