#!/usr/bin/env python3
"""
Standard Autoregressive Inference Script for Commit Message Generation
Model: Qwen/Qwen2.5-1.5B-Instruct

Uses the official HuggingFace model.generate() API.

Usage:
    python run_qwen_inference.py --input data.jsonl --output results/
    python run_qwen_inference.py --input data.jsonl --output results/ --sample
    python run_qwen_inference.py --input data.jsonl --output results/ \
        --max-new-tokens 512 --temperature 0.2 --top-p 0.95
"""

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(device: str):
    """
    Load tokenizer + model pinned to a specific device.

    device_map={"": device} forces every tensor onto exactly that device,
    giving fully independent model instances when called with cuda:0
    and cuda:1 from separate threads.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"[{device}] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log.info(f"[{device}] Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map={"": device},   # pin ALL tensors to this one device
    )
    model.eval()
    log.info(f"[{device}] Model ready.")
    return tokenizer, model


# ─── Stats helpers ────────────────────────────────────────────────────────────

def build_stats(
    prompt_tokens: int,
    generated_tokens: int,
    wall_seconds: float,
    prefill_seconds: float,
) -> dict:
    """
    Benchmarking metrics for LLM generation speed.

    Key metrics (correlatable with dLLM):
        tokens_per_second       – end-to-end throughput (comparable to dLLM batch_tokens_per_second)
        ms_per_token            – inverse throughput (comparable to dLLM effective_ms_per_token)
        prefill_seconds         – prompt processing time (≈ time-to-first-token)
        decode_seconds          – pure token generation time
        prefill_tok_per_second  – prompt processing throughput
        decode_tok_per_second   – decode-phase throughput
    """
    eps = 1e-9  # avoid division by zero
    decode_seconds = max(wall_seconds - prefill_seconds, eps)
    return {
        "prompt_tokens":            prompt_tokens,
        "generated_tokens":         generated_tokens,
        "wall_seconds":             round(wall_seconds, 4),
        "prefill_seconds":          round(prefill_seconds, 4),
        "decode_seconds":           round(decode_seconds, 4),
        "time_to_first_token_ms":   round(prefill_seconds * 1000, 3),
        "tokens_per_second":        round(generated_tokens / (wall_seconds + eps), 2),
        "ms_per_token":             round(wall_seconds / (generated_tokens + eps) * 1000, 3),
        "prefill_tok_per_second":   round(prompt_tokens / (prefill_seconds + eps), 2),
        "decode_tok_per_second":    round(generated_tokens / (decode_seconds + eps), 2),
        # Autoregressive always commits exactly 1 token per decoding step
        "tokens_per_step":          1.0,
    }


# ─── Single-sample inference ──────────────────────────────────────────────────

def run_single(
    task: dict,
    model,
    tokenizer,
    device: str,
    gen_kwargs: dict,
    output_dir: Path,
) -> dict:
    """Run generate() for one task, collect stats, and persist the result."""
    task_id  = task.get("task_id", "unknown")
    out_path = output_dir / f"{task_id}.json"

    # Resume: skip tasks that already completed successfully
    if out_path.exists():
        log.info(f"[{task_id}] Already done — skipping.")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    messages = task.get("messages",[])
    if not messages:
        raise ValueError(f"Task '{task_id}' has no 'messages' field.")

    # Build prompt via chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs     = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # ── Prefill measurement (isolated forward pass for TTFT) ────────────
    if device != "cpu":
        torch.cuda.synchronize(device)

    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    if device != "cpu":
        torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - t_prefill_start

    # ── Generate (full end-to-end) ───────────────────────────────────────
    if device != "cpu":
        torch.cuda.synchronize(device)

    t_start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )

    if device != "cpu":
        torch.cuda.synchronize(device)

    wall_seconds = time.perf_counter() - t_start

    # ── Decode & measure ─────────────────────────────────────────────────
    gen_ids          = outputs[0][prompt_len:]
    generated_text   = tokenizer.decode(gen_ids, skip_special_tokens=True)
    generated_tokens = len(gen_ids)

    stats = build_stats(
        prompt_tokens    = prompt_len,
        generated_tokens = generated_tokens,
        wall_seconds     = wall_seconds,
        prefill_seconds  = prefill_seconds,
    )

    result = {
        "task_id":    task_id,
        "generated":  generated_text,
        "label":      task.get("label"),
        "device":     device,
        "gen_kwargs": gen_kwargs,
        "stats":      stats,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log.info(
        f"[{task_id}] {wall_seconds:.2f}s | "
        f"{generated_tokens} tok | "
        f"{stats['tokens_per_second']:.1f} tok/s | "
        f"TTFT {stats['time_to_first_token_ms']:.0f}ms | "
        f"decode {stats['decode_tok_per_second']:.1f} tok/s"
    )
    return result


# ─── Worker (one per GPU) ─────────────────────────────────────────────────────

def gpu_worker(
    tasks: list,
    device: str,
    gen_kwargs: dict,
    output_dir: Path,
    results_collector: list,
    errors_collector: list,
):
    """Load model once on `device`, then process all assigned tasks sequentially."""
    try:
        tokenizer, model = load_model(device)
    except Exception as e:
        log.error(f"[{device}] Failed to load model: {e}\n{traceback.format_exc()}")
        errors_collector.append({"device": device, "task_id": None, "error": str(e)})
        return

    for task in tasks:
        task_id = task.get("task_id", "unknown")
        try:
            result = run_single(task, model, tokenizer, device, gen_kwargs, output_dir)
            results_collector.append(result)
        except Exception as e:
            log.error(f"[{task_id}] Error on {device}: {e}\n{traceback.format_exc()}")
            err = {
                "task_id":   task_id,
                "device":    device,
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }
            errors_collector.append(err)
            err_path = output_dir / f"{task_id}.error.json"
            try:
                with open(err_path, "w") as f:
                    json.dump(err, f, indent=2)
            except Exception:
                pass


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Standard inference for Qwen2.5 commit message generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input",  "-i", required=True,
                   help="Input JSONL file (one task per line).")
    p.add_argument("--output", "-o", default="./results",
                   help="Output directory for per-task JSON files.")
    p.add_argument("--sample", action="store_true",
                   help="Smoke-test mode: run only the first 5 tasks.")

    # ── Generation hyperparameters ──────────────────────────────────────
    p.add_argument("--max-new-tokens",   type=int,   default=1024,
                   help="Maximum tokens to generate.")
    p.add_argument("--temperature",      type=float, default=0.0,
                   help="Sampling temperature (0 = greedy decoding).")
    p.add_argument("--top-p",            type=float, default=0.95,
                   help="Nucleus sampling p. (Ignored if temperature is 0).")

    return p.parse_args()


# ─── Utilities ────────────────────────────────────────────────────────────────

def detect_devices() -> list[str]:
    n = torch.cuda.device_count()
    if n == 0:
        log.warning("No CUDA GPUs detected — falling back to CPU (expect slow inference).")
        return ["cpu"]
    
    # Optional: adjust if you want to use all available GPUs rather than capping at 2
    # The original script clamped it min(n, 2), adjusting to pure 'n' here
    devices = [f"cuda:{i}" for i in range(n)]
    log.info(f"Detected {n} GPU(s). Using: {devices}")
    return devices


def load_tasks(path: str) -> list[dict]:
    tasks =[]
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


def log_aggregate_stats(results: list[dict]):
    all_stats = [r["stats"] for r in results if "stats" in r]
    if not all_stats:
        return
    n = len(all_stats)
    avg = lambda key: sum(s.get(key, 0) for s in all_stats) / n

    # Compute aggregate throughput (total tokens / total wall time)
    total_tokens = sum(s["generated_tokens"] for s in all_stats)
    total_wall = sum(s["wall_seconds"] for s in all_stats)
    agg_tps = total_tokens / max(total_wall, 1e-9)

    log.info("── Aggregate benchmark (" + str(n) + " tasks) " + "─" * 30)
    log.info(f"  total tokens           : {total_tokens}")
    log.info(f"  total wall_seconds     : {total_wall:.3f} s")
    log.info(f"  aggregate tok/s        : {agg_tps:.2f}")
    log.info(f"  avg wall_seconds       : {avg('wall_seconds'):.3f} s")
    log.info(f"  avg tokens_per_second  : {avg('tokens_per_second'):.2f}")
    log.info(f"  avg ms_per_token       : {avg('ms_per_token'):.3f} ms")
    log.info(f"  avg prefill_seconds    : {avg('prefill_seconds'):.3f} s")
    log.info(f"  avg TTFT               : {avg('time_to_first_token_ms'):.3f} ms")
    log.info(f"  avg decode_tok/s       : {avg('decode_tok_per_second'):.2f}")
    log.info(f"  avg prefill_tok/s      : {avg('prefill_tok_per_second'):.2f}")
    log.info(f"  avg generated_tokens   : {avg('generated_tokens'):.1f}")
    log.info(f"  avg prompt_tokens      : {avg('prompt_tokens'):.1f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Determine sampling behavior dynamically
    do_sample = args.temperature > 0.0

    gen_kwargs = dict(
        max_new_tokens = args.max_new_tokens,
        do_sample      = do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"]       = args.top_p

    log.info("=" * 60)
    log.info("Standard Inference — Commit Message Generation")
    log.info(f"  model  : {MODEL_NAME}")
    log.info(f"  input  : {args.input}")
    log.info(f"  output : {args.output}")
    log.info(f"  kwargs : {gen_kwargs}")
    log.info("=" * 60)

    tasks = load_tasks(args.input)
    if not tasks:
        log.error("No valid tasks found. Exiting.")
        sys.exit(1)

    if args.sample:
        tasks = tasks[:5]
        log.info(f"--sample: using first {len(tasks)} task(s).")
    else:
        log.info(f"Loaded {len(tasks)} task(s).")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    devices = detect_devices()
    chunks  = split_tasks(tasks, len(devices))

    log.info("Task distribution:")
    for dev, chunk in zip(devices, chunks):
        log.info(f"  {dev}: {len(chunk)} task(s)")

    results: list[dict] = []
    errors:  list[dict] =[]

    if len(devices) == 1:
        gpu_worker(chunks[0], devices[0], gen_kwargs, output_dir, results, errors)
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures =[
                pool.submit(
                    gpu_worker,
                    chunk, device, gen_kwargs, output_dir, results, errors,
                )
                for chunk, device in zip(chunks, devices)
            ]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    log.error(f"Worker thread raised: {exc}")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Done.  Successes: {len(results)}   Errors: {len(errors)}")
    log_aggregate_stats(results)

    if errors:
        log.warning(f"{len(errors)} task(s) failed:")
        for e in errors:
            log.warning(f"  task_id={e.get('task_id')}  device={e.get('device')}  {e.get('error')}")
        err_summary = output_dir / "_errors_summary.json"
        with open(err_summary, "w") as f:
            json.dump(errors, f, indent=2)
        log.info(f"Error details → {err_summary}")

    # Rebuild summary from all individual JSON files on disk so that
    # resumed runs and partial completions are fully captured.
    all_results = []
    for jf in sorted(output_dir.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            with open(jf, encoding="utf-8") as f:
                all_results.append(json.load(f))
        except Exception as e:
            log.warning(f"Skipping {jf}: {e}")
    if not all_results:
        all_results = results

    summary_path = output_dir / "_results_summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(all_results)} results → {summary_path}")


if __name__ == "__main__":
    main()