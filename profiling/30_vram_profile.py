#!/usr/bin/env python3
"""
VRAM Profiling Script for Fast-dLLM v2 Batched Inference

Loads the model, then tests increasing batch sizes to find the maximum
that fits in GPU memory and reports VRAM usage at each level.

Usage:
    python 30_vram_profile.py -i build_tasks/tasks_tags.jsonl
    python 30_vram_profile.py -i build_tasks/tasks_tags.jsonl --batch-sizes 1,2,4,8,16,32
    python 30_vram_profile.py -i build_tasks/tasks_tags.jsonl --device cuda:1
"""

import argparse
import json
import logging
import sys
import time
import types
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MODEL_NAME = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
MASK_ID = FAST_DLLM_MASK_ID
STOP_TOKEN = FAST_DLLM_STOP_TOKEN


def fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    return f"{b / (1 << 20):.2f} MiB"


def load_model(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    log.info(f"Loading model to {device} …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    model.mdm_sample = types.MethodType(
        Fast_dLLM_QwenForCausalLM.batch_sample, model
    )

    return tokenizer, model


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return tasks


def tokenize_task(task, tokenizer):
    text = tokenizer.apply_chat_template(
        task["messages"], tokenize=False, add_generation_prompt=True
    )
    return tokenizer([text], return_tensors="pt")["input_ids"][0]


def profile_batch(
    batch_tasks: list[dict],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    block_size: int,
    small_block_size: int,
    threshold: float,
    use_block_cache: bool,
) -> dict:
    """Run one batch and return full stats (mirrors 10_new_eval metrics)."""
    batch_size = len(batch_tasks)

    input_ids_list = []
    seq_lens = []
    for task in batch_tasks:
        ids = tokenize_task(task, tokenizer)
        input_ids_list.append(ids)
        seq_lens.append(len(ids))

    max_len = max(seq_lens)
    min_len = min(seq_lens)

    padded = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), MASK_ID, dtype=torch.long)])
        padded.append(ids.unsqueeze(0))

    batched_ids = torch.cat(padded, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    t_start = time.perf_counter()

    generated_ids, total_steps = model.mdm_sample(
        batched_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        mask_id=MASK_ID,
        min_len=min_len,
        seq_len=seq_len_tensor,
        use_block_cache=use_block_cache,
        threshold=threshold,
        temperature=0.0,
        top_p=0.95,
        stop_token=STOP_TOKEN,
    )

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t_start

    mem_after = torch.cuda.memory_allocated(device)
    mem_peak = torch.cuda.max_memory_allocated(device)

    total_tokens = 0
    for i in range(batch_size):
        gen_part = generated_ids[i][seq_lens[i]:]
        total_tokens += int((gen_part != MASK_ID).sum().item())

    eps = 1e-9
    return {
        "batch_size": batch_size,
        "wall_seconds": round(wall_seconds, 3),
        "total_generated_tokens": total_tokens,
        "total_steps": total_steps,
        "tokens_per_second": round(total_tokens / max(wall_seconds, eps), 2),
        "ms_per_token": round(wall_seconds / max(total_tokens, 1) * 1000, 3),
        "tokens_per_step": round(total_tokens / max(total_steps, 1), 3),
        "mem_before": mem_before,
        "mem_after": mem_after,
        "mem_peak": mem_peak,
        "mem_delta": mem_peak - mem_before,
        "max_prompt_len": max_len,
        "min_prompt_len": min_len,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="VRAM profiling for Fast-dLLM v2 batched inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="Input JSONL file (tasks).")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device to profile on.")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes to test.")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens to generate per task.")
    p.add_argument("--block-size", type=int, default=32,
                   help="Block size for MDM generation.")
    p.add_argument("--small-block-size", type=int, default=8,
                   help="Small block size.")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Confidence threshold.")
    p.add_argument("--use-block-cache", action="store_true",
                   help="Enable block KV-cache.")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if not torch.cuda.is_available():
        log.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    tasks = load_tasks(args.input)
    if not tasks:
        log.error("No tasks found.")
        sys.exit(1)

    tasks.sort(key=lambda t: t.get("context_length", 0))
    log.info(f"Loaded {len(tasks)} task(s), sorted by context_length (ascending).")
    log.info(f"Device: {device}")
    log.info(f"Batch sizes to test: {batch_sizes}")

    # GPU info
    gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
    total_vram = torch.cuda.get_device_properties(gpu_idx).total_memory
    log.info(f"GPU: {torch.cuda.get_device_name(gpu_idx)}")
    log.info(f"Total VRAM: {fmt_bytes(total_vram)}")

    # Load model
    torch.cuda.reset_peak_memory_stats(device)
    mem_before_model = torch.cuda.memory_allocated(device)
    tokenizer, model = load_model(device)
    torch.cuda.synchronize(device)
    mem_after_model = torch.cuda.memory_allocated(device)
    model_mem = mem_after_model - mem_before_model

    log.info(f"Model VRAM: {fmt_bytes(model_mem)}")
    log.info(f"Free VRAM (approx): {fmt_bytes(total_vram - mem_after_model)}")

    # How many reps per batch size: n_reps(bs) = max_batch // bs
    # so the total number of samples processed is equalised across all batch sizes.
    # e.g. batch_sizes=[8,16,32] → 4 reps of 8, 2 of 16, 1 of 32.
    max_batch = max(batch_sizes)

    # Profile each batch size
    W = 104  # table width
    log.info("=" * W)
    log.info(
        f"{'Batch':>6} | {'Rep':>5} | {'Wall(s)':>8} | {'Tokens':>7} | "
        f"{'Steps':>7} | {'Tok/s':>8} | {'Tok/step':>9} | "
        f"{'Peak VRAM':>12} | {'Delta VRAM':>12} | Status"
    )
    log.info("-" * W)

    profile_results = []
    oom_break = False

    for bs in sorted(batch_sizes):
        if oom_break:
            profile_results.append({"batch_size": bs, "status": "SKIPPED (OOM at smaller batch)"})
            continue

        if bs > len(tasks):
            log.info(
                f"{bs:>6} | {'—':>5} | {'—':>8} | {'—':>7} | {'—':>7} | "
                f"{'—':>8} | {'—':>9} | {'—':>12} | {'—':>12} | SKIP (only {len(tasks)} tasks)"
            )
            continue

        n_reps = max(max_batch // bs, 1)
        rep_stats: list[dict] = []
        oom_hit = False

        for r in range(n_reps):
            # Each rep uses a different length-sorted slice (cycling if tasks < n_reps*bs)
            start = (r * bs) % len(tasks)
            batch_tasks = [tasks[(start + i) % len(tasks)] for i in range(bs)]

            torch.cuda.empty_cache()
            try:
                s = profile_batch(
                    batch_tasks, tokenizer, model, device,
                    args.max_new_tokens, args.block_size,
                    args.small_block_size, args.threshold,
                    args.use_block_cache,
                )
                rep_stats.append(s)
                log.info(
                    f"{bs:>6} | {r + 1:>5} | "
                    f"{s['wall_seconds']:>8.3f} | "
                    f"{s['total_generated_tokens']:>7} | "
                    f"{s['total_steps']:>7} | "
                    f"{s['tokens_per_second']:>8.1f} | "
                    f"{s['tokens_per_step']:>9.3f} | "
                    f"{fmt_bytes(s['mem_peak']):>12} | "
                    f"{fmt_bytes(s['mem_delta']):>12} | OK"
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.info(
                    f"{bs:>6} | {r + 1:>5} | {'—':>8} | {'—':>7} | {'—':>7} | "
                    f"{'—':>8} | {'—':>9} | {'—':>12} | {'—':>12} | OOM"
                )
                oom_hit = True
                break
            except Exception as e:
                log.error(f"Batch size {bs} rep {r + 1} failed: {e}")
                oom_hit = True
                break

        if oom_hit and not rep_stats:
            profile_results.append({"batch_size": bs, "n_reps": 0, "status": "OOM"})
            oom_break = True
            continue

        # ── Aggregate across completed reps ─────────────────────────────
        n_ok = len(rep_stats)
        total_wall = sum(s["wall_seconds"] for s in rep_stats)
        total_tokens_all = sum(s["total_generated_tokens"] for s in rep_stats)
        total_steps_all = sum(s["total_steps"] for s in rep_stats)
        eps = 1e-9
        agg_tps = total_tokens_all / max(total_wall, eps)
        agg_mspt = total_wall / max(total_tokens_all, 1) * 1000
        agg_toks_per_step = total_tokens_all / max(total_steps_all, 1)
        mem_peak_max = max(s["mem_peak"] for s in rep_stats)
        mean_wall = total_wall / max(n_ok, 1)
        mean_tokens = total_tokens_all // max(n_ok, 1)
        mean_steps = total_steps_all // max(n_ok, 1)

        log.info("-" * W)
        log.info(
            f"{bs:>6} | {'AGG':>5} | "
            f"{mean_wall:>8.3f} | "
            f"{mean_tokens:>7} | "
            f"{mean_steps:>7} | "
            f"{agg_tps:>8.1f} | "
            f"{agg_toks_per_step:>9.3f} | "
            f"{fmt_bytes(mem_peak_max):>12} | "
            f"{'(mean/rep)':>12} | {n_ok}/{n_reps} reps"
        )
        log.info("=" * W)

        profile_results.append({
            "batch_size": bs,
            "n_reps": n_ok,
            "status": "OK" if not oom_hit else "PARTIAL",
            "per_rep": [
                {
                    "rep": i + 1,
                    "wall_seconds": s["wall_seconds"],
                    "total_generated_tokens": s["total_generated_tokens"],
                    "total_steps": s["total_steps"],
                    "tokens_per_second": s["tokens_per_second"],
                    "ms_per_token": s["ms_per_token"],
                    "tokens_per_step": s["tokens_per_step"],
                    "mem_peak_bytes": s["mem_peak"],
                    "mem_peak_human": fmt_bytes(s["mem_peak"]),
                    "mem_delta_bytes": s["mem_delta"],
                    "mem_delta_human": fmt_bytes(s["mem_delta"]),
                    "max_prompt_len": s["max_prompt_len"],
                    "min_prompt_len": s["min_prompt_len"],
                }
                for i, s in enumerate(rep_stats)
            ],
            "aggregate": {
                "n_complete_reps": n_ok,
                "mean_wall_seconds": round(mean_wall, 3),
                "total_wall_seconds": round(total_wall, 3),
                "total_generated_tokens": total_tokens_all,
                "total_steps": total_steps_all,
                "tokens_per_second": round(agg_tps, 2),
                "ms_per_token": round(agg_mspt, 3),
                "tokens_per_step": round(agg_toks_per_step, 3),
                "mem_peak_max_bytes": mem_peak_max,
                "mem_peak_max_human": fmt_bytes(mem_peak_max),
            },
        })

    # Save results
    out_path = Path("outputs/vram_profile.json")
    report = {
        "model": MODEL_NAME,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(gpu_idx),
        "total_vram_bytes": total_vram,
        "total_vram_human": fmt_bytes(total_vram),
        "model_vram_bytes": model_mem,
        "model_vram_human": fmt_bytes(model_mem),
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "threshold": args.threshold,
        "use_block_cache": args.use_block_cache,
        "num_tasks_available": len(tasks),
        "results": profile_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Profile saved → {out_path}")


if __name__ == "__main__":
    main()
