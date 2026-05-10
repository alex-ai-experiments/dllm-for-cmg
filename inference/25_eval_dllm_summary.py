#!/usr/bin/env python3
"""
Two-Step CMG Pipeline: dLLM File Summarisation → LLM Commit Message Generation
================================================================================

Step 1 (dLLM): Each changed file's diff is summarised by a dLLM in parallel
               batches of up to --summary-batch-size (default 4). This exploits
               the dLLM's ability to generate for multiple prompts simultaneously.

Step 2  (LLM): The per-file summaries are concatenated and fed to a standard
               autoregressive LLM (Qwen2.5-1.5B-Instruct) to produce the final
               commit message.

This is "Condition D" in the two-step CMG experiment. Conditions A (AR-Direct)
and B (dLLM-Direct) use existing results; Condition C uses 26_eval_llm_summary.py.

Ablations supported (grid search):
  --thresholds          dLLM confidence thresholds for Step 1 (default: 0.2,0.4,0.8)
  --block-sizes         dLLM block sizes for Step 1 (default: 16,32)
  --small-block-sizes   dLLM small block sizes for Step 1 (default: 8)
  --summary-max-tokens  Max new tokens per file summary (default: 128)
  --cmg-max-tokens      Max new tokens for the final CMG (default: 128)
  --summary-batch-size  Files processed in parallel by dLLM (default: 4)

Usage:
    python 25_eval_dllm_summary.py -i build_tasks/tasks_tags.jsonl \\
        -o results_two_step_dllm/
    python 25_eval_dllm_summary.py -i build_tasks/tasks_tags.jsonl \\
        -o results_two_step_dllm/ --thresholds 0.2,0.4,0.8 --block-sizes 16,32
    python 25_eval_dllm_summary.py -i build_tasks/tasks_tags.jsonl \\
        -o results_two_step_dllm/ --sample 50   # quick 50-task dry run
"""

import argparse
import itertools
import json
import logging
import sys
import time
import traceback
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
from diff_utils import get_per_file_diffs, build_summary_messages, build_cmg_messages

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DLLM_MODEL_NAME = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MASK_ID = FAST_DLLM_MASK_ID
STOP_TOKEN = FAST_DLLM_STOP_TOKEN

_MODEL_LOAD_LOCK = __import__("threading").Lock()


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_dllm(device: str, compile_model: bool = False):
    """Load the dLLM (Fast_dLLM_v2_1.5B) and bind batch_sample."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"[dLLM/{device}] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(DLLM_MODEL_NAME, trust_remote_code=True)

    log.info(f"[dLLM/{device}] Loading model ...")
    with _MODEL_LOAD_LOCK:
        model = AutoModelForCausalLM.from_pretrained(
            DLLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
    model = model.to(device)
    model.eval()
    model.mdm_sample = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)
    if compile_model:
        log.info(f"[dLLM/{device}] Compiling with torch.compile (reduce-overhead) ...")
        model = torch.compile(model, mode="reduce-overhead")
        log.info(f"[dLLM/{device}] Compilation registered (triggers on first forward pass).")
    log.info(f"[dLLM/{device}] Ready.")
    return tokenizer, model


def load_llm(device: str):
    """Load the autoregressive LLM (Qwen2.5-1.5B-Instruct) for the CMG step."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"[LLM/{device}] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    log.info(f"[LLM/{device}] Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype="auto",
        device_map={"": device},
    )
    model.eval()
    log.info(f"[LLM/{device}] Ready.")
    return tokenizer, model


# ─── dLLM batch inference ─────────────────────────────────────────────────────

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

    Returns:
        generated_texts: list of decoded output strings
        generated_token_counts: tokens generated per item
        wall_seconds: elapsed time for the batch
        total_tokens: total tokens generated across batch
        total_steps: total diffusion steps
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
            ids = torch.cat([ids, torch.full((pad_len,), MASK_ID, dtype=torch.long)])
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
        mask_id=MASK_ID,
        min_len=min_len,
        seq_len=seq_len_tensor,
        use_block_cache=gen_kwargs.get("use_block_cache", True),
        threshold=gen_kwargs["threshold"],
        temperature=gen_kwargs.get("temperature", 0.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        stop_token=STOP_TOKEN,
    )

    if device != "cpu":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t0

    texts = []
    counts = []
    total_tokens = 0
    for i, sl in enumerate(seq_lens):
        gen_part = generated_ids[i][sl:]
        n_tok = int((gen_part != MASK_ID).sum().item())
        total_tokens += n_tok
        counts.append(n_tok)
        texts.append(dllm_tokenizer.decode(gen_part, skip_special_tokens=True))

    return texts, counts, wall_seconds, total_tokens, total_steps


# ─── LLM single inference ─────────────────────────────────────────────────────

def run_llm_single(
    messages: list[dict],
    llm_model,
    llm_tokenizer,
    device: str,
    gen_kwargs: dict,
    max_new_tokens: int,
) -> tuple[str, int, float]:
    """
    Run a single LLM generate call for the CMG step.

    Returns:
        generated_text, generated_token_count, wall_seconds
    """
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if device != "cpu":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=gen_kwargs.get("do_sample", False),
            temperature=gen_kwargs.get("temperature", 1.0),
            top_p=gen_kwargs.get("top_p", 0.95),
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    if device != "cpu":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t0

    gen_ids = outputs[0][prompt_len:]
    generated_text = llm_tokenizer.decode(gen_ids, skip_special_tokens=True)
    return generated_text, len(gen_ids), wall_seconds


# ─── Two-step pipeline for one task ──────────────────────────────────────────

def process_task_two_step(
    task: dict,
    dllm_model,
    dllm_tokenizer,
    llm_model,
    llm_tokenizer,
    dllm_device: str,
    llm_device: str,
    dllm_gen_kwargs: dict,
    llm_gen_kwargs: dict,
    summary_max_tokens: int,
    cmg_max_tokens: int,
    summary_batch_size: int,
    max_diff_chars: int | None = None,
) -> dict:
    """
    Run the full two-step pipeline for a single task.

    Returns a result dict compatible with the existing analysis infrastructure.
    """
    task_id = task.get("task_id", "unknown")
    label = task.get("label", "")

    # ── Step 0: Parse per-file diffs ─────────────────────────────────────
    file_diffs = get_per_file_diffs(task)
    num_files = len(file_diffs)

    # ── Step 1: dLLM file summarisation ──────────────────────────────────
    # Build messages with max_diff_chars truncation applied first, then sort by
    # actual post-truncation prompt token length (ascending).  Sorting by raw
    # diff length is inaccurate when a char-cap is active: a 2000-char diff
    # truncated to 600 chars sorts as 2000, inflating estimated length and
    # misplacing it in the batch.  Using the true prompt length gives the tightest
    # possible length grouping, minimising padding waste and the dying-batch effect.
    all_summary_messages = [
        build_summary_messages(filename, file_diff, max_diff_chars=max_diff_chars)
        for filename, file_diff in file_diffs
    ]
    prompt_tok_lens = [
        len(dllm_tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        ))
        for msgs in all_summary_messages
    ]
    sort_order = sorted(range(len(file_diffs)), key=lambda i: prompt_tok_lens[i])
    file_diffs_sorted = [file_diffs[i] for i in sort_order]
    all_summary_messages = [all_summary_messages[i] for i in sort_order]

    file_summaries: list[tuple[str, str]] = []  # (filename, summary_text)
    summary_token_counts: list[int] = []

    t_summary_total = 0.0
    summary_total_tokens = 0
    summary_total_steps = 0

    for batch_start in range(0, len(all_summary_messages), summary_batch_size):
        batch_msgs = all_summary_messages[batch_start : batch_start + summary_batch_size]
        batch_filenames = [fd[0] for fd in file_diffs_sorted[batch_start : batch_start + summary_batch_size]]

        texts, counts, wall_s, total_tok, total_steps = run_dllm_batch(
            batch_msgs,
            dllm_model,
            dllm_tokenizer,
            dllm_device,
            dllm_gen_kwargs,
            summary_max_tokens,
        )

        t_summary_total += wall_s
        summary_total_tokens += total_tok
        summary_total_steps += total_steps

        for filename, summary_text, n_tok in zip(batch_filenames, texts, counts):
            file_summaries.append((filename, summary_text))
            summary_token_counts.append(n_tok)

    # ── Step 2: LLM commit message generation ────────────────────────────
    cmg_messages = build_cmg_messages(file_summaries)

    cmg_text, cmg_tokens, t_cmg = run_llm_single(
        cmg_messages,
        llm_model,
        llm_tokenizer,
        llm_device,
        llm_gen_kwargs,
        cmg_max_tokens,
    )

    # ── Aggregate stats ───────────────────────────────────────────────────
    t_total = t_summary_total + t_cmg
    eps = 1e-9

    # Overall pipeline throughput (measured on CMG output — what we evaluate quality on)
    overall_tps = cmg_tokens / (t_total + eps)
    # dLLM summary throughput
    summary_tps = summary_total_tokens / (t_summary_total + eps)
    summary_tpstep = summary_total_tokens / max(summary_total_steps, 1)

    result = {
        "task_id": task_id,
        "generated": cmg_text,
        "label": label,
        "device": {"dllm": dllm_device, "llm": llm_device},
        "pipeline": "dllm_summary_llm_cmg",
        "gen_kwargs": {
            "summary_step": dllm_gen_kwargs,
            "cmg_step": llm_gen_kwargs,
            "summary_max_tokens": summary_max_tokens,
            "cmg_max_tokens": cmg_max_tokens,
            "summary_batch_size": summary_batch_size,
        },
        "summaries": [
            {"filename": fn, "summary": s, "tokens": t}
            for (fn, s), t in zip(file_summaries, summary_token_counts)
        ],
        "stats": {
            # ── Fields used by existing analysis (compute_run_metrics) ──
            "generated_tokens": cmg_tokens,
            "batch_tokens_per_second": round(overall_tps, 2),
            "effective_ms_per_token": round(t_total / (cmg_tokens + eps) * 1000, 3),
            "tokens_per_step": round(summary_tpstep, 3),
            "batch_total_steps": summary_total_steps,
            # ── Pipeline-specific breakdown ─────────────────────────────
            "num_files": num_files,
            "t_summary_seconds": round(t_summary_total, 4),
            "t_cmg_seconds": round(t_cmg, 4),
            "t_total_seconds": round(t_total, 4),
            "summary_tokens_per_second": round(summary_tps, 2),
            "summary_total_tokens": summary_total_tokens,
            "summary_total_steps": summary_total_steps,
        },
    }
    return result


# ─── Config naming ────────────────────────────────────────────────────────────

def make_config_name(block_size, small_block_size, threshold, summary_max_tokens, cmg_max_tokens):
    th_str = f"{threshold:.1f}".rstrip("0").rstrip(".")
    return (
        f"bs{block_size}_sbs{small_block_size}_th{th_str}"
        f"_smnt{summary_max_tokens}_cmnt{cmg_max_tokens}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-Step CMG: dLLM summary → LLM CMG")
    p.add_argument("-i", "--input", required=True, help="Input JSONL tasks file")
    p.add_argument("-o", "--output", required=True, help="Output root directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample", type=int, default=None,
                   help="Only process this many tasks (for quick dry runs)")

    # dLLM ablation grid
    p.add_argument("--thresholds", default="0.8",
                   help="Comma-separated dLLM thresholds for Step 1 (default: 0.8). "
                        "Avoid 1.0 — it degrades to pure AR (tokens/step=1).")
    p.add_argument("--block-sizes", default="32",
                   help="Comma-separated dLLM block sizes for Step 1 (default: 32)")
    p.add_argument("--small-block-sizes", default="8",
                   help="Comma-separated dLLM small block sizes for Step 1 (default: 8)")

    # Token budgets
    p.add_argument("--summary-max-tokens", default="1024,128",
                   help="Comma-separated max new tokens per file summary — Step 1 (default: 1024,128)")
    p.add_argument("--cmg-max-tokens", type=int, default=128,
                   help="Max new tokens for final CMG (Step 2)")
    p.add_argument("--summary-batch-size", type=int, default=4,
                   help="Number of files processed in parallel by dLLM")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the dLLM with torch.compile (reduce-overhead mode). "
                        "Speeds up per-step forward passes after the first batch.")
    p.add_argument("--max-diff-chars", type=int, default=None,
                   help="Truncate each file diff to this many characters before building "
                        "the summary prompt. Reduces padding waste in dLLM batches. "
                        "None = no truncation (default).")
    return p.parse_args()


def main():
    args = parse_args()

    # Parse ablation grid
    thresholds = [float(x) for x in args.thresholds.split(",")]
    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    small_block_sizes = [int(x) for x in args.small_block_sizes.split(",")]
    summary_max_tokens_list = [int(x) for x in args.summary_max_tokens.split(",")]

    # Load tasks
    tasks = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if args.sample:
        tasks = tasks[: args.sample]
    log.info(f"Loaded {len(tasks)} tasks from {args.input}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Two-GPU detection: place dLLM and LLM on separate GPUs when available
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus >= 2:
        dllm_device = "cuda:0"
        llm_device = "cuda:1"
        log.info("Two GPUs detected — dLLM → cuda:0, LLM → cuda:1")
    else:
        dllm_device = args.device
        llm_device = args.device
        log.info(f"Single device — both models on {args.device}")

    # Load models once (shared across all ablation configs)
    dllm_tokenizer, dllm_model = load_dllm(dllm_device, compile_model=args.compile)
    llm_tokenizer, llm_model = load_llm(llm_device)

    # LLM CMG gen kwargs (fixed — greedy decoding)
    llm_gen_kwargs = {"do_sample": False}

    # Grid search over dLLM summary configs
    grid = list(itertools.product(block_sizes, small_block_sizes, thresholds, summary_max_tokens_list))
    log.info(f"Running {len(grid)} dLLM config(s) × {len(tasks)} tasks each.")

    for block_size, small_block_size, threshold, summary_max_tokens in grid:
        config_name = make_config_name(
            block_size, small_block_size, threshold,
            summary_max_tokens, args.cmg_max_tokens,
        )
        config_dir = output_root / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        dllm_gen_kwargs = {
            "block_size": block_size,
            "small_block_size": small_block_size,
            "threshold": threshold,
            "use_block_cache": True,
            "temperature": 0.0,
            "top_p": 0.95,
        }

        log.info(f"=== Config: {config_name} ===")
        n_done = 0
        n_err = 0

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            out_path = config_dir / f"{task_id}.json"

            # Resume support
            if out_path.exists():
                try:
                    with open(out_path, encoding="utf-8") as f:
                        json.load(f)  # verify it's valid JSON
                    log.info(f"[{task_id}] Already done — skipping.")
                    n_done += 1
                    continue
                except Exception:
                    pass  # re-process if corrupted

            try:
                result = process_task_two_step(
                    task=task,
                    dllm_model=dllm_model,
                    dllm_tokenizer=dllm_tokenizer,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    dllm_device=dllm_device,
                    llm_device=llm_device,
                    dllm_gen_kwargs=dllm_gen_kwargs,
                    llm_gen_kwargs=llm_gen_kwargs,
                    summary_max_tokens=summary_max_tokens,
                    cmg_max_tokens=args.cmg_max_tokens,
                    summary_batch_size=args.summary_batch_size,
                    max_diff_chars=args.max_diff_chars,
                )

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                stats = result["stats"]
                log.info(
                    f"[{task_id}] files={stats['num_files']} "
                    f"t_sum={stats['t_summary_seconds']:.2f}s "
                    f"t_cmg={stats['t_cmg_seconds']:.2f}s "
                    f"cmg_tok={stats['generated_tokens']} "
                    f"tps={stats['batch_tokens_per_second']:.1f}"
                )
                n_done += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.error(f"[{task_id}] OOM — skipping.")
                n_err += 1
            except Exception as e:
                log.error(f"[{task_id}] Error: {e}\n{traceback.format_exc()}")
                n_err += 1

        log.info(f"Config {config_name}: {n_done} done, {n_err} errors.")

    log.info("All configurations complete.")


if __name__ == "__main__":
    main()
