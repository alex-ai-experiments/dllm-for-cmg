#!/usr/bin/env python3
"""
Two-Step CMG Pipeline: LLM File Summarisation → LLM Commit Message Generation
===============================================================================

Step 1 (LLM): Each changed file's diff is summarised by a standard autoregressive
              LLM (Qwen2.5-1.5B-Instruct), processed sequentially.

Step 2 (LLM): The per-file summaries are concatenated and fed to the same LLM
              to produce the final commit message.

This is "Condition C" in the two-step CMG experiment.

Condition A (AR-Direct) uses existing results in results_llm_baseline/.
Condition B (dLLM-Direct) uses existing results in fresh_ablation_results/.
Condition C (this script): LLM summary → LLM CMG
Condition D uses 25_eval_dllm_summary.py: dLLM summary → LLM CMG

Usage:
    python 26_eval_llm_summary.py -i build_tasks/tasks_tags.jsonl \\
        -o results_two_step_llm/
    python 26_eval_llm_summary.py -i build_tasks/tasks_tags.jsonl \\
        -o results_two_step_llm/ --sample 50   # quick 50-task dry run
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from diff_utils import get_per_file_diffs, build_summary_messages, build_cmg_messages

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_llm(device: str):
    """Load the autoregressive LLM (Qwen2.5-1.5B-Instruct)."""
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


# ─── LLM single inference ─────────────────────────────────────────────────────

def run_llm_single(
    messages: list[dict],
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    do_sample: bool = False,
) -> tuple[str, int, float]:
    """
    Run a single LLM generate call.

    Returns:
        generated_text, generated_token_count, wall_seconds
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if device != "cpu":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device != "cpu":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - t0

    gen_ids = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return generated_text, len(gen_ids), wall_seconds


# ─── Two-step pipeline for one task ──────────────────────────────────────────

def process_task_two_step(
    task: dict,
    summary_model,
    summary_tokenizer,
    summary_device: str,
    cmg_model,
    cmg_tokenizer,
    cmg_device: str,
    summary_max_tokens: int,
    cmg_max_tokens: int,
    max_diff_chars: int | None = None,
) -> dict:
    """
    Run the full two-step LLM→LLM pipeline for a single task.

    Returns a result dict compatible with the existing analysis infrastructure.
    """
    task_id = task.get("task_id", "unknown")
    label = task.get("label", "")

    # ── Step 0: Parse per-file diffs ─────────────────────────────────────
    file_diffs = get_per_file_diffs(task)
    num_files = len(file_diffs)

    # ── Step 1: LLM file summarisation (sequential) ───────────────────────
    file_summaries: list[tuple[str, str]] = []
    summary_token_counts: list[int] = []
    t_summary_total = 0.0
    summary_total_tokens = 0

    for filename, file_diff in file_diffs:
        messages = build_summary_messages(filename, file_diff, max_diff_chars=max_diff_chars)
        summary_text, n_tok, wall_s = run_llm_single(
            messages, summary_model, summary_tokenizer, summary_device, summary_max_tokens
        )
        file_summaries.append((filename, summary_text))
        summary_token_counts.append(n_tok)
        t_summary_total += wall_s
        summary_total_tokens += n_tok

    # ── Step 2: LLM commit message generation ────────────────────────────
    cmg_messages = build_cmg_messages(file_summaries)
    cmg_text, cmg_tokens, t_cmg = run_llm_single(
        cmg_messages, cmg_model, cmg_tokenizer, cmg_device, cmg_max_tokens
    )

    # ── Aggregate stats ───────────────────────────────────────────────────
    t_total = t_summary_total + t_cmg
    eps = 1e-9

    # Overall pipeline throughput measured on CMG output
    overall_tps = cmg_tokens / (t_total + eps)
    summary_tps = summary_total_tokens / (t_summary_total + eps)

    result = {
        "task_id": task_id,
        "generated": cmg_text,
        "label": label,
        "device": {"summary": summary_device, "cmg": cmg_device},
        "pipeline": "llm_summary_llm_cmg",
        "gen_kwargs": {
            "model": LLM_MODEL_NAME,
            "summary_max_tokens": summary_max_tokens,
            "cmg_max_tokens": cmg_max_tokens,
        },
        "summaries": [
            {"filename": fn, "summary": s, "tokens": t}
            for (fn, s), t in zip(file_summaries, summary_token_counts)
        ],
        "stats": {
            # ── Fields used by existing analysis (compute_run_metrics) ──
            "generated_tokens": cmg_tokens,
            # For AR, tokens_per_step = 1.0 always
            "tokens_per_second": round(overall_tps, 2),
            "ms_per_token": round(t_total / (cmg_tokens + eps) * 1000, 3),
            "tokens_per_step": 1.0,
            # Using batch_total_steps = total summary tokens (number of AR steps in step 1)
            "batch_total_steps": summary_total_tokens,
            # ── Pipeline-specific breakdown ─────────────────────────────
            "num_files": num_files,
            "t_summary_seconds": round(t_summary_total, 4),
            "t_cmg_seconds": round(t_cmg, 4),
            "t_total_seconds": round(t_total, 4),
            "summary_tokens_per_second": round(summary_tps, 2),
            "summary_total_tokens": summary_total_tokens,
        },
    }
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-Step CMG: LLM summary → LLM CMG (Condition C)")
    p.add_argument("-i", "--input", required=True, help="Input JSONL tasks file")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample", type=int, default=None,
                   help="Only process this many tasks (for quick dry runs)")
    p.add_argument("--summary-max-tokens", type=int, default=128,
                   help="Max new tokens per file summary (Step 1)")
    p.add_argument("--cmg-max-tokens", type=int, default=128,
                   help="Max new tokens for the final CMG (Step 2)")
    p.add_argument("--max-diff-chars", type=int, default=None,
                   help="Truncate each file diff to this many characters before building "
                        "the summary prompt. None = no truncation (default).")
    return p.parse_args()


def main():
    args = parse_args()

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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Two-GPU detection: load separate LLM instances on different GPUs when available.
    # In the single-GPU case both steps share the same model object.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus >= 2:
        summary_device = "cuda:0"
        cmg_device = "cuda:1"
        log.info("Two GPUs detected — summary LLM → cuda:0, CMG LLM → cuda:1")
        summary_tokenizer, summary_model = load_llm(summary_device)
        cmg_tokenizer, cmg_model = load_llm(cmg_device)
    else:
        summary_device = args.device
        cmg_device = args.device
        log.info(f"Single device — LLM on {args.device}")
        summary_tokenizer, summary_model = load_llm(summary_device)
        cmg_model = summary_model
        cmg_tokenizer = summary_tokenizer

    n_done = 0
    n_err = 0

    for task in tasks:
        task_id = task.get("task_id", "unknown")
        out_path = output_dir / f"{task_id}.json"

        # Resume support
        if out_path.exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    json.load(f)
                log.info(f"[{task_id}] Already done — skipping.")
                n_done += 1
                continue
            except Exception:
                pass

        try:
            result = process_task_two_step(
                task=task,
                summary_model=summary_model,
                summary_tokenizer=summary_tokenizer,
                summary_device=summary_device,
                cmg_model=cmg_model,
                cmg_tokenizer=cmg_tokenizer,
                cmg_device=cmg_device,
                summary_max_tokens=args.summary_max_tokens,
                cmg_max_tokens=args.cmg_max_tokens,
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
                f"tps={stats['tokens_per_second']:.1f}"
            )
            n_done += 1

        except Exception as e:
            log.error(f"[{task_id}] Error: {e}\n{traceback.format_exc()}")
            n_err += 1

    log.info(f"Done: {n_done} tasks processed, {n_err} errors.")


if __name__ == "__main__":
    main()
