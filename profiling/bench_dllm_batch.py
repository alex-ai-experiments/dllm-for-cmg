#!/usr/bin/env python3
"""
Batch-size / padding benchmark for the dLLM summary step.

Runs the same set of file-diff prompts through batch_sample with batch sizes
1, 2, and 4 and prints a detailed breakdown of:

  - Padding ratio (wasted MASK tokens vs useful prompt+gen tokens)
  - Wall time per batch and per sample
  - Total diffusion steps per batch
  - Tokens generated per step (efficiency metric)
  - GPU memory high-water mark

Usage:
    python bench_dllm_batch.py -i build_tasks/tasks_3to10files.jsonl --sample-task 0
    python bench_dllm_batch.py -i build_tasks/tasks_3to10files.jsonl --sample-task 5 --max-new-tokens 256
    python bench_dllm_batch.py -i build_tasks/tasks_tags.jsonl --task-id apache_hive_6517872
"""

import argparse
import json
import sys
import itertools
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
from diff_utils import get_per_file_diffs, build_summary_messages

DLLM_MODEL_NAME = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
MASK_ID = FAST_DLLM_MASK_ID
STOP_TOKEN = FAST_DLLM_STOP_TOKEN

SEP = "─" * 72


def load_dllm(device: str, compile_model: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(DLLM_MODEL_NAME, trust_remote_code=True)
    print(f"Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        DLLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()
    model.mdm_sample = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)
    if compile_model:
        print("Compiling model with torch.compile (reduce-overhead) ...")
        model = torch.compile(model, mode="reduce-overhead")
        print("Compilation done (first forward pass will still be slow).")
    print("Model ready.\n")
    return tokenizer, model


def build_padded_batch(messages_list, tokenizer, device):
    """Tokenize and pad a list of message sequences. Returns (batched_ids, seq_lens)."""
    input_ids_list = []
    seq_lens = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer([text], return_tensors="pt")["input_ids"][0]
        input_ids_list.append(ids)
        seq_lens.append(len(ids))

    max_len = max(seq_lens)
    padded = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), MASK_ID, dtype=torch.long)])
        padded.append(ids.unsqueeze(0))
    batched_ids = torch.cat(padded, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, device=device)
    return batched_ids, seq_len_tensor, seq_lens


def run_batch(messages_list, tokenizer, model, device, gen_kwargs, max_new_tokens):
    """Run batch_sample and return (texts, wall_s, total_steps, seq_lens, batch_tensor_len)."""
    batched_ids, seq_len_tensor, seq_lens = build_padded_batch(messages_list, tokenizer, device)
    batch_tensor_len = batched_ids.shape[1]

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    generated_ids, total_steps = model.mdm_sample(
        batched_ids,
        tokenizer=tokenizer,
        block_size=gen_kwargs["block_size"],
        small_block_size=gen_kwargs["small_block_size"],
        max_new_tokens=max_new_tokens,
        mask_id=MASK_ID,
        min_len=min(seq_lens),
        seq_len=seq_len_tensor,
        use_block_cache=gen_kwargs.get("use_block_cache", True),
        threshold=gen_kwargs["threshold"],
        temperature=gen_kwargs.get("temperature", 0.0),
        top_p=gen_kwargs.get("top_p", 0.95),
        stop_token=STOP_TOKEN,
    )

    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - t0

    texts = []
    gen_token_counts = []
    for i, sl in enumerate(seq_lens):
        gen_part = generated_ids[i][sl:]
        n_tok = int((gen_part != MASK_ID).sum().item())
        gen_token_counts.append(n_tok)
        texts.append(tokenizer.decode(gen_part, skip_special_tokens=True))

    return texts, wall_s, total_steps, seq_lens, gen_token_counts, batch_tensor_len


def print_batch_report(label, wall_s, total_steps, seq_lens, gen_token_counts, batch_tensor_len, max_new_tokens):
    n = len(seq_lens)
    total_gen = sum(gen_token_counts)
    total_prompt = sum(seq_lens)
    # Padding = (max_seq_len - each_seq_len) summed, these are "wasted" prompt padding tokens
    prompt_padding = sum(max(seq_lens) - sl for sl in seq_lens)
    # Generation mask tokens allocated = n * max_new_tokens (maximum possible)
    gen_alloc = n * max_new_tokens
    useful_tokens = total_prompt + total_gen
    total_tensor_tokens = batch_tensor_len * n  # rough: we measure batch_tensor_len per row
    # More precise: sum over rows of actual tensor length
    actual_total = batch_tensor_len * n  # padded batch tensor total token slots

    pad_pct = 100.0 * prompt_padding / (max(seq_lens) * n) if n > 1 else 0.0
    tps = total_gen / wall_s if wall_s > 0 else 0
    tps_per_sample = tps / n

    print(f"\n{label}")
    print(f"  Batch size          : {n}")
    print(f"  Wall time           : {wall_s:.3f}s  ({wall_s/n:.3f}s per sample)")
    print(f"  Total diff steps    : {total_steps}  ({total_steps/n:.1f} per sample)")
    print(f"  Tokens generated    : {total_gen}  ({total_gen/n:.1f} per sample)")
    print(f"  Tokens/step         : {total_gen/max(total_steps,1):.2f}  (higher = better parallelism)")
    print(f"  Gen throughput      : {tps:.1f} tok/s  ({tps_per_sample:.1f} per sample)")
    print(f"  Prompt lengths      : {seq_lens}  (range {min(seq_lens)}–{max(seq_lens)})")
    print(f"  Prompt padding waste: {prompt_padding} tokens ({pad_pct:.1f}% of prompt rows)")
    print(f"  Gen tokens/sample   : {gen_token_counts}")

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU mem (peak)      : {mem_mb:.0f} MB")


def resolve_diff_cap(spec, file_diffs, tokenizer):
    """
    Parse a diff-cap spec string and return (max_diff_chars, max_diff_tokens, label).

    Exactly one of max_diff_chars / max_diff_tokens is non-None (or both None for "none").

    Spec syntax:
      "none"           → no cap (full diff)
      "<N>"            → fixed N-character cap (e.g. "600")
      "tok:<N>"        → fixed N diff-token cap (e.g. "tok:80") — precise, char-independent
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
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            raw_lens.append(len(tokenizer([text], return_tensors="pt")["input_ids"][0]))

        sorted_lens = sorted(raw_lens)
        n_raw = len(sorted_lens)
        idx_f = pct / 100.0 * (n_raw - 1)
        lo, hi = int(idx_f), min(int(idx_f) + 1, n_raw - 1)
        target_len = int(sorted_lens[lo] + (idx_f - lo) * (sorted_lens[hi] - sorted_lens[lo]))

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
            f"Unknown config spec '{spec}'. "
            f"Valid: 'none', '<N>' (chars), 'tok:<N>' (diff tokens), "
            f"'adaptive', 'adaptive:<pct>'"
        )


def run_one_config(config_name, max_diff_chars, max_diff_tokens, diff_cap_label,
                   file_diffs, tokenizer, model, device,
                   gen_kwargs, max_new_tokens, batch_sizes, no_summaries,
                   sort_by_length=True, use_all_files=False):
    """Run sequential baseline + all batch sizes for one diff-cap configuration."""
    n_files = len(file_diffs)

    # ── Step 1: Pre-process diffs (apply truncation) ───────────────────────
    # processed: list of (fn, fd_proc, raw_metric, was_truncated)
    #   tok-mode:  raw_metric = raw diff token count; fd_proc = decoded truncated diff
    #   char-mode: raw_metric = raw diff char count;  fd_proc = original diff string
    processed = []
    for fn, fd in file_diffs:
        if max_diff_tokens is not None:
            raw_diff_ids = tokenizer(fd, return_tensors="pt")["input_ids"][0]
            raw_metric = len(raw_diff_ids)
            if raw_metric > max_diff_tokens:
                fd_proc = tokenizer.decode(raw_diff_ids[:max_diff_tokens], skip_special_tokens=True)
                was_truncated = True
            else:
                fd_proc = fd
                was_truncated = False
        else:
            raw_metric = len(fd)
            fd_proc = fd
            was_truncated = max_diff_chars is not None and raw_metric > max_diff_chars
        processed.append((fn, fd_proc, raw_metric, was_truncated))

    # ── Step 2: Build messages and tokenize ────────────────────────────────
    # In tok-mode the diff is already truncated; pass char_cap=None so it isn't truncated again.
    char_cap = None if max_diff_tokens is not None else max_diff_chars
    all_messages = [build_summary_messages(fn, fd_proc, max_diff_chars=char_cap)
                    for fn, fd_proc, _, _ in processed]
    all_ids = []
    for msgs in all_messages:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer([text], return_tensors="pt")["input_ids"][0]
        all_ids.append(ids)

    # ── Step 3: Optionally sort by prompt token length ─────────────────────
    if sort_by_length:
        order = sorted(range(n_files), key=lambda i: len(all_ids[i]))
        processed = [processed[i] for i in order]
        all_messages = [all_messages[i] for i in order]
        all_ids = [all_ids[i] for i in order]
        file_diffs = [file_diffs[i] for i in order]

    print(SEP)
    print(f"CONFIG     : {config_name}")
    print(f"Diff cap   : {diff_cap_label}")
    sort_note = "ascending prompt length (shortest-first)" if sort_by_length else "original diff order (--no-sort)"
    print(f"File order : {sort_note}")
    print(f"Batch sizes: {batch_sizes}")
    print(SEP)

    print("Prompt lengths per file (after diff cap):")
    for i, (fn, fd_proc, raw_metric, was_truncated) in enumerate(processed):
        if max_diff_tokens is not None:
            cap_note = (f"  [✂ {raw_metric} → ≤{max_diff_tokens} diff tok]" if was_truncated
                        else f"  [{raw_metric} diff tok, no cap]")
        else:
            cap_note = (f"  [✂ {raw_metric} → ≤{max_diff_chars} diff chars]" if was_truncated
                        else f"  [{raw_metric} diff chars, no cap]")
        print(f"  [{i}] {fn:<50}  {len(all_ids[i]):>5} tok{cap_note}")
    print()

    # Sequential baseline
    print(f"── Sequential baseline  (1 call × {n_files} files) ──")
    seq_times: list[float] = []
    seq_texts: list[str] = []
    seq_steps: list[int] = []
    seq_gen_tok: list[int] = []

    for i, msgs in enumerate(all_messages):
        torch.cuda.reset_peak_memory_stats(device)
        texts_i, w, steps, sl, gc, tl = run_batch(
            [msgs], tokenizer, model, device, gen_kwargs, max_new_tokens
        )
        seq_times.append(w)
        seq_texts.append(texts_i[0])
        seq_steps.append(steps)
        seq_gen_tok.append(gc[0])
        fn = file_diffs[i][0]
        print(f"  [{i}] {fn}:  {w:.3f}s  {steps} steps  {gc[0]} gen tok")

    total_seq = sum(seq_times)
    print(f"  TOTAL: {total_seq:.3f}s  ({total_seq / n_files:.3f}s avg per file)")
    print()

    if not no_summaries:
        print("── Generated summaries (sequential) ──")
        for i, (fn, _) in enumerate(file_diffs):
            print(f"\n  [{i}] {fn}:")
            for line in seq_texts[i].strip().splitlines():
                print(f"      {line}")
        print()

    # Batch runs
    speedup_table: list[tuple[int, float]] = []
    batch_results: dict = {}

    ran_effective_bs = set()
    for bs in batch_sizes:
        if bs > n_files:
            if use_all_files:
                effective_bs = n_files
            else:
                print(f"[batch={bs}] skipped — only {n_files} files available\n")
                continue
        else:
            effective_bs = bs

        if effective_bs in ran_effective_bs:
            print(f"[batch={bs}] → effective batch {effective_bs} (all files) already benchmarked above, skipping duplicate\n")
            continue
        ran_effective_bs.add(effective_bs)

        all_note = f" → using all {effective_bs} files" if bs != effective_bs else ""
        seq_baseline = sum(seq_times[:effective_bs])

        if effective_bs == 1:
            print_batch_report(
                f"── Batch size {bs}{all_note}  (file 0) ──",
                seq_times[0], seq_steps[0],
                [len(all_ids[0])], [seq_gen_tok[0]],
                len(all_ids[0]), max_new_tokens,
            )
            print(f"  Seq baseline (file 0)       : {seq_baseline:.3f}s")
            print(f"  Speedup vs sequential       : 1.00×  (this IS the baseline)")
            speedup_table.append((bs, 1.0))
            batch_results[bs] = {
                "wall_s": seq_times[0], "speedup": 1.0,
                "steps": seq_steps[0], "gen_tokens": [seq_gen_tok[0]],
                "texts": [seq_texts[0]],
            }
            continue

        msgs_subset = all_messages[:effective_bs]
        torch.cuda.reset_peak_memory_stats(device)

        try:
            texts, wall_s, total_steps, seq_lens, gen_counts, tensor_len = run_batch(
                msgs_subset, tokenizer, model, device, gen_kwargs, max_new_tokens
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\n── Batch size {bs}{all_note} ──")
            print(f"  OOM: batch={effective_bs} exceeded available GPU memory. Skipping.")
            torch.cuda.empty_cache()
            batch_results[bs] = {"wall_s": None, "speedup": None, "oom": True}
            continue

        speedup = seq_baseline / wall_s if wall_s > 0 else float("inf")
        speedup_table.append((bs, speedup))
        batch_results[bs] = {
            "wall_s": wall_s, "speedup": speedup,
            "steps": total_steps, "gen_tokens": gen_counts,
            "texts": texts,
        }

        print_batch_report(
            f"── Batch size {bs}{all_note}  (files 0..{effective_bs-1}) ──",
            wall_s, total_steps, seq_lens, gen_counts, tensor_len, max_new_tokens,
        )
        print(f"  Seq baseline (files 0..{effective_bs-1})  : {seq_baseline:.3f}s")
        print(f"  Speedup vs sequential          : {speedup:.2f}×")
        if speedup < 1.5:
            print(f"  ⚠  < 1.5× — batching provides little benefit at bs={bs}.")

        if not no_summaries:
            print(f"\n  Generated summaries (batch={bs}{all_note}):")
            for i, t in enumerate(texts):
                fn = file_diffs[i][0]
                print(f"\n    [{i}] {fn}:")
                for line in t.strip().splitlines():
                    print(f"        {line}")
            print()

    # Per-config speedup summary
    print(SEP)
    print(f"Speedup summary — config: {config_name}")
    print(f"  {'Batch':>5}  {'Speedup':>8}  {'Rating'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*20}")
    for bs, sp in speedup_table:
        rating = "good" if sp >= 2.0 else "marginal" if sp >= 1.5 else "poor"
        print(f"  {bs:>5}  {sp:>7.2f}×  {rating}")
    lengths = [len(ids) for ids in all_ids]
    print(f"\n  Prompt length variance: min={min(lengths)}  max={max(lengths)}  "
          f"waste≤{100*(max(lengths)-min(lengths))/max(lengths):.1f}%")
    print(SEP)
    print()

    return {
        "config": config_name,
        "diff_cap": max_diff_chars,
        "diff_cap_tokens": max_diff_tokens,
        "diff_cap_label": diff_cap_label,
        "prompt_lengths": [len(ids) for ids in all_ids],
        "seq_times": seq_times,
        "seq_total": total_seq,
        "seq_texts": seq_texts,
        "speedup_table": [[bs, sp] for bs, sp in speedup_table],
        "batches": {str(bs): v for bs, v in batch_results.items()},
    }


def _print_grid_table(all_results: list, batch_sizes: list) -> None:
    """Print a flat grid comparison table grouping results by task."""
    GRID_SEP = "═" * 88
    print(GRID_SEP)
    print("GRID COMPARISON  (speedup: batch vs same files sequentially)")
    task_w   = max((len(r.get("task_id", "?")) for r in all_results), default=10)
    config_w = max((len(r["config"])            for r in all_results), default=6)
    hdr = (f"  {'Task':<{task_w}}  {'BS':>4}  {'SBS':>4}  {'Thresh':>7}"
           f"  {'MNT':>5}  {'Config':<{config_w}}")
    for bs in batch_sizes:
        hdr += f"  {'bs='+str(bs):>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    prev_task = None
    for r in all_results:
        tid = r.get("task_id", "?")
        if prev_task is not None and tid != prev_task:
            print()  # blank line between tasks
        prev_task = tid
        row = (f"  {tid:<{task_w}}"
               f"  {r.get('block_size', '?'):>4}"
               f"  {r.get('small_block_size', '?'):>4}"
               f"  {r.get('threshold', 0.0):>7.2f}"
               f"  {r.get('max_new_tokens', '?'):>5}"
               f"  {r['config']:<{config_w}}")
        for bs in batch_sizes:
            bdata = r["batches"].get(str(bs), {})
            if not bdata:
                cell = "─"
            elif bdata.get("oom"):
                cell = "OOM"
            elif bdata.get("speedup") is not None:
                cell = f"{bdata['speedup']:.2f}×"
            else:
                cell = "—"
            row += f"  {cell:>7}"
        print(row)
    print(GRID_SEP)
    print()


def main():
    p = argparse.ArgumentParser(
        description="dLLM batch-size benchmark — grid search over tasks and gen params"
    )
    p.add_argument("-i", "--input", required=True, help="Input JSONL tasks file")
    p.add_argument("--sample-task", type=int, default=0,
                   help="Index of the task to use when no --task-id is given (default: 0)")
    p.add_argument("--task-id", default=None,
                   help="Comma-separated task IDs to benchmark (all must exist in the input). "
                        "Example: --task-id apache_mesos_0597b3c,apache_kafka_460b3a6")
    p.add_argument("--device", default="cuda:0")
    # ── dLLM grid params — all accept comma-separated values ──────────────
    p.add_argument("--block-sizes", default="32",
                   help="Comma-separated block sizes to sweep (default: 32). "
                        "block_size must be divisible by small_block_size. "
                        "Example: --block-sizes 32,64")
    p.add_argument("--small-block-sizes", default="8",
                   help="Comma-separated small block sizes to sweep (default: 8). "
                        "Example: --small-block-sizes 8,16")
    p.add_argument("--thresholds", default="0.8",
                   help="Comma-separated unmasking thresholds to sweep (default: 0.8). "
                        "Lower = faster but potentially lower quality. "
                        "Example: --thresholds 0.8,0.7,0.6,0.5")
    p.add_argument("--max-new-tokens", default="128",
                   help="Comma-separated max-new-tokens values to sweep (default: 128). "
                        "Example: --max-new-tokens 128,256")
    p.add_argument("--batch-sizes", default="1,2,4,8",
                   help="Comma-separated batch sizes to test (default: 1,2,4,8)")
    # ── Single-config diff cap (ignored when --configs is set) ────────────
    p.add_argument("--max-diff-chars", type=int, default=None,
                   help="Fixed diff character cap. Ignored when --configs is set.")
    p.add_argument("--adaptive-diff-chars", type=float, nargs="?", const=50.0, default=None,
                   metavar="PERCENTILE",
                   help="Adaptive diff cap at this percentile (0-100). "
                        "Bare flag → p50. Ignored when --configs is set.")
    # ── Multi-config sweep ────────────────────────────────────────────────
    p.add_argument("--configs", metavar="SPEC[,SPEC,...]", default=None,
                   help="Comma-separated diff-cap configs to sweep in one run. "
                        "Spec: 'none' | '<N>' (chars) | 'tok:<N>' (diff tokens) | "
                        "'adaptive' | 'adaptive:<pct>'. "
                        "Example: --configs none,800,600,tok:80,adaptive.")
    p.add_argument("--output-file", metavar="FILE", default=None,
                   help="Save all results to this JSON file.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the dLLM with torch.compile (reduce-overhead mode). "
                        "First batch is slow (compilation); subsequent ones are faster.")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip the per-(block-size, small-block-size) warmup calls. "
                        "By default one real mdm_sample is run per unique (bs, sbs) pair "
                        "so all grid points start with primed CUDA kernels.")
    p.add_argument("--no-summaries", action="store_true",
                   help="Suppress printing of generated summaries (perf numbers still shown)")
    p.add_argument("--no-sort", action="store_true",
                   help="Disable the default ascending sort by prompt token length.")
    p.add_argument("--use-all-files", action="store_true",
                   help="When a requested batch size exceeds the number of available files, "
                        "run all files as one batch instead of skipping. If multiple batch "
                        "sizes map to the same effective file count, only the first is run.")
    args = p.parse_args()

    # ── Parse multi-value grid params ─────────────────────────────────────
    block_sizes_grid = sorted(set(int(x)   for x in args.block_sizes.split(",")),       reverse=True)
    sbs_grid         = sorted(set(int(x)   for x in args.small_block_sizes.split(",")), reverse=True)
    thresholds_grid  = sorted(set(float(x) for x in args.thresholds.split(",")),        reverse=True)
    mnt_grid         = sorted(set(int(x)   for x in args.max_new_tokens.split(",")))
    batch_sizes      = sorted(set(int(x)   for x in args.batch_sizes.split(",")))

    # ── Load & select tasks ───────────────────────────────────────────────
    all_tasks = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tasks.append(json.loads(line))

    if args.task_id:
        task_id_list = [t.strip() for t in args.task_id.split(",")]
        selected_tasks = []
        for tid in task_id_list:
            t = next((t for t in all_tasks if t["task_id"] == tid), None)
            if t is None:
                print(f"Task ID '{tid}' not found in {args.input}.", file=sys.stderr)
                sys.exit(1)
            selected_tasks.append(t)
    else:
        selected_tasks = [all_tasks[args.sample_task]]

    # ── Resolve diff-cap specs (same for all grid points) ─────────────────
    if args.configs:
        specs = [s.strip() for s in args.configs.split(",")]
    else:
        if args.adaptive_diff_chars is not None:
            pct = args.adaptive_diff_chars
            specs = ["adaptive" if pct == 50.0 else f"adaptive:{pct}"]
        elif args.max_diff_chars is not None:
            specs = [str(args.max_diff_chars)]
        else:
            specs = ["none"]

    # ── Validate (block_size, sbs) pairs ──────────────────────────────────
    valid_bs_sbs = [(bs, sbs)
                    for bs, sbs in itertools.product(block_sizes_grid, sbs_grid)
                    if bs % sbs == 0]
    invalid_pairs = [(bs, sbs)
                     for bs, sbs in itertools.product(block_sizes_grid, sbs_grid)
                     if bs % sbs != 0]
    if invalid_pairs:
        print(f"⚠  Skipping invalid (block_size, sbs) pairs "
              f"(block_size must be divisible by sbs): {invalid_pairs}")

    # ── Print grid plan ───────────────────────────────────────────────────
    n_gen_combos = len(valid_bs_sbs) * len(thresholds_grid) * len(mnt_grid)
    n_total = len(selected_tasks) * n_gen_combos * len(specs)
    print(f"Grid plan: {len(selected_tasks)} task(s) × {n_gen_combos} gen-kwarg combo(s)"
          f" × {len(specs)} diff-cap spec(s) = {n_total} config run(s)")
    print(f"  tasks      : {[t['task_id'] for t in selected_tasks]}")
    print(f"  (bs, sbs)  : {valid_bs_sbs}  thresholds: {thresholds_grid}  mnt: {mnt_grid}")
    print(f"  specs      : {specs}  batch_sizes: {batch_sizes}")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    device = args.device
    tokenizer, model = load_dllm(device, compile_model=args.compile)

    # ── Warmup: one real mdm_sample per unique valid (block_size, sbs) pair
    if not args.no_warmup:
        first_file_diffs = get_per_file_diffs(selected_tasks[0])
        _fn, _fd = first_file_diffs[0]
        _warmup_msgs = build_summary_messages(_fn, _fd, max_diff_chars=600)
        print(f"Warming up {len(valid_bs_sbs)} (block_size, sbs) combination(s) ...")
        for _bs, _sbs in valid_bs_sbs:
            _gk = {
                "block_size": _bs, "small_block_size": _sbs,
                "threshold": thresholds_grid[0], "use_block_cache": True,
                "temperature": 0.0, "top_p": 0.95,
            }
            run_batch([_warmup_msgs], tokenizer, model, device, _gk, mnt_grid[0])
            torch.cuda.synchronize(device)
            print(f"  block_size={_bs}  sbs={_sbs} ✓")
        torch.cuda.reset_peak_memory_stats(device)
        print("Warm-up complete — CUDA kernels primed for all block-size variants.\n")

    # ── Main grid loop ────────────────────────────────────────────────────
    all_results = []
    for task in selected_tasks:
        task_id = task["task_id"]
        file_diffs = get_per_file_diffs(task)
        n_files = len(file_diffs)

        print("═" * 72)
        print(f"TASK: {task_id}  ({n_files} files)")
        print(f"  {', '.join(fn for fn, _ in file_diffs)}")
        print("═" * 72)
        print()

        for (block_size, sbs), threshold, mnt in itertools.product(
            valid_bs_sbs, thresholds_grid, mnt_grid
        ):
            gen_kwargs = {
                "block_size": block_size,
                "small_block_size": sbs,
                "threshold": threshold,
                "use_block_cache": True,
                "temperature": 0.0,
                "top_p": 0.95,
            }
            for spec in specs:
                max_diff_chars, max_diff_tokens, diff_cap_label = resolve_diff_cap(
                    spec, file_diffs, tokenizer
                )
                result = run_one_config(
                    config_name=spec,
                    max_diff_chars=max_diff_chars,
                    max_diff_tokens=max_diff_tokens,
                    diff_cap_label=diff_cap_label,
                    file_diffs=file_diffs,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    gen_kwargs=gen_kwargs,
                    max_new_tokens=mnt,
                    batch_sizes=batch_sizes,
                    no_summaries=args.no_summaries,
                    sort_by_length=not args.no_sort,
                    use_all_files=args.use_all_files,
                )
                result["task_id"]        = task_id
                result["block_size"]     = block_size
                result["small_block_size"] = sbs
                result["threshold"]      = threshold
                result["max_new_tokens"] = mnt
                all_results.append(result)

    # ── Final grid comparison table ───────────────────────────────────────
    if len(all_results) > 1:
        _print_grid_table(all_results, batch_sizes)

    # ── Save results ──────────────────────────────────────────────────────
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved → {args.output_file}")


if __name__ == "__main__":
    main()

