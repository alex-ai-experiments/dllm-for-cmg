#!/usr/bin/env python3
"""
Select the best 250 tasks suited for evaluating dLLM performance.

Selection criteria (tasks where dLLMs should perform well):
  1. Multi-file commits (2+ files) — dLLM batching advantage
  2. Moderate per-file diff sizes (200-3000 chars avg) — fits dLLM context
  3. No single file exceeds 5000 chars — avoids OOM / truncation
  4. Prefers tasks with more files (better batch utilization)
  5. Prefers tasks with consistent diff sizes across files (better padding efficiency)
  6. Code-only (no excluded file extensions eating into diff budget)

Scoring: tasks are scored by a composite of file count, diff-size suitability,
and diff-size consistency. Top 250 are selected.

Usage:
    python build_tasks/select_dllm_tasks.py
    python build_tasks/select_dllm_tasks.py --n 250 --output build_tasks/dllm_eval_tasks.jsonl
    python build_tasks/select_dllm_tasks.py --n 50 --output build_tasks/dllm_grid_search_tasks.jsonl
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def score_task(task_stat: dict) -> float:
    """
    Score a task for dLLM suitability (higher = better).

    Components:
      - file_bonus:   log2(n_files) — more files = better batch utilization
      - size_score:   bell curve around ideal avg diff (600-1500 chars)
      - consistency:  1 / (1 + CV) — lower coefficient of variation = better padding
      - code_purity:  1.0 if no excluded extensions, 0.5 otherwise
    """
    diffs = list(task_stat["diff_length_per_file"].values())
    n_files = len(diffs)
    if n_files < 2:
        return -1.0  # single-file tasks don't benefit from dLLM batching

    avg_diff = sum(diffs) / n_files
    max_diff = max(diffs)

    # Hard filter: no extreme diffs
    if avg_diff < 150 or avg_diff > 4000 or max_diff > 5000:
        return -1.0

    # File count bonus (log scale, capped)
    file_bonus = math.log2(min(n_files, 10))  # 0.0 → 3.32

    # Diff size: bell curve centered around 800 chars (ideal for dLLM summarization)
    ideal_avg = 800
    size_score = math.exp(-((avg_diff - ideal_avg) / 1200) ** 2)  # 0.0 → 1.0

    # Consistency: coefficient of variation (lower = more uniform diffs = less padding waste)
    if n_files >= 2 and avg_diff > 0:
        cv = statistics.stdev(diffs) / avg_diff
        consistency = 1.0 / (1.0 + cv)
    else:
        consistency = 0.5

    # Code purity
    code_purity = 1.0 if not task_stat.get("excluded_file_extensions") else 0.7

    return file_bonus * 1.5 + size_score * 2.0 + consistency * 1.0 + code_purity * 0.5


def main():
    p = argparse.ArgumentParser(description="Select best tasks for dLLM evaluation.")
    p.add_argument("--stats", default="build_tasks/task_stats.json",
                   help="Path to task_stats.json")
    p.add_argument("--tasks", default="build_tasks/tasks_5k.jsonl",
                   help="Path to tasks JSONL (for full task data)")
    p.add_argument("--n", type=int, default=250,
                   help="Number of tasks to select")
    p.add_argument("--output", default="build_tasks/dllm_eval_tasks_250.jsonl",
                   help="Output path for selected task IDs")
    p.add_argument("--output-ids", default=None,
                   help="Also write a plain text file with one task_id per line")
    args = p.parse_args()

    # Load stats
    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)
    print(f"Loaded {len(stats)} task stats")

    # Score all tasks
    scored = []
    for t in stats:
        s = score_task(t)
        if s > 0:
            scored.append((s, t["task_id"]))

    scored.sort(reverse=True)
    print(f"Eligible tasks (score > 0): {len(scored)}")

    selected_ids = set(tid for _, tid in scored[:args.n])
    print(f"Selected top {args.n} tasks")

    # Print score distribution
    sel_scores = [s for s, _ in scored[:args.n]]
    print(f"  Score range: {sel_scores[-1]:.3f} — {sel_scores[0]:.3f}")
    print(f"  Mean score:  {sum(sel_scores)/len(sel_scores):.3f}")

    # Load full tasks and write selected
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.tasks, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if task["task_id"] in selected_ids:
                fout.write(json.dumps(task, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} tasks → {out_path}")

    # Optionally write plain ID list
    ids_path = args.output_ids or str(out_path).replace(".jsonl", "_ids.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        # Maintain score order
        for _, tid in scored[:args.n]:
            f.write(tid + "\n")
    print(f"Wrote task IDs → {ids_path}")

    # Summary stats for selected tasks
    stats_lookup = {t["task_id"]: t for t in stats}
    file_counts = []
    avg_diffs = []
    for _, tid in scored[:args.n]:
        t = stats_lookup[tid]
        diffs = list(t["diff_length_per_file"].values())
        file_counts.append(len(diffs))
        avg_diffs.append(sum(diffs) / len(diffs))

    print(f"\nSelected task stats:")
    print(f"  Files/task: mean={sum(file_counts)/len(file_counts):.1f}, "
          f"median={sorted(file_counts)[len(file_counts)//2]}, "
          f"range={min(file_counts)}-{max(file_counts)}")
    print(f"  Avg diff/file: mean={sum(avg_diffs)/len(avg_diffs):.0f}, "
          f"median={sorted(avg_diffs)[len(avg_diffs)//2]:.0f}")


if __name__ == "__main__":
    main()
