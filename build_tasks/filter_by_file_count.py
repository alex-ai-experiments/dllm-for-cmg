#!/usr/bin/env python3
"""
Filter tasks_tags.jsonl by number of changed files.

Usage:
    python build_tasks/filter_by_file_count.py
    python build_tasks/filter_by_file_count.py --min-files 3 --max-files 10
    python build_tasks/filter_by_file_count.py --min-files 2 --max-files 5 \
        -i build_tasks/tasks_tags.jsonl -o build_tasks/tasks_2to5files.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Filter a task JSONL by number of changed files."
    )
    p.add_argument(
        "-i", "--input",
        default="build_tasks/tasks_tags.jsonl",
        help="Input JSONL file (default: build_tasks/tasks_tags.jsonl)",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output JSONL file. Defaults to "
            "build_tasks/tasks_<min>to<max>files.jsonl"
        ),
    )
    p.add_argument(
        "--min-files", type=int, default=3,
        help="Minimum number of files in a task (inclusive, default: 3)",
    )
    p.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of files in a task (inclusive, default: no limit)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.min_files < 1:
        print("--min-files must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_files is not None and args.max_files < args.min_files:
        print("--max-files must be >= --min-files", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    elif args.max_files is None:
        output_path = input_path.parent / f"tasks_min{args.min_files}files.jsonl"
    else:
        output_path = input_path.parent / f"tasks_{args.min_files}to{args.max_files}files.jsonl"

    tasks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    filtered = [
        t for t in tasks
        if len(t.get("files", [])) >= args.min_files
        and (args.max_files is None or len(t.get("files", [])) <= args.max_files)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in filtered:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Print a brief summary
    from collections import Counter
    counts = Counter(len(t["files"]) for t in filtered)
    max_label = str(args.max_files) if args.max_files is not None else "∞"
    print(f"Input : {len(tasks):,} tasks  ({input_path})")
    print(f"Filter: {args.min_files} ≤ files ≤ {max_label}")
    print(f"Output: {len(filtered):,} tasks  ({output_path})")
    print()
    print(f"{'Files':>6}  {'Tasks':>6}")
    for k in sorted(counts):
        print(f"{k:>6}  {counts[k]:>6}")


if __name__ == "__main__":
    main()
