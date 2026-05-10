"""
Build model-agnostic training data from ApacheCM/full.jsonl.

Steps:
  1. Load test.jsonl commit_shas (the evaluation set)
  2. Stream full.jsonl, dropping any entry whose commit_sha is in the test set
  3. Write the deduped subset to a temp file
  4. Run build_tasks.make_tasks() to produce cleaned training tasks
     (same cleaning pipeline as evaluation, with matching prompt format)

Output:
  build_tasks/train_tasks.jsonl   – training tasks (messages + label)
  build_tasks/train_labels.jsonl  – just task_id + label

Usage:
  python 65_build_training_data.py [--max-diff-length 30000] [--stats]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Add build_tasks to path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "build_tasks"))
from build_tasks import make_tasks

# ── Paths ────────────────────────────────────────────────────────────────────
FULL_PATH = "datasets/ApacheCM/full.jsonl"
TEST_PATH = "datasets/ApacheCM/test.jsonl"
OUT_TASKS = "build_tasks/train_tasks.jsonl"


def load_test_shas(path: str) -> set[str]:
    shas = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                shas.add(json.loads(line)["commit_sha"])
    print(f"Loaded {len(shas)} test commit SHAs to exclude")
    return shas


def deduplicate(full_path: str, test_shas: set[str], out_path: str) -> int:
    """Write full.jsonl entries whose commit_sha is NOT in test_shas."""
    kept = 0
    skipped = 0
    with open(full_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["commit_sha"] in test_shas:
                skipped += 1
                continue
            fout.write(line)
            kept += 1
    print(f"Deduplication: {kept} kept, {skipped} removed (test overlap)")
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Build training tasks from ApacheCM/full.jsonl (excluding test set)"
    )
    parser.add_argument(
        "--max-diff-length", type=int, default=None,
        help="Skip commits whose filtered diff exceeds N chars (default: no limit)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Generate histogram + per-task stats JSON"
    )
    parser.add_argument(
        "--omit-message-tags", action="store_true",
        help="Use plain-text prompt format without <msg> tags"
    )
    args = parser.parse_args()

    # 1. Load test set commit SHAs
    test_shas = load_test_shas(TEST_PATH)

    # 2. Write deduped full.jsonl to a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    tmp.close()
    try:
        kept = deduplicate(FULL_PATH, test_shas, tmp.name)
        if kept == 0:
            print("ERROR: No entries remain after deduplication!")
            return

        # 3. Run through the build_tasks pipeline
        print(f"\nBuilding tasks from {kept} entries …")
        make_tasks(
            dataset_path=tmp.name,
            tasks_path=OUT_TASKS,
            max_diff_length=args.max_diff_length,
            run_stats=args.stats,
            omit_message_tags=args.omit_message_tags,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    # 4. Summary
    count = 0
    with open(OUT_TASKS, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    print(f"\nFinal training set: {count} tasks → {OUT_TASKS}")


if __name__ == "__main__":
    main()
