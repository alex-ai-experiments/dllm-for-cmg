#!/usr/bin/env python3
"""
Extract task metadata (task_id, file count, main language) from a tasks JSONL file,
filtered by number of files.

Usage examples:
    # Exactly 10 files
    python extract_tasks_by_filecount.py -i tasks_4to12files.jsonl --count 10

    # Between 10 and 12 files (inclusive)
    python extract_tasks_by_filecount.py -i tasks_4to12files.jsonl --min 10 --max 12

    # At least 8 files, no upper limit
    python extract_tasks_by_filecount.py -i tasks_4to12files.jsonl --min 8

    # Custom output path and input from parent dir
    python extract_tasks_by_filecount.py -i ../build_tasks/tasks_tags.jsonl --min 10 --max 12 -o my_output.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Extensions treated as non-code — excluded from language detection
NON_CODE_EXTS = {
    # Documentation / markup
    "md", "txt", "rst", "adoc", "html", "css", "svg",
    # Config / data
    "xml", "json", "yaml", "yml", "csv", "ini", "cfg", "toml",
    "lock", "gitignore", "geojson", "properties",
    # Images
    "png", "jpg", "jpeg", "gif", "ico", "graffle",
    # Archives
    "zip", "gz", "tar", "jar", "war", "ear", "aar", "snap",
    # Compiled / binary
    "so", "dll", "exe", "bin", "dylib", "class", "o", "obj",
    # Keystores / certs
    "jks", "p12", "pem", "crt", "key", "pub", "sha1",
    # Data files
    "xls", "xlsx", "avro", "parquet",
    # Platform-specific binaries
    "android_arm64", "android_armv7", "arm64", "armv6", "armv7",
    "amzn_linux_cpu", "ubuntu", "ubuntu1404_cuda75_cudnn5",
    "jetson", "ubuntu_cpu", "ubuntu_gpu",
}

# Map file extensions to human-readable language names
EXT_TO_LANG = {
    "py":    "Python",
    "java":  "Java",
    "scala": "Scala",
    "kt":    "Kotlin",
    "go":    "Go",
    "rs":    "Rust",
    "cpp":   "C++",
    "cc":    "C++",
    "cxx":   "C++",
    "c":     "C",
    "h":     "C/C++ header",
    "hpp":   "C/C++ header",
    "cs":    "C#",
    "ts":    "TypeScript",
    "tsx":   "TypeScript",
    "js":    "JavaScript",
    "jsx":   "JavaScript",
    "rb":    "Ruby",
    "php":   "PHP",
    "sh":    "Shell",
    "bash":  "Shell",
    "zsh":   "Shell",
    "yaml":  "YAML",
    "yml":   "YAML",
    "json":  "JSON",
    "xml":   "XML",
    "toml":  "TOML",
    "conf":  "Config",
    "cfg":   "Config",
    "ini":   "Config",
    "properties": "Config",
    "sql":   "SQL",
    "html":  "HTML",
    "css":   "CSS",
    "md":    "Markdown",
    "rst":   "reStructuredText",
    "lua":   "Lua",
    "r":     "R",
    "swift": "Swift",
    "proto": "Protobuf",
    "gradle": "Gradle",
    "tf":    "Terraform",
    "t":     "Perl/Test",
}


def detect_language(files: list[str]) -> str:
    """Return the dominant language based on file extensions."""
    ext_counts: Counter = Counter()
    for path in files:
        parts = path.rsplit(".", 1)
        if len(parts) == 2:
            ext = parts[1].lower()
            if ext in NON_CODE_EXTS:
                continue
            lang = EXT_TO_LANG.get(ext, ext)
            ext_counts[lang] += 1
    if not ext_counts:
        return "unknown"
    return ext_counts.most_common(1)[0][0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract task IDs filtered by file count, with language detection."
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="Input JSONL tasks file (can be relative to this script's directory).",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output JSON file path. Default: <input_stem>_N[to M]files.json "
             "written next to this script.",
    )
    p.add_argument(
        "--count", type=int, default=None,
        help="Exact file count to match (overrides --min / --max).",
    )
    p.add_argument(
        "--min", type=int, default=None,
        help="Minimum file count (inclusive). Default: no lower bound.",
    )
    p.add_argument(
        "--max", type=int, default=None,
        help="Maximum file count (inclusive). Default: no upper bound.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Stop after collecting this many matching tasks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve input path relative to this script's directory if not absolute
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine file-count filter bounds
    if args.count is not None:
        lo = hi = args.count
        count_label = f"{args.count}files"
    else:
        lo = args.min  # None = no lower bound
        hi = args.max  # None = no upper bound
        if lo is None and hi is None:
            print("Error: specify at least one of --count, --min, --max.", file=sys.stderr)
            sys.exit(1)
        parts = []
        if lo is not None:
            parts.append(f"min{lo}")
        if hi is not None:
            parts.append(f"max{hi}")
        count_label = "_".join(parts)

    # Build default output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
    else:
        stem = input_path.stem
        output_path = script_dir / f"{stem}_{count_label}.json"

    def matches(n: int) -> bool:
        if lo is not None and n < lo:
            return False
        if hi is not None and n > hi:
            return False
        return True

    results = []
    seen = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            n_files = len(task.get("files", []))
            if matches(n_files):
                results.append({
                    "task_id":   task["task_id"],
                    "num_files": n_files,
                    "language":  detect_language(task.get("files", [])),
                    "files":     task.get("files", []),
                })
                seen += 1
                if args.limit is not None and seen >= args.limit:
                    break

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Matched {len(results)} task(s).")
    print(f"Output → {output_path}")
    if results:
        lang_summary = Counter(r["language"] for r in results)
        print("Language breakdown:")
        for lang, count in lang_summary.most_common():
            print(f"  {lang:<20} {count}")


if __name__ == "__main__":
    main()
