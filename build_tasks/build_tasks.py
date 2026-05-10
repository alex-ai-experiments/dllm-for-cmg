import json
import re
import pathlib
import argparse

from dataclasses import dataclass, field
from collections import Counter

# ── Optional stats deps (only needed with --stats) ───────────────────────────
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ── Config ────────────────────────────────────────────────────────────────────

# Set to a positive integer to skip commits whose filtered diff exceeds this
# many characters.  None (default) means no limit — include everything.
MAX_DIFF_LENGTH: int | None = None


@dataclass
class SkipStats:
    reasons: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=lambda: {})  # reason → first few task_ids

    def record(self, reason: str, task_id: str):
        self.reasons[reason] += 1
        if reason not in self.examples:
            self.examples[reason] = []
        if len(self.examples[reason]) < 3:  # keep up to 3 examples per reason
            self.examples[reason].append(task_id)

    def report(self):
        total = sum(self.reasons.values())
        print(f"\n{'─' * 60}")
        print(f"  Skipped {total} tasks total")
        print(f"{'─' * 60}")
        for reason, count in self.reasons.most_common():
            print(f"  {count:>4}  {reason}")
            for ex in self.examples[reason]:
                print(f"           ↳ {ex}")
        print(f"{'─' * 60}\n")

# Sample generic loader if you don't have one
def load_from_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# ── Compile-time constants ────────────────────────────────────────────────────

# Apache-style JIRA module prefixes — leading noise with no semantic payload
_MODULE_PREFIXES = sorted([
    'AIRFLOW', 'ARROW', 'AVRO', 'AWS', 'AZURE', 'BUILD', 'CASSANDRA',
    'CI', 'CLI', 'CLOUD', 'CONFIG', 'CORE', 'DEPLOY', 'DOCKER', 'DOCS',
    'DRIVER', 'ELASTIC', 'EXECUTOR', 'FLINK', 'GCP', 'HADOOP', 'HDFS',
    'HIVE', 'HTTP', 'INFRA', 'K8S', 'KAFKA', 'LOGGING', 'METRICS', 'ML',
    'MONITORING', 'PERF', 'PYTHON', 'REST', 'RPC', 'SCHEDULER', 'SECURITY',
    'SPARK', 'SQL', 'STREAMING', 'TEST', 'UI', 'WEB', 'WORKER', 'YARN',
], key=len, reverse=True)  # longest first so greedier alternatives match first

_MODULE_PREFIX_RE = re.compile(
    r'^\s*(?:' + '|'.join(re.escape(p) for p in _MODULE_PREFIXES) + r')\s+',
    re.IGNORECASE,
)

_ATTRIBUTION_RE = re.compile(
    r'\b(?:via|patch(?:ed)?\s+by|contributed\s+by|co[\s\-]authored\s+by|authored\s+by|'
    r'reviewed\s+by|reported\s+by|suggested\s+by|thanks\s+to|'
    r'signed[\s\-]off[\s\-]by)\b'
    r'(?:\s+[A-Za-z][\w.]*(?:\s+[A-Za-z][\w.]*)*)?',
    re.IGNORECASE,
)

_TRAILING_ATTRIBUTION_RE = re.compile(
    r'\s*\.\s*(?:via|patch(?:ed)?\s+by|contributed\s+by|co[\s\-]authored\s+by|authored\s+by|'
    r'reviewed\s+by|reported\s+by|suggested\s+by|thanks\s+to|'
    r'signed[\s\-]off[\s\-]by)\b.*$',
    re.IGNORECASE | re.DOTALL,
)

_FILLER_RE = re.compile(
    r'^\s*(?:wip|clean\s*up|cleanup|chore|hot\s*fix|hotfix|minor|nit|'
    r'trivial|addendum|follow\s*up|followup|polish|cosmetic|style|typo)'
    r'\b[\s:,\-]*',
    re.IGNORECASE,
)


# ── Non-code extensions ───────────────────────────────────────────────────────

NON_CODE_EXTENSIONS = {
    '.md', '.txt', '.xml', '.json', '.yaml', '.yml', '.csv',
    '.rst', '.ini', '.cfg', '.toml', '.lock', '.gitignore',
    '.html', '.css', '.svg', '.png', '.jpg', '.jpeg', '.adoc', '.geojson'
}

# Extension → language display name (expandable)
_EXT_TO_LANG: dict[str, str] = {
    '.py':    'Python',
    '.java':  'Java',
    '.scala': 'Scala',
    '.kt':    'Kotlin',
    '.groovy':'Groovy',
    '.js':    'JavaScript',
    '.ts':    'TypeScript',
    '.jsx':   'JavaScript (JSX)',
    '.tsx':   'TypeScript (TSX)',
    '.c':     'C',
    '.h':     'C/C++ Header',
    '.cpp':   'C++',
    '.cc':    'C++',
    '.cxx':   'C++',
    '.hpp':   'C++ Header',
    '.cs':    'C#',
    '.go':    'Go',
    '.rs':    'Rust',
    '.rb':    'Ruby',
    '.php':   'PHP',
    '.swift': 'Swift',
    '.r':     'R',
    '.m':     'Objective-C / MATLAB',
    '.sh':    'Shell',
    '.bash':  'Bash',
    '.zsh':   'Zsh',
    '.ps1':   'PowerShell',
    '.sql':   'SQL',
    '.lua':   'Lua',
    '.pl':    'Perl',
    '.ex':    'Elixir',
    '.exs':   'Elixir Script',
    '.erl':   'Erlang',
    '.hs':    'Haskell',
    '.clj':   'Clojure',
    '.proto': 'Protobuf',
    '.thrift':'Thrift',
}


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def semantic_clean(label: str) -> str:
    label = _TRAILING_ATTRIBUTION_RE.sub('', label)
    label = _ATTRIBUTION_RE.sub('', label)
    for _ in range(4):
        label, n = _MODULE_PREFIX_RE.subn('', label, count=1)
        if not n:
            break
    label = re.sub(r'\b[A-Z]{2,}[\s\-]\d+\b', '', label)
    label = re.sub(r'\b[A-Z]{2,}\d+\b\s*:?', '', label)
    label = re.sub(r'\b(?:bug|issue|ticket|jira|pr)\s*[:\-]?\s*\d+\b', '', label, flags=re.IGNORECASE)
    label = re.sub(r'\s*\.\s*', ' ', label)
    label = re.sub(r'\bv?\d+(?:\.\d+){1,5}(?:-[a-zA-Z0-9]+)?\b', '', label)
    label = re.sub(r'\bv\d+\b', '', label, flags=re.IGNORECASE)
    label = re.sub(r'(?<!\S)\d+(?!\S)', '', label)
    label = label.lower()
    label = re.sub(r'\bbug\s*fix(?:e[sd]|ing)?\b', 'fix', label)
    label = re.sub(r'\bfix(?:e[sd]|ing)\b', 'fix', label)
    label = re.sub(r'(?<!\w)bug(?!\s+\w+\s+fix)\b', 'fix', label)
    for _ in range(3):
        new = _FILLER_RE.sub('', label)
        if new == label:
            break
        label = new
    label = re.sub(r'\s+', ' ', label).strip().strip('.,;:/-')
    return label


def clean_label(label: str) -> str:
    label = re.sub(r'https?://\S+', '', label)
    label = re.sub(
        r'`?[\w-]+\.(?:com|org|net|io|gov|edu|co|ai|dev|info|me|us|uk|de|fr'
        r'|cn|jp|ru|apache|jira|github|gitlab|bitbucket|sonar)\b`?(?!\s*\()',
        '', label, flags=re.IGNORECASE,
    )
    label = re.sub(r'\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d+\b', '', label)
    label = re.sub(r'(?:[\w][\w/.:-]*)?\s*#\d+', '', label)
    label = re.sub(r'\s+', ' ', label).strip().strip('.,;:/-')
    return label


def normalize_label(label: str) -> str:
    label = re.sub(
        r'[Cc]ontributed\s+by\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\.?',
        '', label,
    )
    label = re.sub(r'\(\s*[A-Z][a-zA-Z][^)]*\)', '', label)
    label = re.sub(r'[(){}\[\]]', ' ', label)
    label = re.sub(r'[-/#]+', ' ', label)
    label = re.sub(r"""["`';,*]+""", ' ', label)
    label = re.sub(r'\s+', ' ', label).strip()
    label = re.sub(r'\b[a-zA-Z]\b', '', label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label


# ── Diff utilities ────────────────────────────────────────────────────────────

def split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """
    Split a multi-file diff into (filename, full_chunk) pairs.
    Returns every file, regardless of extension.
    """
    parts = re.split(r'(^diff --git a/.* b/.*$)', diff_text, flags=re.MULTILINE)
    files: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        header  = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        m = re.search(r' b/(.+)$', header)
        if m:
            files.append((m.group(1).strip(), header + content))
    return files


def filter_diff(diff_text: str) -> str:
    """Keep only code files (non-NON_CODE_EXTENSIONS)."""
    chunks = [
        chunk
        for filename, chunk in split_diff_by_file(diff_text)
        if pathlib.Path(filename).suffix.lower() not in NON_CODE_EXTENSIONS
    ]
    return "".join(chunks).strip()


def diff_length_per_file(diff_text: str) -> dict[str, int]:
    """Return {filename: len(chunk)} for every file in the raw diff."""
    return {
        filename: len(chunk)
        for filename, chunk in split_diff_by_file(diff_text)
    }


def classify_diff_languages(diff_text: str) -> tuple[list[str], list[str]]:
    """
    Returns (code_languages, excluded_extensions) from all files in the diff.
    Uses the raw diff so excluded files are still visible.
    """
    seen_langs: set[str] = set()
    seen_excl:  set[str] = set()
    for filename, _ in split_diff_by_file(diff_text):
        ext = pathlib.Path(filename).suffix.lower()
        if ext in NON_CODE_EXTENSIONS:
            seen_excl.add(ext)
        else:
            lang = _EXT_TO_LANG.get(ext, ext if ext else "unknown")
            seen_langs.add(lang)
    return sorted(seen_langs), sorted(seen_excl)


# ── Prompt Templates ──────────────────────────────────────────────────────────

PROMPT_ROLE = "You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit."

PROMPT_FORMAT_AND_OUTPUT_WITH_TAGS = """## Input Format:
--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

## Output Format:
A concise commit message describing the code changes as plain text, wrapped in <msg> </msg> tags. Nothing else after the </msg> tag."""

PROMPT_FORMAT_AND_OUTPUT_NO_TAGS = """## Input Format:
--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

## Output Format:
A concise commit message describing the code changes as plain text, no multiple sentences."""

PROMPT_EXAMPLES_WITH_TAGS = """Example output: <msg>Fix indefinite loading for users</msg>
Example output: <msg>feat(server): Add new API endpoint for user registration</msg>
Example output: <msg>Refactor data processing logic for improved readability</msg>
Example output: <msg>Add new feature { fileName/methodName/etc }</msg>"""

PROMPT_EXAMPLES_NO_TAGS = """Example output: Fix indefinite loading for users
Example output: feat(server): Add new API endpoint for user registration
Example output: Refactor data processing logic for improved readability
Example output: Add new feature { fileName/methodName/etc }"""

PROMPT_CLOSING_WITH_TAGS = """Given the following diff, write ONLY the appropriate one sentence condensed commit message, between tags, that follows the above instructions."""

PROMPT_CLOSING_NO_TAGS = """Given the following diff, write ONLY the appropriate one sentence condensed commit message text, following the above instructions."""


# ── Core pipeline ─────────────────────────────────────────────────────────────

def make_tasks(
    dataset_path: str,
    tasks_path: str,
    max_diff_length: int | None = MAX_DIFF_LENGTH,
    run_stats: bool = False,
    basic_system_prompt: bool = False,
    no_examples_prompt: bool = False,
    omit_message_tags: bool = False,
):
    dataset = load_from_jsonl(dataset_path)
    stats   = SkipStats()
    tasks   =[]

    if omit_message_tags:
        prompt_format_and_output = PROMPT_FORMAT_AND_OUTPUT_NO_TAGS
        prompt_examples = PROMPT_EXAMPLES_NO_TAGS
        prompt_closing = PROMPT_CLOSING_NO_TAGS
    else:
        prompt_format_and_output = PROMPT_FORMAT_AND_OUTPUT_WITH_TAGS
        prompt_examples = PROMPT_EXAMPLES_WITH_TAGS
        prompt_closing = PROMPT_CLOSING_WITH_TAGS

    for instance in dataset:
        original_diff  = instance.get("diff", "")
        filtered_diff  = filter_diff(original_diff)

        owner      = instance.get("owner",      "unknown")
        repo       = instance.get("repo",       "unknown")
        commit_sha = instance.get("commit_sha", "unknown")[:7]
        task_id    = f"{owner}_{repo}_{commit_sha}"

        if not filtered_diff:
            stats.record("no code files in diff", task_id)
            continue

        diff_length = len(filtered_diff)

        if max_diff_length is not None and diff_length > max_diff_length:
            stats.record(f"diff too long (>{max_diff_length})", task_id)
            continue

        raw_label = instance.get("message", "")

        after_clean     = clean_label(raw_label)
        after_normalize = normalize_label(after_clean)
        after_semantic  = semantic_clean(after_normalize)
        label           = after_semantic

        if not label:
            if not after_clean:
                reason = f"empty after clean_label       | was: {raw_label!r}"
            elif not after_normalize:
                reason = f"empty after normalize_label   | was: {after_clean!r}"
            else:
                reason = f"empty after semantic_clean    | was: {after_normalize!r}"
            stats.record(reason, task_id)
            continue

        # ── Assemble dynamic prompt sections based on flags
        body_parts =[prompt_format_and_output]
        if not no_examples_prompt:
            body_parts.append(prompt_examples)
        body_parts.append("") # Enforces an empty line before the closing statement
        body_parts.append(prompt_closing)
        
        instructions_body = "\n".join(body_parts)
        diff_block = f"--- START OF CODE DIFF ---\n{filtered_diff}\n--- END OF CODE DIFF ---"

        if basic_system_prompt:
            sys_content = PROMPT_ROLE
            usr_content = f"{instructions_body}\n\n{diff_block}"
        else:
            # Trailing newline matches the historical format's triple-quote wrap
            sys_content = f"{PROMPT_ROLE}\n{instructions_body}\n"
            usr_content = diff_block

        messages =[
            {"role": "system", "content": sys_content},
            {"role": "user",   "content": usr_content},
        ]

        # context_length: system prompt + user prompt (rough char count)
        context_length = len(sys_content) + len(usr_content)

        # Keep only code files — mirror the filter applied to the diff content
        raw_files = instance.get("files", [])
        code_files = [
            f for f in raw_files
            if pathlib.Path(f).suffix.lower() not in NON_CODE_EXTENSIONS
        ]

        tasks.append({
            "task_id":        task_id,
            "messages":       messages,
            "label":          label,
            "files":          code_files,
            "diff_length":    diff_length,
            "context_length": context_length,
        })

    # ── Write tasks.jsonl + labels.jsonl ─────────────────────────────────────
    labels_path = pathlib.Path(tasks_path).with_name("labels.jsonl")

    with open(tasks_path,   "w", encoding="utf-8") as f_tasks, \
         open(labels_path,  "w", encoding="utf-8") as f_labels:
        for task in tasks:
            f_tasks.write(json.dumps(task) + "\n")
            f_labels.write(json.dumps({
                "task_id": task["task_id"],
                "label":   task["label"],
            }) + "\n")

    print(f"Wrote {len(tasks)} tasks → {tasks_path}")
    stats.report()

    # ── Optional stats ────────────────────────────────────────────────────────
    if run_stats:
        _generate_stats(dataset, tasks, tasks_path)


def _generate_stats(
    dataset:    list[dict],
    tasks:      list[dict],
    tasks_path: str,
):
    """
    Produce:
      1. diff_length_histogram.png  – histogram of filtered diff lengths
      2. task_stats.json            – per-task label + length + language breakdown
    """
    out_dir = pathlib.Path(tasks_path).parent

    # Build a lookup: task_id → original instance
    id_to_instance: dict[str, dict] = {}
    for inst in dataset:
        owner      = inst.get("owner",      "unknown")
        repo       = inst.get("repo",       "unknown")
        commit_sha = inst.get("commit_sha", "unknown")[:7]
        id_to_instance[f"{owner}_{repo}_{commit_sha}"] = inst

    # ── 1. Histogram ──────────────────────────────────────────────────────────
    diff_lengths = [t["diff_length"] for t in tasks]

    if diff_lengths:
        if not _HAS_MPL:
            print("Warning: matplotlib not installed — skipping histogram. "
                  "Run `pip install matplotlib` to enable it.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(diff_lengths, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Filtered diff length (chars)", fontsize=12)
            ax.set_ylabel("Number of tasks",              fontsize=12)
            ax.set_title("Distribution of diff lengths across tasks", fontsize=14)
            ax.yaxis.grid(True, linestyle="--", alpha=0.6)
            ax.set_axisbelow(True)

            # Annotate with basic stats
            import statistics
            med = statistics.median(diff_lengths)
            mn  = statistics.mean(diff_lengths)
            ax.axvline(med, color="tomato",     linestyle="--", linewidth=1.2, label=f"Median {med:,.0f}")
            ax.axvline(mn,  color="darkorange", linestyle=":",  linewidth=1.2, label=f"Mean   {mn:,.0f}")
            ax.legend(fontsize=10)

            hist_path = out_dir / "diff_length_histogram.png"
            fig.tight_layout()
            fig.savefig(hist_path, dpi=150)
            plt.close(fig)
            print(f"Saved histogram → {hist_path}")

    # ── 2. Per-task JSON ──────────────────────────────────────────────────────
    task_stats: list[dict] =[]
    for task in tasks:
        tid  = task["task_id"]
        inst = id_to_instance.get(tid, {})

        raw_diff         = inst.get("diff", "")
        full_diff_length = len(raw_diff)
        per_file         = diff_length_per_file(raw_diff)
        languages, excl  = classify_diff_languages(raw_diff)

        task_stats.append({
            "task_id":                   tid,
            "original_label":            inst.get("message", ""),
            "cleaned_label":             task["label"],
            "full_diff_length":          full_diff_length,
            "filtered_diff_length":      task["diff_length"],
            "diff_length_per_file":      per_file,
            "programming_languages":     languages,
            "excluded_file_extensions":  excl,
        })

    stats_path = out_dir / "task_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(task_stats, f, indent=2, ensure_ascii=False)
    print(f"Saved per-task stats → {stats_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build tasks.jsonl from a commit-diff dataset."
    )
    parser.add_argument(
        "--dataset",
        default="ApacheCM/test.jsonl",
        help="Path to the input .jsonl dataset (default: ApacheCM/test.jsonl)",
    )
    parser.add_argument(
        "--tasks",
        default="tasks.jsonl",
        help="Path for the output tasks.jsonl (default: tasks.jsonl)",
    )
    parser.add_argument(
        "--max-diff-length",
        type=int,
        default=None,
        metavar="N",
        help="Skip commits whose filtered diff exceeds N characters (default: no limit)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Also generate diff_length_histogram.png and task_stats.json",
    )
    parser.add_argument(
        "--basic-system-prompt",
        action="store_true",
        help="Limit the system prompt to just the role definition, moving task instructions to the user prompt.",
    )
    parser.add_argument(
        "--no-examples-prompt",
        action="store_true",
        help="Remove the <msg> examples from the prompt instructions.",
    )
    parser.add_argument(
        "--omit-message-tags",
        action="store_true",
        help="Generate prompt instructions that request plain commit text without <msg> tags.",
    )
    args = parser.parse_args()

    make_tasks(
        dataset_path       = args.dataset,
        tasks_path         = args.tasks,
        max_diff_length    = args.max_diff_length,
        run_stats          = args.stats,
        basic_system_prompt= args.basic_system_prompt,
        no_examples_prompt = args.no_examples_prompt,
        omit_message_tags  = args.omit_message_tags,
    )