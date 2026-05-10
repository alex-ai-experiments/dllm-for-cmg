"""
diff_utils.py — Utilities for splitting unified git diffs into per-file chunks.

Used by the two-step CMG pipeline scripts (25_eval_dllm_summary.py,
26_eval_llm_summary.py) to isolate each changed file's diff so it can be
summarised independently.
"""

import re
from typing import Optional

# Marker strings that wrap the diff in the task message format
_DIFF_START = "--- START OF CODE DIFF ---"
_DIFF_END = "--- END OF CODE DIFF ---"

# Matches "diff --git a/<path> b/<path>" — captures the b-side filename
_HEADER_RE = re.compile(r"^diff --git a/.+ b/(.+)$", re.MULTILINE)


def extract_diff_content(user_content: str) -> Optional[str]:
    """Return the raw diff text between the START / END markers, or None."""
    start = user_content.find(_DIFF_START)
    end = user_content.find(_DIFF_END)
    if start == -1 or end == -1 or end <= start:
        return None
    return user_content[start + len(_DIFF_START) : end].strip()


def split_diff_by_file(diff: str) -> list[tuple[str, str]]:
    """
    Split a unified git diff into (filename, file_diff) pairs.

    Each pair contains the filename (b-side path) and the full diff chunk
    for that file, starting from its "diff --git" header line.

    Returns an empty list if no "diff --git" headers are found.
    """
    matches = list(_HEADER_RE.finditer(diff))
    if not matches:
        return []

    result: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        filename = match.group(1).strip()
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(diff)
        file_diff = diff[start_pos:end_pos].strip()
        result.append((filename, file_diff))
    return result


def get_per_file_diffs(task: dict) -> list[tuple[str, str]]:
    """
    Extract per-file (filename, diff) pairs from a task dict.

    Looks for the user message, extracts the diff content, and splits by file.

    Falls back:
      - If splitting fails but a diff exists → returns [(\"(full diff)\", diff)]
      - If no diff markers exist → returns [(\"(full diff)\", user_content)]
      - If no user message exists → returns []
    """
    user_content = ""
    for msg in task.get("messages", []):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    if not user_content:
        return []

    diff = extract_diff_content(user_content)
    if diff is None:
        # No markers found — treat entire user content as the diff
        return [("(full diff)", user_content)]

    per_file = split_diff_by_file(diff)
    if not per_file:
        return [("(full diff)", diff)]
    return per_file


# ── Prompt builders ───────────────────────────────────────────────────────────

_SUMMARY_SYSTEM = (
    "You are a developer. Given the diff for a single file in a commit, "
    "write a concise technical summary (2-4 sentences) of what changed and why. "
    "Focus on the semantic meaning of the changes, not line-by-line description."
)

_CMG_SYSTEM = (
    "You are a developer, and your task is to write a concise commit message "
    "based on per-file summaries of the code changes in a commit.\n"
    "## Output Format:\n"
    "A concise commit message describing the code changes as plain text, "
    "wrapped in <msg> </msg> tags. Nothing else after the </msg> tag.\n"
    "Example output: <msg>Fix indefinite loading for users</msg>\n"
    "Example output: <msg>feat(server): Add new API endpoint for user registration</msg>"
)


def build_summary_messages(filename: str, file_diff: str, max_diff_chars: int | None = None) -> list[dict]:
    """
    Build the chat messages for summarising a single file diff.

    max_diff_chars: if set, truncate the diff content to this many characters
    before building the prompt. This caps prompt length variance across files,
    reducing padding waste in dLLM batches and lowering per-step forward-pass cost.
    The truncation point is chosen at the last newline within the limit to avoid
    cutting mid-line.
    """
    if max_diff_chars is not None and len(file_diff) > max_diff_chars:
        cut = file_diff.rfind("\n", 0, max_diff_chars)
        if cut == -1:
            cut = max_diff_chars
        file_diff = file_diff[:cut] + "\n... [diff truncated]"

    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"File: {filename}\n\n"
                "--- START OF FILE DIFF ---\n"
                f"{file_diff}\n"
                "--- END OF FILE DIFF ---\n\n"
                "Write a concise technical summary of the changes in this file."
            ),
        },
    ]


def build_cmg_messages(
    file_summaries: list[tuple[str, str]],
    original_system: Optional[str] = None,
) -> list[dict]:
    """
    Build the chat messages for the final CMG step.

    file_summaries: list of (filename, summary_text) pairs.
    original_system: if provided, used as the system prompt instead of the
                     default CMG system prompt (for ablation purposes).
    """
    summary_block = "\n\n".join(
        f"### {filename}\n{summary}" for filename, summary in file_summaries
    )
    user_content = (
        "The following are concise summaries of each file changed in a commit. "
        "Based on these summaries, write ONLY the appropriate one sentence "
        "commit message, between <msg> </msg> tags.\n\n"
        "## File Summaries:\n\n"
        f"{summary_block}"
    )
    system = original_system if original_system else _CMG_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
