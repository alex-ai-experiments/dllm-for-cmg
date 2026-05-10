#!/usr/bin/env python3
"""
Quality Metrics Evaluation Script for Commit Message Generation

Computes:
  - BLEU-4              (standard 4-gram BLEU)
  - BLEU-NORM           (BLEU-4 on lowercased / punctuation-stripped text)
  - BLEU-CODE           (custom: camelCase splitting + stemming + code-identifier tolerance)
  - ROUGE-1 / ROUGE-2 / ROUGE-L   (F1)
  - METEOR
  - CIDEr
  - QS-Score            (quality–speed composite for trade-off graphs)

Input:  one or more result directories / summary JSONL files
        optionally a tasks file (to extract code identifiers for BLEU-CODE)
Output: per-sample CSV, corpus-level summary JSON, optional console table

Usage examples:
    python 40_quality_eval.py -i results/
    python 40_quality_eval.py -i results/_results_summary.jsonl results_llm/_results_summary.jsonl
    python 40_quality_eval.py -i results/ -t build_tasks/tasks_tags.jsonl -o quality_metrics/
    python 40_quality_eval.py -i results/ results_llm/ extra/results/ --labels run1 run2 run3
"""

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Lazy / optional imports ──────────────────────────────────────────────────

try:
    import nltk
    from nltk.translate.bleu_score import (
        SmoothingFunction,
        corpus_bleu,
        sentence_bleu,
    )
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    from nltk.stem import PorterStemmer
except ImportError:
    sys.exit(
        "nltk is required.  Install with:\n  pip install nltk\n"
        "Then run once:  python -c \"import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')\""
    )

try:
    from rouge_score import rouge_scorer
except ImportError:
    sys.exit("rouge-score is required.  Install with:\n  pip install rouge-score")

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Text normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

_STEMMER = PorterStemmer()

# Matches camelCase / PascalCase boundaries  (e.g. CxfEndpoint → Cxf Endpoint)
_CAMEL_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE2 = re.compile(r"([a-z\d])([A-Z])")
# Matches snake_case underscores
_SNAKE_RE = re.compile(r"_")
# Strip the <msg>…</msg> wrapper produced by the model
# Extracts the inner content if tags are present; handles malformed tags like <</msg>
_MSG_EXTRACT_RE = re.compile(r"<msg>(.*?)</?\s*msg>", re.IGNORECASE | re.DOTALL)
_MSG_TAG_RE = re.compile(r"</?<?\s*msg\s*>", re.IGNORECASE)
# Non-alphanumeric (for punctuation stripping)
_NONALPHA_RE = re.compile(r"[^a-z0-9\s]")
# Sentence boundary: period/exclamation/question followed by whitespace or end,
# but not inside common abbreviations like "e.g." or version numbers like "v1.2"
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])(?:\s|$)')


def extract_first_sentence(text: str) -> str:
    """Return only the first sentence of *text*.

    If no sentence boundary is detected, return the whole text.
    Handles newlines as sentence breaks too.
    """
    # Treat newlines as sentence boundaries
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return text.strip()
    m = _SENTENCE_END_RE.search(first_line)
    if m:
        return first_line[: m.start() + 1].strip()
    return first_line


def strip_msg_tags(text: str) -> str:
    """Extract content from <msg>…</msg> wrapper; strips any residual tag fragments."""
    # First, try to extract the content between opening and closing tags
    m = _MSG_EXTRACT_RE.search(text)
    if m:
        # Strip any stray < left by malformed tags like <</msg>
        return m.group(1).strip().rstrip("<").strip()
    # Fallback: strip any remaining tag-like fragments (<msg>, </msg>, <</msg>, etc.)
    return _MSG_TAG_RE.sub("", text).strip()


def tokenize_simple(text: str) -> list[str]:
    """Lowercase + whitespace tokenize."""
    return text.lower().split()


def tokenize_norm(text: str) -> list[str]:
    """Lowercase, strip punctuation, whitespace tokenize."""
    t = text.lower()
    t = _NONALPHA_RE.sub(" ", t)
    return t.split()


def split_identifiers(text: str) -> str:
    """Split camelCase / PascalCase / snake_case tokens into sub-words."""
    t = _CAMEL_RE1.sub(r"\1 \2", text)
    t = _CAMEL_RE2.sub(r"\1 \2", t)
    t = _SNAKE_RE.sub(" ", t)
    return t


def tokenize_code(text: str) -> list[str]:
    """BLEU-CODE tokenizer: split identifiers, lowercase, stem."""
    t = split_identifiers(text)
    t = t.lower()
    t = _NONALPHA_RE.sub(" ", t)
    tokens = t.split()
    return [_STEMMER.stem(tok) for tok in tokens]


# ═══════════════════════════════════════════════════════════════════════════════
# Code-identifier extraction (for BLEU-CODE tolerance)
# ═══════════════════════════════════════════════════════════════════════════════

# Matches plausible identifiers in code (Java/Python/C++ etc.)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def extract_code_identifiers(task: dict) -> set[str]:
    """Pull class/method/variable names from the diff in the prompt."""
    identifiers: set[str] = set()

    # The diff is embedded in the last message's content
    messages = task.get("messages", [])
    for msg in messages:
        content = msg.get("content", "")
        for m in _IDENT_RE.finditer(content):
            identifiers.add(m.group().lower())

    # Also add file basenames (without extension)
    for fpath in task.get("files", []):
        stem = Path(fpath).stem
        identifiers.add(stem.lower())
        # split the stem too
        for part in split_identifiers(stem).lower().split():
            if len(part) > 2:
                identifiers.add(part)

    return identifiers


# ═══════════════════════════════════════════════════════════════════════════════
# Metric implementations
# ═══════════════════════════════════════════════════════════════════════════════

_SMOOTH = SmoothingFunction().method1


def compute_bleu4(hypothesis_tokens: list[str], reference_tokens: list[str]) -> float:
    if not hypothesis_tokens or not reference_tokens:
        return 0.0
    return sentence_bleu(
        [reference_tokens],
        hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_SMOOTH,
    )


def compute_meteor(hypothesis: str, reference: str) -> float:
    """METEOR with NLTK (handles stemming + synonyms internally)."""
    hyp_tokens = tokenize_simple(hypothesis)
    ref_tokens = tokenize_simple(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return _nltk_meteor([ref_tokens], hyp_tokens)


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ── CIDEr (corpus-level, simplified) ─────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _build_doc_freq(references_tokenized: list[list[str]], max_n: int = 4) -> Counter:
    df = Counter()
    for ref_tokens in references_tokenized:
        seen = set()
        for n in range(1, max_n + 1):
            for ng in _ngrams(ref_tokens, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    return df


def _tfidf_vec(tokens: list[str], df: Counter, num_docs: int, max_n: int = 4) -> Counter:
    vec = Counter()
    length = len(tokens)
    for n in range(1, max_n + 1):
        ngs = _ngrams(tokens, n)
        for ng, count in ngs.items():
            tf = count / max(length - n + 1, 1)
            idf = math.log(max(1.0, num_docs) / max(1.0, df.get(ng, 0)))
            vec[ng] = tf * idf
    return vec


def _cos_sim(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_cider_corpus(
    hypotheses_tokenized: list[list[str]],
    references_tokenized: list[list[str]],
) -> list[float]:
    """Compute per-sample CIDEr scores (requires corpus-level DF stats)."""
    num_docs = len(references_tokenized)
    df = _build_doc_freq(references_tokenized)

    scores = []
    for hyp_tok, ref_tok in zip(hypotheses_tokenized, references_tokenized):
        hyp_vec = _tfidf_vec(hyp_tok, df, num_docs)
        ref_vec = _tfidf_vec(ref_tok, df, num_docs)
        # CIDEr multiplies by 10 (convention from the paper)
        scores.append(10.0 * _cos_sim(hyp_vec, ref_vec))
    return scores


# ── BLEU-CODE with code-identifier tolerance ──────────────────────────────────

def _filter_code_only_tokens(
    hyp_tokens: list[str],
    ref_tokens: list[str],
    code_identifiers: set[str],
) -> tuple[list[str], list[str]]:
    """
    For tokens in hypothesis that are NOT in the reference but ARE derived
    from a code identifier found in the diff, we add them to the reference
    so they are not penalised.  This handles the case where the model names
    a class/method that the human didn't mention.
    """
    if not code_identifiers:
        return hyp_tokens, ref_tokens

    ref_set = set(ref_tokens)
    stemmed_identifiers = set()
    for ident in code_identifiers:
        for part in tokenize_code(ident):
            stemmed_identifiers.add(part)

    extra = []
    for tok in hyp_tokens:
        if tok not in ref_set and tok in stemmed_identifiers:
            extra.append(tok)

    return hyp_tokens, ref_tokens + extra


# ── QS-Score (Quality–Speed composite) ───────────────────────────────────────

def compute_qs_score(quality: float, tokens_per_second: float) -> float:
    """
    Quality–Speed Score.

    QS = quality × log2(1 + tokens_per_second)

    Rationale: quality matters most, but we reward throughput on a
    log-scale so doubling speed always gives the same additive bonus
    regardless of the baseline.  Useful for Pareto-front analysis.
    """
    if tokens_per_second <= 0:
        return 0.0
    return quality * math.log2(1 + tokens_per_second)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _rebuild_summary_from_files(directory: Path) -> list[dict]:
    """Read individual .json result files and write a _results_summary.jsonl."""
    results = []
    for jf in sorted(directory.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            with open(jf, encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            log.warning("Skipping %s: %s", jf, e)
    if results:
        summary_path = directory / "_results_summary.jsonl"
        with open(summary_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Created missing summary (%d results) → %s", len(results), summary_path)
    return results


def load_results(path: Path) -> list[dict]:
    """
    Load results from a JSONL file, a directory containing a
    _results_summary.jsonl, or a directory of experiment sub-directories
    each containing their own _results_summary.jsonl.

    If _results_summary.jsonl is missing but individual .json files exist,
    the summary is rebuilt automatically.
    """
    results = []
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    elif path.is_dir():
        # Direct summary in this directory?
        summary = path / "_results_summary.jsonl"
        if summary.exists():
            return load_results(summary)
        # Experiment sub-directories (e.g. bs32_sbs8_th0.9_…/)
        sub_summaries = sorted(path.rglob("_results_summary.jsonl"))
        if sub_summaries:
            for ss in sub_summaries:
                results.extend(load_results(ss))
            return results
        # Fall back to individual JSON files — and create the missing summary
        results = _rebuild_summary_from_files(path)
    else:
        log.warning("Path not found: %s", path)
    return results


def load_tasks(path: Path) -> dict[str, dict]:
    """Load task JSONL → dict keyed by task_id."""
    tasks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                tasks[obj["task_id"]] = obj
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_run(
    results: list[dict],
    tasks: dict[str, dict] | None = None,
    run_label: str = "",
    first_sentence: bool = False,
) -> tuple[list[dict], dict]:
    """
    Evaluate a single run.
    Returns (per_sample_rows, corpus_summary).
    """
    if not results:
        log.warning("No results to evaluate for run '%s'", run_label)
        return [], {}

    # Prepare tokenized lists for corpus-level metrics
    all_hyp_simple, all_ref_simple = [], []       # BLEU-4
    all_hyp_norm, all_ref_norm = [], []            # BLEU-NORM
    all_hyp_code, all_ref_code = [], []            # BLEU-CODE
    all_hyp_cider, all_ref_cider = [], []          # CIDEr (uses norm tokens)

    per_sample = []
    rouge_accum = defaultdict(float)
    meteor_accum = 0.0

    for res in results:
        task_id = res["task_id"]
        raw_gen = strip_msg_tags(res.get("generated", ""))
        if first_sentence:
            raw_gen = extract_first_sentence(raw_gen)
        raw_ref = res.get("label", "")

        if not raw_ref:
            log.warning("Skipping %s – no label", task_id)
            continue

        # ── Tokenize ──────────────────────────────────────────────────
        hyp_simple = tokenize_simple(raw_gen)
        ref_simple = tokenize_simple(raw_ref)

        hyp_norm = tokenize_norm(raw_gen)
        ref_norm = tokenize_norm(raw_ref)

        hyp_code = tokenize_code(raw_gen)
        ref_code = tokenize_code(raw_ref)

        # Code-identifier tolerance for BLEU-CODE
        code_idents: set[str] = set()
        if tasks and task_id in tasks:
            code_idents = extract_code_identifiers(tasks[task_id])
        hyp_code_adj, ref_code_adj = _filter_code_only_tokens(
            hyp_code, ref_code, code_idents
        )

        # ── Per-sample metrics ────────────────────────────────────────
        bleu4 = compute_bleu4(hyp_simple, ref_simple)
        bleu_norm = compute_bleu4(hyp_norm, ref_norm)
        bleu_code = compute_bleu4(hyp_code_adj, ref_code_adj)
        rouge = compute_rouge(raw_gen, raw_ref)
        meteor = compute_meteor(raw_gen, raw_ref)

        # Speed stats — normalise across the two schemas:
        #   dLLM new:  batch_tokens_per_second / effective_ms_per_token / batch_wall_seconds
        #              + tokens_per_step / batch_size / batch_total_steps
        #   AR / old:  tokens_per_second / ms_per_token / wall_seconds
        #              + prefill_seconds / time_to_first_token_ms / decode_tok_per_second
        stats = res.get("stats", {})
        tps  = stats.get("batch_tokens_per_second",
               stats.get("tokens_per_second", 0))
        mspt = stats.get("effective_ms_per_token",
               stats.get("ms_per_token", 0))
        wall = stats.get("batch_wall_seconds",
               stats.get("wall_seconds", 0))
        tokens_per_step        = stats.get("tokens_per_step", 1.0)
        batch_size             = stats.get("batch_size", 1)
        batch_total_steps      = stats.get("batch_total_steps",
                                  stats.get("generated_tokens", 0))  # AR: 1 step per token
        time_to_first_token_ms = stats.get("time_to_first_token_ms", None)
        prefill_seconds        = stats.get("prefill_seconds", None)
        decode_tok_per_second  = stats.get("decode_tok_per_second", None)

        row = {
            "run": run_label,
            "task_id": task_id,
            "bleu4": round(bleu4, 4),
            "bleu_norm": round(bleu_norm, 4),
            "bleu_code": round(bleu_code, 4),
            "rouge1": round(rouge["rouge1"], 4),
            "rouge2": round(rouge["rouge2"], 4),
            "rougeL": round(rouge["rougeL"], 4),
            "meteor": round(meteor, 4),
            # CIDEr will be filled corpus-level
            "cider": None,
            # Speed — shared fields
            "tokens_per_second": round(tps, 2),
            "ms_per_token": round(mspt, 3),
            "wall_seconds": round(wall, 4),
            "tokens_per_step": round(tokens_per_step, 4),
            "batch_size": batch_size,
            "batch_total_steps": batch_total_steps,
            # Speed — AR-only fields (None for dLLM)
            "time_to_first_token_ms": round(time_to_first_token_ms, 3) if time_to_first_token_ms is not None else None,
            "prefill_seconds": round(prefill_seconds, 4) if prefill_seconds is not None else None,
            "decode_tok_per_second": round(decode_tok_per_second, 2) if decode_tok_per_second is not None else None,
            # QS scores (filled after CIDEr)
            "qs_bleu4": None,
            "qs_meteor": None,
            "qs_rougeL": None,
        }
        per_sample.append(row)

        # Accumulate
        all_hyp_simple.append(hyp_simple)
        all_ref_simple.append(ref_simple)
        all_hyp_norm.append(hyp_norm)
        all_ref_norm.append(ref_norm)
        all_hyp_code.append(hyp_code_adj)
        all_ref_code.append(ref_code_adj)
        all_hyp_cider.append(hyp_norm)
        all_ref_cider.append(ref_norm)

        for k, v in rouge.items():
            rouge_accum[k] += v
        meteor_accum += meteor

    n = len(per_sample)
    if n == 0:
        return [], {}

    # ── Corpus-level BLEU ─────────────────────────────────────────────────
    corpus_bleu4 = corpus_bleu(
        [[r] for r in all_ref_simple],
        all_hyp_simple,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_SMOOTH,
    )
    corpus_bleu_norm = corpus_bleu(
        [[r] for r in all_ref_norm],
        all_hyp_norm,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_SMOOTH,
    )
    corpus_bleu_code = corpus_bleu(
        [[r] for r in all_ref_code],
        all_hyp_code,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_SMOOTH,
    )

    # ── Corpus-level CIDEr ────────────────────────────────────────────────
    cider_scores = compute_cider_corpus(all_hyp_cider, all_ref_cider)
    for i, score in enumerate(cider_scores):
        per_sample[i]["cider"] = round(score, 4)

    # ── QS-Scores ─────────────────────────────────────────────────────────
    for i, row in enumerate(per_sample):
        tps = row["tokens_per_second"]
        row["qs_bleu4"] = round(compute_qs_score(row["bleu4"], tps), 4)
        row["qs_meteor"] = round(compute_qs_score(row["meteor"], tps), 4)
        row["qs_rougeL"] = round(compute_qs_score(row["rougeL"], tps), 4)

    # ── Corpus summary ────────────────────────────────────────────────────
    avg = lambda key: round(sum(r[key] for r in per_sample) / n, 4)
    # avg for optional fields (skip None)
    def avg_opt(key):
        vals = [r[key] for r in per_sample if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "run": run_label,
        "num_samples": n,
        # Quality
        "corpus_bleu4": round(corpus_bleu4, 4),
        "corpus_bleu_norm": round(corpus_bleu_norm, 4),
        "corpus_bleu_code": round(corpus_bleu_code, 4),
        "avg_bleu4": avg("bleu4"),
        "avg_bleu_norm": avg("bleu_norm"),
        "avg_bleu_code": avg("bleu_code"),
        "avg_rouge1": round(rouge_accum["rouge1"] / n, 4),
        "avg_rouge2": round(rouge_accum["rouge2"] / n, 4),
        "avg_rougeL": round(rouge_accum["rougeL"] / n, 4),
        "avg_meteor": round(meteor_accum / n, 4),
        "avg_cider": round(sum(cider_scores) / n, 4),
        # Speed — shared
        "avg_tokens_per_second": avg("tokens_per_second"),
        "avg_ms_per_token": avg("ms_per_token"),
        "avg_tokens_per_step": avg("tokens_per_step"),
        "avg_batch_total_steps": avg("batch_total_steps"),
        # Speed — AR-only (None when not applicable)
        "avg_time_to_first_token_ms": avg_opt("time_to_first_token_ms"),
        "avg_prefill_seconds": avg_opt("prefill_seconds"),
        "avg_decode_tok_per_second": avg_opt("decode_tok_per_second"),
        # Quality–Speed composites
        "avg_qs_bleu4": avg("qs_bleu4"),
        "avg_qs_meteor": avg("qs_meteor"),
        "avg_qs_rougeL": avg("qs_rougeL"),
    }
    return per_sample, summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute quality metrics for commit-message generation results."
    )
    p.add_argument(
        "-i", "--inputs", nargs="+", required=True,
        help="Result directories or JSONL files (one per run).",
    )
    p.add_argument(
        "--labels", nargs="*", default=None,
        help="Human-readable label for each input (same order). "
             "Defaults to the directory/file name.",
    )
    p.add_argument(
        "-t", "--tasks", default=None,
        help="Path to tasks JSONL (e.g. build_tasks/tasks_tags.jsonl). "
             "Enables code-identifier tolerance in BLEU-CODE.",
    )
    p.add_argument(
        "-o", "--output-dir", default="quality_metrics",
        help="Directory to write output files.",
    )
    p.add_argument(
        "--first-sentence", action="store_true",
        help="Evaluate only the first sentence of each generated message. "
             "Useful when generations contain verbose explanations after a concise summary.",
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help="After evaluation, select the top-N tasks by --top-metric and "
             "write a separate .jsonl tasks file.",
    )
    p.add_argument(
        "--top-metric", default="bleu_code",
        help="Metric to rank tasks by when using --top-n. "
             "Must match a per-sample column name (default: bleu_code).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure NLTK data
    for resource in ["wordnet", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "wordnet" else f"tokenizers/{resource}")
        except LookupError:
            log.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

    # Load tasks if provided
    tasks = None
    if args.tasks:
        log.info("Loading tasks from %s", args.tasks)
        tasks = load_tasks(Path(args.tasks))
        log.info("Loaded %d tasks", len(tasks))

    # Prepare labels
    labels = args.labels or [Path(p).name for p in args.inputs]
    if len(labels) < len(args.inputs):
        labels.extend(Path(p).name for p in args.inputs[len(labels):])

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_per_sample = []
    all_summaries = []

    for inp, label in zip(args.inputs, labels):
        log.info("Evaluating run '%s' from %s", label, inp)
        results = load_results(Path(inp))
        log.info("  loaded %d results", len(results))

        per_sample, summary = evaluate_run(results, tasks, run_label=label,
                                              first_sentence=args.first_sentence)
        all_per_sample.extend(per_sample)
        if summary:
            all_summaries.append(summary)

            # Print summary
            log.info("  ── Corpus summary for '%s' ──", label)
            for k, v in summary.items():
                if k == "run":
                    continue
                log.info("    %-22s %s", k, v)

    # ── Write per-sample CSV ──────────────────────────────────────────────
    if all_per_sample:
        csv_path = out_dir / "per_sample_metrics.csv"
        fieldnames = list(all_per_sample[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_per_sample)
        log.info("Wrote per-sample metrics → %s", csv_path)

    # ── Write corpus summaries ────────────────────────────────────────────
    if all_summaries:
        summary_path = out_dir / "corpus_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        log.info("Wrote corpus summaries  → %s", summary_path)

    # ── Write per-sample JSONL (for downstream tools) ─────────────────────
    if all_per_sample:
        jsonl_path = out_dir / "per_sample_metrics.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in all_per_sample:
                f.write(json.dumps(row) + "\n")
        log.info("Wrote per-sample JSONL  → %s", jsonl_path)

    # ── Top-N task selection ────────────────────────────────────────────
    if args.top_n and all_per_sample:
        metric = args.top_metric
        # Validate metric exists
        if metric not in all_per_sample[0]:
            log.error("Metric '%s' not found in per-sample columns. Available: %s",
                      metric, list(all_per_sample[0].keys()))
        else:
            # Sort descending by chosen metric (treat None as -inf)
            ranked = sorted(all_per_sample,
                            key=lambda r: r.get(metric) if r.get(metric) is not None else -float("inf"),
                            reverse=True)
            top = ranked[: args.top_n]
            top_task_ids = {r["task_id"] for r in top}

            log.info("Top-%d tasks by %s:", args.top_n, metric)
            for r in top:
                log.info("  %-30s %s=%.4f", r["task_id"], metric, r.get(metric, 0))

            # If tasks file was provided, write the full task objects
            if tasks:
                top_tasks = [tasks[tid] for tid in top_task_ids if tid in tasks]
            else:
                # Fall back: write minimal stubs with task_id + score
                top_tasks = [{"task_id": r["task_id"], metric: r.get(metric)} for r in top]

            top_path = out_dir / f"top{args.top_n}_{metric}.jsonl"
            with open(top_path, "w", encoding="utf-8") as f:
                for t in top_tasks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            log.info("Wrote top-%d tasks → %s", args.top_n, top_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
