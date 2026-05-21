#!/usr/bin/env python3
"""
Main Evaluation Script for Commit Message Generation Pipelines
================================================================

Evaluates quality and speed across pipeline modes:
  - dllm_llm   (dLLM summaries → LLM CMG)        — primary experimental mode
  - llm_llm    (LLM summaries → LLM CMG)          — two-step baseline
  - llm_only   (LLM direct from diff → CMG)        — single-step baseline
  - dllm_only  (dLLM direct from diff → CMG)       — single-step dLLM

Research Questions:
  RQ-a: How can dLLMs improve a 2-step pipeline?
         Compares dllm_llm vs llm_llm (same 2-step architecture, different summariser)
  RQ-b: How does input representation affect CMG quality?
         Compares 2-step (summary-based) vs 1-step (raw diff) pipelines

Metrics:
  Quality:  BLEU-4, BLEU-NORM, BLEU-CODE, ROUGE-1/2/L, METEOR, CIDEr
  Speed:    total_wall_s, summary_wall_s, cmg_wall_s, dllm tok/step, llm tok/s
  Combined: QS-Score (quality × log2(1 + tok/s))

Usage:
  python analysis/100_main_eval.py \\
      --results outputs/bench_results/results_machine1.json.ckpt.jsonl \\
                outputs/bench_results/results_machine2.json.ckpt.jsonl \\
      --tasks build_tasks/tasks_5k.jsonl \\
      --max-tasks 1200 \\
      --output-dir analysis/eval_output

  python analysis/100_main_eval.py \\
      --results outputs/bench_results/results_machine1.json.ckpt.jsonl \\
      --max-tasks 250 --first-sentence
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
from typing import Optional

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
# Text normalisation helpers  (from 40_quality_eval.py)
# ═══════════════════════════════════════════════════════════════════════════════

_STEMMER = PorterStemmer()
_CAMEL_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE2 = re.compile(r"([a-z\d])([A-Z])")
_SNAKE_RE = re.compile(r"_")
_MSG_EXTRACT_RE = re.compile(r"<msg>(.*?)</?\s*msg>", re.IGNORECASE | re.DOTALL)
_MSG_TAG_RE = re.compile(r"</?<?\s*msg\s*>", re.IGNORECASE)
_NONALPHA_RE = re.compile(r"[^a-z0-9\s]")
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])(?:\s|$)')
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

# ── Label-cleaning: exact verbatim copies from build_tasks/build_tasks.py ─────
# Applied only to generated text; reference labels are already the output of
# this pipeline.  Keep in sync with build_tasks/build_tasks.py.

_MODULE_PREFIXES = sorted([
    'AIRFLOW', 'ARROW', 'AVRO', 'AWS', 'AZURE', 'BUILD', 'CASSANDRA',
    'CI', 'CLI', 'CLOUD', 'CONFIG', 'CORE', 'DEPLOY', 'DOCKER', 'DOCS',
    'DRIVER', 'ELASTIC', 'EXECUTOR', 'FLINK', 'GCP', 'HADOOP', 'HDFS',
    'HIVE', 'HTTP', 'INFRA', 'K8S', 'KAFKA', 'LOGGING', 'METRICS', 'ML',
    'MONITORING', 'PERF', 'PYTHON', 'REST', 'RPC', 'SCHEDULER', 'SECURITY',
    'SPARK', 'SQL', 'STREAMING', 'TEST', 'UI', 'WEB', 'WORKER', 'YARN',
], key=len, reverse=True)

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


def _bt_semantic_clean(label: str) -> str:
    """Verbatim copy of semantic_clean() from build_tasks/build_tasks.py."""
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


def _bt_clean_label(label: str) -> str:
    """Verbatim copy of clean_label() from build_tasks/build_tasks.py."""
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


def _bt_normalize_label(label: str) -> str:
    """Verbatim copy of normalize_label() from build_tasks/build_tasks.py."""
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


def clean_generated(text: str) -> str:
    """
    Put a model-generated commit message through the exact same pipeline that
    build_tasks/build_tasks.py used to produce the reference labels:

        strip_msg_tags  →  clean_label  →  normalize_label  →  semantic_clean

    The reference label stored in results is already the output of this
    pipeline, so only the generated text needs this treatment.
    """
    text = strip_msg_tags(text)
    text = _bt_clean_label(text)
    text = _bt_normalize_label(text)
    text = _bt_semantic_clean(text)
    return text


def extract_first_sentence(text: str) -> str:
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return text.strip()
    m = _SENTENCE_END_RE.search(first_line)
    if m:
        return first_line[: m.start() + 1].strip()
    return first_line


def strip_msg_tags(text: str) -> str:
    m = _MSG_EXTRACT_RE.search(text)
    if m:
        return m.group(1).strip().rstrip("<").strip()
    return _MSG_TAG_RE.sub("", text).strip()


def tokenize_simple(text: str) -> list[str]:
    return text.lower().split()


def tokenize_norm(text: str) -> list[str]:
    t = _NONALPHA_RE.sub(" ", text.lower())
    return t.split()


def split_identifiers(text: str) -> str:
    t = _CAMEL_RE1.sub(r"\1 \2", text)
    t = _CAMEL_RE2.sub(r"\1 \2", t)
    t = _SNAKE_RE.sub(" ", t)
    return t


def tokenize_code(text: str) -> list[str]:
    t = split_identifiers(text).lower()
    t = _NONALPHA_RE.sub(" ", t)
    return [_STEMMER.stem(tok) for tok in t.split()]


def extract_code_identifiers(task: dict) -> set[str]:
    identifiers: set[str] = set()
    for msg in task.get("messages", []):
        for m in _IDENT_RE.finditer(msg.get("content", "")):
            identifiers.add(m.group().lower())
    for fpath in task.get("files", []):
        stem = Path(fpath).stem
        identifiers.add(stem.lower())
        for part in split_identifiers(stem).lower().split():
            if len(part) > 2:
                identifiers.add(part)
    return identifiers


# ═══════════════════════════════════════════════════════════════════════════════
# Metric implementations
# ═══════════════════════════════════════════════════════════════════════════════

_SMOOTH = SmoothingFunction().method1


def compute_bleu4(hyp_tokens: list[str], ref_tokens: list[str]) -> float:
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_SMOOTH,
    )


def compute_meteor(hypothesis: str, reference: str) -> float:
    hyp_tok = tokenize_simple(hypothesis)
    ref_tok = tokenize_simple(reference)
    if not hyp_tok or not ref_tok:
        return 0.0
    return _nltk_meteor([ref_tok], hyp_tok)


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: scores[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}


# ── CIDEr ─────────────────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _build_doc_freq(refs_tok: list[list[str]], max_n: int = 4) -> Counter:
    df = Counter()
    for ref in refs_tok:
        seen = set()
        for n in range(1, max_n + 1):
            for ng in _ngrams(ref, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    return df


def _tfidf_vec(tokens: list[str], df: Counter, num_docs: int, max_n: int = 4) -> Counter:
    vec = Counter()
    length = len(tokens)
    for n in range(1, max_n + 1):
        for ng, count in _ngrams(tokens, n).items():
            tf = count / max(length - n + 1, 1)
            idf = math.log(max(1.0, num_docs) / max(1.0, df.get(ng, 0)))
            vec[ng] = tf * idf
    return vec


def _cos_sim(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compute_cider_corpus(hyps_tok: list[list[str]], refs_tok: list[list[str]]) -> list[float]:
    num_docs = len(refs_tok)
    df = _build_doc_freq(refs_tok)
    scores = []
    for h, r in zip(hyps_tok, refs_tok):
        hv = _tfidf_vec(h, df, num_docs)
        rv = _tfidf_vec(r, df, num_docs)
        scores.append(10.0 * _cos_sim(hv, rv))
    return scores


# ── BLEU-CODE tolerance ───────────────────────────────────────────────────────

def _filter_code_only_tokens(hyp: list[str], ref: list[str], idents: set[str]):
    if not idents:
        return hyp, ref
    ref_set = set(ref)
    stemmed = set()
    for i in idents:
        for p in tokenize_code(i):
            stemmed.add(p)
    extra = [t for t in hyp if t not in ref_set and t in stemmed]
    return hyp, ref + extra


# ── QS-Score ──────────────────────────────────────────────────────────────────

def compute_qs_score(quality: float, tokens_per_second: float) -> float:
    if tokens_per_second <= 0:
        return 0.0
    return quality * math.log2(1 + tokens_per_second)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_results_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_tasks(path: Path) -> dict[str, dict]:
    tasks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                tasks[obj["task_id"]] = obj
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# Core: per-sample + corpus evaluation for a single mode
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_mode(
    results: list[dict],
    tasks: Optional[dict[str, dict]] = None,
    mode_label: str = "",
    first_sentence: bool = False,
) -> tuple[list[dict], dict]:
    """Evaluate a single pipeline mode. Returns (per_sample_rows, corpus_summary)."""
    if not results:
        return [], {}

    all_hyp_simple, all_ref_simple = [], []
    all_hyp_norm, all_ref_norm = [], []
    all_hyp_code, all_ref_code = [], []
    per_sample = []
    rouge_accum = defaultdict(float)
    meteor_accum = 0.0

    for res in results:
        task_id = res["task_id"]
        raw_gen = clean_generated(res.get("generated", ""))
        if first_sentence:
            raw_gen = extract_first_sentence(raw_gen)
        raw_ref = res.get("label", "")  # already cleaned by build_tasks.py pipeline
        if not raw_ref:
            continue

        hyp_simple = tokenize_simple(raw_gen)
        ref_simple = tokenize_simple(raw_ref)
        hyp_norm = tokenize_norm(raw_gen)
        ref_norm = tokenize_norm(raw_ref)
        hyp_code = tokenize_code(raw_gen)
        ref_code = tokenize_code(raw_ref)

        code_idents = set()
        if tasks and task_id in tasks:
            code_idents = extract_code_identifiers(tasks[task_id])
        hyp_code_adj, ref_code_adj = _filter_code_only_tokens(hyp_code, ref_code, code_idents)

        bleu4 = compute_bleu4(hyp_simple, ref_simple)
        bleu_norm = compute_bleu4(hyp_norm, ref_norm)
        bleu_code = compute_bleu4(hyp_code_adj, ref_code_adj)
        rouge = compute_rouge(raw_gen, raw_ref)
        meteor = compute_meteor(raw_gen, raw_ref)

        # Speed stats from result structure
        timing = res.get("timing", {})
        total_wall = timing.get("total_wall_s", timing.get("task_wall_s", 0))
        summary_wall = timing.get("summary_wall_s", 0)
        cmg_wall = timing.get("cmg_wall_s", 0)
        n_files = res.get("n_files", len(res.get("file_summaries", [])))

        dllm_stats = res.get("dllm_stats") or {}
        llm_stats = res.get("llm_stats") or {}
        llm_summary_stats = res.get("llm_summary_stats") or {}

        # Tokens per second: prioritise total pipeline throughput
        cmg_tokens = res.get("token_counts", {}).get("cmg_tokens", 0)
        eps = 1e-9
        overall_tps = cmg_tokens / (total_wall + eps) if total_wall > 0 else 0

        row = {
            "mode": mode_label,
            "task_id": task_id,
            "n_files": n_files,
            # Quality
            "bleu4": round(bleu4, 4),
            "bleu_norm": round(bleu_norm, 4),
            "bleu_code": round(bleu_code, 4),
            "rouge1": round(rouge["rouge1"], 4),
            "rouge2": round(rouge["rouge2"], 4),
            "rougeL": round(rouge["rougeL"], 4),
            "meteor": round(meteor, 4),
            "cider": None,  # filled corpus-level
            # Speed
            "total_wall_s": round(total_wall, 4),
            "summary_wall_s": round(summary_wall, 4),
            "cmg_wall_s": round(cmg_wall, 4),
            "cmg_tokens": cmg_tokens,
            "overall_tps": round(overall_tps, 2),
            # dLLM-specific
            "dllm_total_steps": dllm_stats.get("total_steps"),
            "dllm_tokens_per_step": dllm_stats.get("tokens_per_step"),
            "dllm_tokens_per_second": dllm_stats.get("tokens_per_second"),
            # LLM generator speed
            "llm_gen_tps": llm_stats.get("tokens_per_second"),
            "llm_gen_prompt_tokens": llm_stats.get("prompt_tokens"),
            # LLM summariser speed (llm_llm mode)
            "llm_summary_tps": llm_summary_stats.get("tokens_per_second"),
            # QS scores (filled after CIDEr)
            "qs_bleu4": None,
            "qs_meteor": None,
            "qs_rougeL": None,
        }
        per_sample.append(row)

        all_hyp_simple.append(hyp_simple)
        all_ref_simple.append(ref_simple)
        all_hyp_norm.append(hyp_norm)
        all_ref_norm.append(ref_norm)
        all_hyp_code.append(hyp_code_adj)
        all_ref_code.append(ref_code_adj)

        for k, v in rouge.items():
            rouge_accum[k] += v
        meteor_accum += meteor

    n = len(per_sample)
    if n == 0:
        return [], {}

    # Corpus BLEU
    c_bleu4 = corpus_bleu([[r] for r in all_ref_simple], all_hyp_simple,
                          weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)
    c_bleu_norm = corpus_bleu([[r] for r in all_ref_norm], all_hyp_norm,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)
    c_bleu_code = corpus_bleu([[r] for r in all_ref_code], all_hyp_code,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)

    # CIDEr
    cider_scores = compute_cider_corpus(
        [tokenize_norm(strip_msg_tags(r.get("generated", ""))) for r in results[:n]],
        [tokenize_norm(r.get("label", "")) for r in results[:n]],
    )
    for i, s in enumerate(cider_scores):
        per_sample[i]["cider"] = round(s, 4)

    # QS scores
    for row in per_sample:
        tps = row["overall_tps"]
        row["qs_bleu4"] = round(compute_qs_score(row["bleu4"], tps), 4)
        row["qs_meteor"] = round(compute_qs_score(row["meteor"], tps), 4)
        row["qs_rougeL"] = round(compute_qs_score(row["rougeL"], tps), 4)

    avg = lambda key: round(sum(r[key] for r in per_sample) / n, 4)
    def avg_opt(key):
        vals = [r[key] for r in per_sample if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    def median_opt(key):
        vals = sorted(r[key] for r in per_sample if r[key] is not None)
        if not vals:
            return None
        mid = len(vals) // 2
        return round(vals[mid], 4) if len(vals) % 2 else round((vals[mid - 1] + vals[mid]) / 2, 4)

    summary = {
        "mode": mode_label,
        "num_samples": n,
        # Quality — corpus-level
        "corpus_bleu4": round(c_bleu4, 4),
        "corpus_bleu_norm": round(c_bleu_norm, 4),
        "corpus_bleu_code": round(c_bleu_code, 4),
        # Quality — avg
        "avg_bleu4": avg("bleu4"),
        "avg_bleu_norm": avg("bleu_norm"),
        "avg_bleu_code": avg("bleu_code"),
        "avg_rouge1": round(rouge_accum["rouge1"] / n, 4),
        "avg_rouge2": round(rouge_accum["rouge2"] / n, 4),
        "avg_rougeL": round(rouge_accum["rougeL"] / n, 4),
        "avg_meteor": round(meteor_accum / n, 4),
        "avg_cider": round(sum(cider_scores) / n, 4),
        # Speed — timing
        "avg_total_wall_s": avg("total_wall_s"),
        "median_total_wall_s": median_opt("total_wall_s"),
        "avg_summary_wall_s": avg("summary_wall_s"),
        "avg_cmg_wall_s": avg("cmg_wall_s"),
        # Speed — throughput
        "avg_overall_tps": avg("overall_tps"),
        # dLLM stats
        "avg_dllm_steps": avg_opt("dllm_total_steps"),
        "avg_dllm_tok_per_step": avg_opt("dllm_tokens_per_step"),
        "avg_dllm_tps": avg_opt("dllm_tokens_per_second"),
        # LLM stats
        "avg_llm_gen_tps": avg_opt("llm_gen_tps"),
        "avg_llm_summary_tps": avg_opt("llm_summary_tps"),
        # QS composites
        "avg_qs_bleu4": avg("qs_bleu4"),
        "avg_qs_meteor": avg("qs_meteor"),
        "avg_qs_rougeL": avg("qs_rougeL"),
    }
    return per_sample, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-mode comparison tables
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(summaries: list[dict], baseline_mode: str = "llm_llm") -> dict:
    """
    Build a comparison table: for each mode, compute deltas vs baseline.
    Returns dict with comparison data for RQ-a and RQ-b analysis.
    """
    baseline = None
    for s in summaries:
        if s["mode"] == baseline_mode:
            baseline = s
            break

    comparison = {"baseline_mode": baseline_mode, "modes": []}
    quality_keys = [
        "corpus_bleu4", "corpus_bleu_norm", "corpus_bleu_code",
        "avg_rouge1", "avg_rouge2", "avg_rougeL",
        "avg_meteor", "avg_cider",
    ]
    speed_keys = ["avg_total_wall_s", "avg_summary_wall_s", "avg_cmg_wall_s"]
    qs_keys = ["avg_qs_bleu4", "avg_qs_meteor", "avg_qs_rougeL"]

    for s in summaries:
        entry = {"mode": s["mode"], "num_samples": s["num_samples"]}

        # Absolute values
        for k in quality_keys + speed_keys + qs_keys:
            entry[k] = s.get(k)

        # Deltas vs baseline
        if baseline:
            for k in quality_keys:
                bv = baseline.get(k, 0) or 0
                sv = s.get(k, 0) or 0
                entry[f"delta_{k}"] = round(sv - bv, 4)
                if bv != 0:
                    entry[f"pct_{k}"] = round(100 * (sv - bv) / bv, 2)
                else:
                    entry[f"pct_{k}"] = None

            # Speedup (wall time: lower is better → speedup = baseline/current)
            for k in speed_keys:
                bv = baseline.get(k, 0) or 0
                sv = s.get(k, 0) or 0
                if sv > 0:
                    entry[f"speedup_{k}"] = round(bv / sv, 3)
                else:
                    entry[f"speedup_{k}"] = None

        comparison["modes"].append(entry)

    return comparison


def build_rq_analysis(summaries: list[dict], per_sample_all: list[dict]) -> dict:
    """Build focused RQ analysis blocks."""
    modes_present = {s["mode"] for s in summaries}
    analysis = {}

    # RQ-a: dllm_llm vs llm_llm (how dLLMs improve the 2-step pipeline)
    if "dllm_llm" in modes_present and "llm_llm" in modes_present:
        dllm = next(s for s in summaries if s["mode"] == "dllm_llm")
        llm = next(s for s in summaries if s["mode"] == "llm_llm")
        analysis["RQ_a_dllm_vs_llm_twostep"] = {
            "description": "How dLLMs improve the 2-step pipeline (dllm_llm vs llm_llm)",
            "dllm_llm": {k: dllm.get(k) for k in [
                "num_samples", "corpus_bleu4", "corpus_bleu_code", "avg_rougeL",
                "avg_meteor", "avg_cider", "avg_total_wall_s", "avg_summary_wall_s",
                "avg_dllm_tok_per_step", "avg_qs_meteor",
            ]},
            "llm_llm": {k: llm.get(k) for k in [
                "num_samples", "corpus_bleu4", "corpus_bleu_code", "avg_rougeL",
                "avg_meteor", "avg_cider", "avg_total_wall_s", "avg_summary_wall_s",
                "avg_llm_summary_tps", "avg_qs_meteor",
            ]},
            "summary_speedup": round(
                (llm["avg_summary_wall_s"] or 1) / max(dllm["avg_summary_wall_s"] or 1, 1e-9), 3
            ),
            "total_speedup": round(
                (llm["avg_total_wall_s"] or 1) / max(dllm["avg_total_wall_s"] or 1, 1e-9), 3
            ),
            "bleu4_delta": round((dllm["corpus_bleu4"] or 0) - (llm["corpus_bleu4"] or 0), 4),
            "meteor_delta": round((dllm["avg_meteor"] or 0) - (llm["avg_meteor"] or 0), 4),
        }

    # RQ-b: 2-step vs 1-step (how input representation affects quality)
    twostep_modes = [m for m in ["dllm_llm", "llm_llm"] if m in modes_present]
    onestep_modes = [m for m in ["llm_only", "dllm_only"] if m in modes_present]
    if twostep_modes and onestep_modes:
        rq_b = {"description": "How input (summary vs raw diff) affects CMG quality"}
        for m in twostep_modes + onestep_modes:
            s = next(x for x in summaries if x["mode"] == m)
            rq_b[m] = {k: s.get(k) for k in [
                "num_samples", "corpus_bleu4", "corpus_bleu_code", "avg_rougeL",
                "avg_meteor", "avg_cider", "avg_total_wall_s", "avg_qs_meteor",
            ]}
        analysis["RQ_b_input_representation"] = rq_b

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Console output helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(summaries: list[dict]):
    """Pretty-print a comparison table to console."""
    if not summaries:
        return

    quality_keys = [
        ("corpus_bleu4", "BLEU-4"),
        ("corpus_bleu_code", "BLEU-CODE"),
        ("avg_rougeL", "ROUGE-L"),
        ("avg_meteor", "METEOR"),
        ("avg_cider", "CIDEr"),
    ]
    speed_keys = [
        ("avg_total_wall_s", "Total(s)"),
        ("avg_summary_wall_s", "Summ(s)"),
        ("avg_cmg_wall_s", "CMG(s)"),
    ]
    qs_keys = [
        ("avg_qs_meteor", "QS-METEOR"),
    ]

    # Header
    modes = [s["mode"] for s in summaries]
    col_w = max(12, max(len(m) for m in modes) + 2)
    hdr = f"{'Metric':<18}" + "".join(f"{m:>{col_w}}" for m in modes)
    sep = "─" * len(hdr)

    print(f"\n{sep}")
    print(f"  QUALITY + SPEED COMPARISON  ({summaries[0]['num_samples']} tasks)")
    print(sep)
    print(hdr)
    print("─" * 18 + "─" * col_w * len(modes))

    for key, label in quality_keys:
        vals = "".join(
            f"{(s.get(key) or 0):>{col_w}.4f}" for s in summaries
        )
        print(f"{label:<18}{vals}")

    print("─" * 18 + "─" * col_w * len(modes))

    for key, label in speed_keys:
        vals = "".join(
            f"{(s.get(key) or 0):>{col_w}.2f}" for s in summaries
        )
        print(f"{label:<18}{vals}")

    print("─" * 18 + "─" * col_w * len(modes))

    for key, label in qs_keys:
        vals = "".join(
            f"{(s.get(key) or 0):>{col_w}.4f}" for s in summaries
        )
        print(f"{label:<18}{vals}")

    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate CMG pipeline results (quality + speed + cross-mode)."
    )
    p.add_argument(
        "--results", nargs="+", required=True,
        help="One or more .ckpt.jsonl or .jsonl result files to evaluate."
    )
    p.add_argument(
        "--tasks", default=None,
        help="Path to tasks JSONL (e.g. build_tasks/tasks_5k.jsonl). "
             "Enables code-identifier tolerance for BLEU-CODE."
    )
    p.add_argument(
        "--max-tasks", type=int, default=None,
        help="Evaluate only the first N tasks (by order in input). "
             "Useful when result files have different completion counts."
    )
    p.add_argument(
        "--task-ids", default=None,
        help="Path to a file containing task IDs (one per line) to restrict evaluation to."
    )
    p.add_argument(
        "--modes", default=None,
        help="Comma-separated modes to evaluate (default: all found in results)."
    )
    p.add_argument(
        "--baseline", default="llm_llm",
        help="Mode to use as baseline for delta comparisons (default: llm_llm)."
    )
    p.add_argument(
        "--output-dir", default="analysis/eval_output",
        help="Directory to write output files."
    )
    p.add_argument(
        "--first-sentence", action="store_true",
        help="Evaluate only the first sentence of each generated message."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure NLTK data
    for resource in ["wordnet", "punkt_tab"]:
        try:
            nltk.data.find(
                f"corpora/{resource}" if resource == "wordnet" else f"tokenizers/{resource}"
            )
        except LookupError:
            log.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

    # Load tasks
    tasks = None
    if args.tasks:
        log.info("Loading tasks from %s", args.tasks)
        tasks = load_tasks(Path(args.tasks))
        log.info("Loaded %d tasks", len(tasks))

    # Load task ID filter
    allowed_task_ids = None
    if args.task_ids:
        with open(args.task_ids, encoding="utf-8") as f:
            allowed_task_ids = set(line.strip() for line in f if line.strip())
        log.info("Loaded %d allowed task IDs from %s", len(allowed_task_ids), args.task_ids)

    # Load all results and group by mode
    all_results: dict[str, list[dict]] = defaultdict(list)
    for rpath in args.results:
        log.info("Loading results from %s", rpath)
        results = load_results_jsonl(Path(rpath))
        for r in results:
            mode = r.get("pipeline_mode", "unknown")
            if r.get("error"):
                continue
            all_results[mode].append(r)
    log.info("Loaded modes: %s", {m: len(v) for m, v in all_results.items()})

    # Filter modes if specified
    if args.modes:
        wanted = set(m.strip() for m in args.modes.split(","))
        all_results = {m: v for m, v in all_results.items() if m in wanted}

    if not all_results:
        log.warning("No results to evaluate. Exiting.")
        return

    # Determine common task set: intersect all modes' task IDs,
    # then apply max-tasks cutoff based on ordering from first file
    task_id_sets = [set(r["task_id"] for r in v) for v in all_results.values()]
    common_task_ids = task_id_sets[0]
    for s in task_id_sets[1:]:
        common_task_ids &= s

    if allowed_task_ids:
        common_task_ids &= allowed_task_ids

    # Determine ordering: use task order from the first result file's first mode
    first_mode = list(all_results.keys())[0]
    ordered_task_ids = []
    seen = set()
    for r in all_results[first_mode]:
        tid = r["task_id"]
        if tid in common_task_ids and tid not in seen:
            ordered_task_ids.append(tid)
            seen.add(tid)

    if args.max_tasks and args.max_tasks < len(ordered_task_ids):
        ordered_task_ids = ordered_task_ids[:args.max_tasks]

    eval_task_set = set(ordered_task_ids)
    log.info(
        "Evaluating %d common tasks across %d modes",
        len(eval_task_set), len(all_results),
    )

    # Filter results to common task set (preserving per-mode ordering)
    filtered: dict[str, list[dict]] = {}
    for mode, results in all_results.items():
        # Build lookup: task_id → result (keep first occurrence)
        lookup = {}
        for r in results:
            tid = r["task_id"]
            if tid in eval_task_set and tid not in lookup:
                lookup[tid] = r
        # Maintain consistent ordering
        filtered[mode] = [lookup[tid] for tid in ordered_task_ids if tid in lookup]

    # Evaluate each mode
    all_per_sample = []
    all_summaries = []

    for mode, results in sorted(filtered.items()):
        log.info("Evaluating mode '%s' (%d tasks)", mode, len(results))
        per_sample, summary = evaluate_mode(
            results, tasks, mode_label=mode,
            first_sentence=args.first_sentence,
        )
        all_per_sample.extend(per_sample)
        if summary:
            all_summaries.append(summary)

    # Cross-mode comparison
    comparison = build_comparison_table(all_summaries, baseline_mode=args.baseline)
    rq_analysis = build_rq_analysis(all_summaries, all_per_sample)

    # Console output
    print_summary_table(all_summaries)

    if rq_analysis.get("RQ_a_dllm_vs_llm_twostep"):
        rqa = rq_analysis["RQ_a_dllm_vs_llm_twostep"]
        print("─── RQ-a: dLLM vs LLM in 2-step pipeline ───")
        print(f"  BLEU-4 delta (dllm_llm - llm_llm): {rqa['bleu4_delta']:+.4f}")
        print(f"  METEOR delta:                       {rqa['meteor_delta']:+.4f}")
        print(f"  Summary speedup:                    {rqa['summary_speedup']:.3f}x")
        print(f"  Total pipeline speedup:             {rqa['total_speedup']:.3f}x")
        print()

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample CSV
    if all_per_sample:
        csv_path = out_dir / "per_sample_metrics.csv"
        fieldnames = list(all_per_sample[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_per_sample)
        log.info("Wrote per-sample CSV → %s", csv_path)

    # Per-sample JSONL
    if all_per_sample:
        jsonl_path = out_dir / "per_sample_metrics.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in all_per_sample:
                f.write(json.dumps(row) + "\n")
        log.info("Wrote per-sample JSONL → %s", jsonl_path)

    # Corpus summaries
    if all_summaries:
        with open(out_dir / "corpus_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        log.info("Wrote corpus summaries → %s", out_dir / "corpus_summary.json")

    # Comparison table
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    log.info("Wrote comparison → %s", out_dir / "comparison.json")

    # RQ analysis
    with open(out_dir / "rq_analysis.json", "w", encoding="utf-8") as f:
        json.dump(rq_analysis, f, indent=2)
    log.info("Wrote RQ analysis → %s", out_dir / "rq_analysis.json")

    log.info("Done.")


if __name__ == "__main__":
    main()
