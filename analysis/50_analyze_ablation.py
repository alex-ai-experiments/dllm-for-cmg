#!/usr/bin/env python3
"""
Ablation Analysis & Reporting Script

Loads all fresh_ablation_results + results_llm_baseline, computes quality metrics
using 40_quality_eval.py logic, generates plots and tables, and writes a markdown report.

Usage:
    python 50_analyze_ablation.py
"""

import json
import math
import re
import os
import sys
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Reuse metric helpers from 40_quality_eval.py ────────────────────────────

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    from nltk.stem import PorterStemmer
except ImportError:
    sys.exit("nltk is required.")

try:
    from rouge_score import rouge_scorer
except ImportError:
    sys.exit("rouge-score is required.")

for resource in ["wordnet", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource == "wordnet" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Suppress noisy NLTK tokenizer messages
logging.getLogger("nltk").setLevel(logging.WARNING)

# ─── Text helpers (from 40_quality_eval.py) ───────────────────────────────────

_STEMMER = PorterStemmer()
_CAMEL_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE2 = re.compile(r"([a-z\d])([A-Z])")
_SNAKE_RE = re.compile(r"_")
_MSG_EXTRACT_RE = re.compile(r"<msg>(.*?)</?\s*msg>", re.IGNORECASE | re.DOTALL)
_MSG_TAG_RE = re.compile(r"</?<?\s*msg\s*>", re.IGNORECASE)
_NONALPHA_RE = re.compile(r"[^a-z0-9\s]")
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])(?:\s|$)')
_SMOOTH = SmoothingFunction().method1


def strip_msg_tags(text):
    m = _MSG_EXTRACT_RE.search(text)
    if m:
        return m.group(1).strip().rstrip("<").strip()
    return _MSG_TAG_RE.sub("", text).strip()


def extract_first_sentence(text):
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return text.strip()
    m = _SENTENCE_END_RE.search(first_line)
    if m:
        return first_line[: m.start() + 1].strip()
    return first_line


def tokenize_simple(text):
    return text.lower().split()


def tokenize_norm(text):
    t = _NONALPHA_RE.sub(" ", text.lower())
    return t.split()


def split_identifiers(text):
    t = _CAMEL_RE1.sub(r"\1 \2", text)
    t = _CAMEL_RE2.sub(r"\1 \2", t)
    t = _SNAKE_RE.sub(" ", t)
    return t


def tokenize_code(text):
    t = split_identifiers(text).lower()
    t = _NONALPHA_RE.sub(" ", t)
    return [_STEMMER.stem(tok) for tok in t.split()]


def compute_bleu4(hyp_tokens, ref_tokens):
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=_SMOOTH)


def compute_meteor(hypothesis, reference):
    hyp_tokens = tokenize_simple(hypothesis)
    ref_tokens = tokenize_simple(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return _nltk_meteor([ref_tokens], hyp_tokens)


def compute_rouge(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ─── CIDEr (corpus-level, simplified TF-IDF cosine) ──────────────────────────

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _build_doc_freq(references_tokenized, max_n=4):
    df = Counter()
    for ref_tokens in references_tokenized:
        seen = set()
        for n in range(1, max_n + 1):
            for ng in _ngrams(ref_tokens, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    return df


def _tfidf_vec(tokens, df, num_docs, max_n=4):
    vec = Counter()
    length = len(tokens)
    for n in range(1, max_n + 1):
        ngs = _ngrams(tokens, n)
        for ng, count in ngs.items():
            tf = count / max(length - n + 1, 1)
            idf = math.log(max(1.0, num_docs) / max(1.0, df.get(ng, 0)))
            vec[ng] = tf * idf
    return vec


def _cos_sim(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_cider_corpus(hypotheses_tokenized, references_tokenized):
    """Compute per-sample CIDEr scores (corpus-level DF stats)."""
    num_docs = len(references_tokenized)
    df = _build_doc_freq(references_tokenized)
    scores = []
    for hyp_tok, ref_tok in zip(hypotheses_tokenized, references_tokenized):
        hyp_vec = _tfidf_vec(hyp_tok, df, num_docs)
        ref_vec = _tfidf_vec(ref_tok, df, num_docs)
        scores.append(10.0 * _cos_sim(hyp_vec, ref_vec))
    return scores


# ─── Config parser ────────────────────────────────────────────────────────────

CONFIG_RE = re.compile(
    r"bs(?P<block_size>\d+)_sbs(?P<small_block_size>\d+)_th(?P<threshold>[\d.]+)"
    r"_cache(?P<cache>\w+)_batch(?P<batch_size>\d+)_mnt(?P<max_new_tokens>\d+)"
)


def parse_config_name(name):
    m = CONFIG_RE.match(name)
    if not m:
        return None
    return {
        "block_size": int(m.group("block_size")),
        "small_block_size": int(m.group("small_block_size")),
        "threshold": float(m.group("threshold")),
        "cache": m.group("cache") == "True",
        "batch_size": int(m.group("batch_size")),
        "max_new_tokens": int(m.group("max_new_tokens")),
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_results_from_dir(directory):
    """Load individual JSON result files from a directory."""
    results = []
    for jf in sorted(directory.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            with open(jf, encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


def compute_run_metrics(results, first_sentence=True):
    """Compute corpus-level quality + speed metrics for a list of result dicts."""
    all_hyp_simple, all_ref_simple = [], []
    all_hyp_norm, all_ref_norm = [], []
    all_hyp_code, all_ref_code = [], []

    bleu4_scores, bleu_norm_scores, bleu_code_scores = [], [], []
    meteor_scores = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    tps_list, mspt_list, steps_list, tpstep_list = [], [], [], []
    gen_tokens_list = []

    for res in results:
        raw_gen = strip_msg_tags(res.get("generated", ""))
        if first_sentence:
            raw_gen = extract_first_sentence(raw_gen)
        raw_ref = res.get("label", "")
        if not raw_ref:
            continue

        hyp_simple = tokenize_simple(raw_gen)
        ref_simple = tokenize_simple(raw_ref)
        hyp_norm = tokenize_norm(raw_gen)
        ref_norm = tokenize_norm(raw_ref)
        hyp_code = tokenize_code(raw_gen)
        ref_code = tokenize_code(raw_ref)

        bleu4 = compute_bleu4(hyp_simple, ref_simple)
        bleu_norm = compute_bleu4(hyp_norm, ref_norm)
        bleu_code = compute_bleu4(hyp_code, ref_code)
        rouge = compute_rouge(raw_gen, raw_ref)
        meteor = compute_meteor(raw_gen, raw_ref)

        bleu4_scores.append(bleu4)
        bleu_norm_scores.append(bleu_norm)
        bleu_code_scores.append(bleu_code)
        meteor_scores.append(meteor)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])

        all_hyp_simple.append(hyp_simple)
        all_ref_simple.append(ref_simple)
        all_hyp_norm.append(hyp_norm)
        all_ref_norm.append(ref_norm)
        all_hyp_code.append(hyp_code)
        all_ref_code.append(ref_code)

        stats = res.get("stats", {})
        tps = stats.get("batch_tokens_per_second", stats.get("tokens_per_second", 0))
        mspt = stats.get("effective_ms_per_token", stats.get("ms_per_token", 0))
        steps = stats.get("batch_total_steps", stats.get("generated_tokens", 0))
        tpstep = stats.get("tokens_per_step", 1.0)
        gen_tok = stats.get("generated_tokens", 0)

        tps_list.append(tps)
        mspt_list.append(mspt)
        steps_list.append(steps)
        tpstep_list.append(tpstep)
        gen_tokens_list.append(gen_tok)

    n = len(bleu4_scores)
    if n == 0:
        return None

    # Corpus BLEU
    c_bleu4 = corpus_bleu([[r] for r in all_ref_simple], all_hyp_simple,
                          weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)
    c_bleu_norm = corpus_bleu([[r] for r in all_ref_norm], all_hyp_norm,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)
    c_bleu_code = corpus_bleu([[r] for r in all_ref_code], all_hyp_code,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)

    # CIDEr (corpus-level)
    cider_scores = compute_cider_corpus(all_hyp_simple, all_ref_simple)

    avg = lambda lst: sum(lst) / len(lst)
    return {
        "num_samples": n,
        "corpus_bleu4": round(c_bleu4, 4),
        "corpus_bleu_norm": round(c_bleu_norm, 4),
        "corpus_bleu_code": round(c_bleu_code, 4),
        "avg_bleu4": round(avg(bleu4_scores), 4),
        "avg_bleu_norm": round(avg(bleu_norm_scores), 4),
        "avg_bleu_code": round(avg(bleu_code_scores), 4),
        "avg_rouge1": round(avg(rouge1_scores), 4),
        "avg_rouge2": round(avg(rouge2_scores), 4),
        "avg_rougeL": round(avg(rougeL_scores), 4),
        "avg_meteor": round(avg(meteor_scores), 4),
        "avg_cider": round(avg(cider_scores), 4),
        "avg_tokens_per_second": round(avg(tps_list), 2),
        "avg_ms_per_token": round(avg(mspt_list), 3),
        "avg_tokens_per_step": round(avg(tpstep_list), 4),
        "avg_diffusion_steps": round(avg(steps_list), 2),
        "avg_generated_tokens": round(avg(gen_tokens_list), 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

ROOT = Path("f:/local-dllm-exploration")
OUTPUTS = ROOT / "outputs"
ABLATION_DIR = OUTPUTS / "fresh_ablation_results"
NEW_ABLATION_DIR = OUTPUTS / "new_ablation_64_32"
BASELINE_DIR = OUTPUTS / "results_llm_baseline"
OUTPUT_DIR = OUTPUTS / "analysis_output"
PLOT_DIR = OUTPUT_DIR / "plots"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and compute metrics for all configs ──────────────────────
    log.info("Loading ablation results...")
    all_metrics = {}

    # Ablation configs
    ablation_dirs = sorted(ABLATION_DIR.iterdir())
    for i, d in enumerate(ablation_dirs):
        if not d.is_dir():
            continue
        name = d.name
        cfg = parse_config_name(name)
        if cfg is None:
            continue
        log.info(f"[{i+1}/{len(ablation_dirs)}] Computing metrics for {name}...")
        results = load_results_from_dir(d)
        metrics = compute_run_metrics(results, first_sentence=True)
        if metrics:
            metrics["config"] = name
            metrics.update(cfg)
            metrics["model_type"] = "dLLM"
            all_metrics[name] = metrics

    # New ablation (low mnt: 32, 64)
    if NEW_ABLATION_DIR.exists():
        new_ablation_dirs = sorted(NEW_ABLATION_DIR.iterdir())
        for i, d in enumerate(new_ablation_dirs):
            if not d.is_dir():
                continue
            name = d.name
            cfg = parse_config_name(name)
            if cfg is None:
                continue
            log.info(f"[new_ablation {i+1}/{len(new_ablation_dirs)}] Computing metrics for {name}...")
            results = load_results_from_dir(d)
            metrics = compute_run_metrics(results, first_sentence=True)
            if metrics:
                metrics["config"] = name
                metrics.update(cfg)
                metrics["model_type"] = "dLLM"
                all_metrics[name] = metrics

    # Baseline
    log.info("Computing metrics for AR baseline...")
    baseline_results = load_results_from_dir(BASELINE_DIR)
    baseline_metrics = compute_run_metrics(baseline_results, first_sentence=True)
    if baseline_metrics:
        baseline_metrics["config"] = "AR_baseline"
        baseline_metrics["block_size"] = None
        baseline_metrics["small_block_size"] = None
        baseline_metrics["threshold"] = None
        baseline_metrics["cache"] = None
        baseline_metrics["batch_size"] = 1
        baseline_metrics["max_new_tokens"] = 1024
        baseline_metrics["model_type"] = "AR"
        all_metrics["AR_baseline"] = baseline_metrics

    # ── Two-step pipeline results (Conditions C and D) ───────────────────
    # Condition C: LLM summary → LLM CMG  (results_two_step_llm/)
    # Condition D: dLLM summary → LLM CMG (results_two_step_dllm/<config>/)
    TWO_STEP_DLLM_DIR = OUTPUTS / "results_two_step_dllm"
    TWO_STEP_LLM_DIR = OUTPUTS / "results_two_step_llm"

    if TWO_STEP_DLLM_DIR.exists():
        ts_dllm_dirs = sorted(d for d in TWO_STEP_DLLM_DIR.iterdir() if d.is_dir())
        for ts_dir in ts_dllm_dirs:
            name = f"twostep_dllm_{ts_dir.name}"
            log.info(f"Loading two-step dLLM results: {ts_dir.name} ...")
            ts_results = load_results_from_dir(ts_dir)
            ts_metrics = compute_run_metrics(ts_results, first_sentence=True)
            if ts_metrics:
                ts_metrics["config"] = name
                ts_metrics["model_type"] = "dLLM_TwoStep"
                ts_metrics["block_size"] = None
                ts_metrics["small_block_size"] = None
                ts_metrics["threshold"] = None
                ts_metrics["cache"] = True
                ts_metrics["batch_size"] = 4
                ts_metrics["max_new_tokens"] = 128
                all_metrics[name] = ts_metrics

    if TWO_STEP_LLM_DIR.exists() and any(TWO_STEP_LLM_DIR.glob("*.json")):
        log.info("Loading two-step LLM results (Condition C) ...")
        ts_llm_results = load_results_from_dir(TWO_STEP_LLM_DIR)
        ts_llm_metrics = compute_run_metrics(ts_llm_results, first_sentence=True)
        if ts_llm_metrics:
            ts_llm_metrics["config"] = "twostep_llm"
            ts_llm_metrics["model_type"] = "LLM_TwoStep"
            ts_llm_metrics["block_size"] = None
            ts_llm_metrics["small_block_size"] = None
            ts_llm_metrics["threshold"] = None
            ts_llm_metrics["cache"] = None
            ts_llm_metrics["batch_size"] = 1
            ts_llm_metrics["max_new_tokens"] = 128
            all_metrics["twostep_llm"] = ts_llm_metrics

    log.info(f"Total configs with metrics: {len(all_metrics)}")

    # Save raw metrics
    metrics_path = OUTPUT_DIR / "all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Saved all metrics → {metrics_path}")

    # ── 2. Build DataFrame-like structure ────────────────────────────────
    rows = list(all_metrics.values())
    dllm_rows = [r for r in rows if r["model_type"] == "dLLM"]
    baseline = all_metrics.get("AR_baseline")

    # ── 3. Write summary CSV ─────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "ablation_summary.csv"
    fieldnames = [
        "config", "model_type", "block_size", "small_block_size", "threshold",
        "batch_size", "max_new_tokens", "num_samples",
        "corpus_bleu4", "corpus_bleu_norm", "corpus_bleu_code",
        "avg_bleu4", "avg_bleu_norm", "avg_bleu_code",
        "avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_meteor", "avg_cider",
        "avg_tokens_per_second", "avg_ms_per_token",
        "avg_tokens_per_step", "avg_diffusion_steps", "avg_generated_tokens",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Sort: baseline first, then dLLM sorted by config
        sorted_rows = sorted(rows, key=lambda r: (0 if r["model_type"] == "AR" else 1, r["config"]))
        writer.writerows(sorted_rows)
    log.info(f"Wrote CSV → {csv_path}")

    # ══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    quality_metrics = ["avg_bleu4", "avg_bleu_norm", "avg_bleu_code",
                       "avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_meteor", "avg_cider"]
    quality_labels = ["BLEU-4", "BLEU-NORM", "BLEU-CODE",
                      "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "CIDEr"]

    # ── PLOT 1: Threshold vs Quality (block_size sweep, mnt=128, batch=1) ─
    log.info("Generating Plot 1: Threshold vs Quality (block size sweep)...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    # Group: varying bs/sbs at mnt=128, batch=1
    block_configs = [
        ("bs=16, sbs=4", 16, 4),
        ("bs=16, sbs=8", 16, 8),
        ("bs=32, sbs=8", 32, 8),
        ("bs=32, sbs=16", 32, 16),
        ("bs=64, sbs=16", 64, 16),
        ("bs=64, sbs=32", 64, 32),
    ]

    for idx, (metric, label) in enumerate(zip(quality_metrics, quality_labels)):
        ax = axes[idx]
        for cfg_label, bs, sbs in block_configs:
            subset = [r for r in dllm_rows
                      if r["block_size"] == bs and r["small_block_size"] == sbs
                      and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
            if not subset:
                continue
            subset.sort(key=lambda r: r["threshold"])
            thresholds = [r["threshold"] for r in subset]
            values = [r[metric] for r in subset]
            ax.plot(thresholds, values, "o-", label=cfg_label, markersize=5)

        if baseline:
            ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=1.5,
                       label="AR baseline", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(label)
        ax.set_title(label)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    # Speed in last subplot
    ax = axes[8]
    for cfg_label, bs, sbs in block_configs:
        subset = [r for r in dllm_rows
                  if r["block_size"] == bs and r["small_block_size"] == sbs
                  and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in subset]
        values = [r["avg_tokens_per_second"] for r in subset]
        ax.plot(thresholds, values, "o-", label=cfg_label, markersize=5)
    if baseline:
        ax.axhline(baseline["avg_tokens_per_second"], color="red", linestyle="--",
                   linewidth=1.5, label="AR baseline", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Tokens/second")
    ax.set_title("Throughput")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Effect of Threshold on Quality & Speed (block size sweep, mnt=128, batch=1)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / "01_threshold_vs_quality_blocksize.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 2: Threshold vs Quality (max_new_tokens sweep, bs=32, sbs=8, batch=1) ─
    log.info("Generating Plot 2: Threshold vs Quality (max_new_tokens sweep)...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    mnt_values = [128, 256, 512, 1024]
    colors_mnt = plt.cm.viridis(np.linspace(0.2, 0.9, len(mnt_values)))

    for idx, (metric, label) in enumerate(zip(quality_metrics, quality_labels)):
        ax = axes[idx]
        for mnt, color in zip(mnt_values, colors_mnt):
            subset = [r for r in dllm_rows
                      if r["block_size"] == 32 and r["small_block_size"] == 8
                      and r["max_new_tokens"] == mnt and r["batch_size"] == 1]
            if not subset:
                continue
            subset.sort(key=lambda r: r["threshold"])
            thresholds = [r["threshold"] for r in subset]
            values = [r[metric] for r in subset]
            ax.plot(thresholds, values, "o-", label=f"mnt={mnt}", color=color, markersize=5)
        if baseline:
            ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=1.5,
                       label="AR baseline", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(label)
        ax.set_title(label)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    ax = axes[8]
    for mnt, color in zip(mnt_values, colors_mnt):
        subset = [r for r in dllm_rows
                  if r["block_size"] == 32 and r["small_block_size"] == 8
                  and r["max_new_tokens"] == mnt and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in subset]
        values = [r["avg_tokens_per_second"] for r in subset]
        ax.plot(thresholds, values, "o-", label=f"mnt={mnt}", color=color, markersize=5)
    if baseline:
        ax.axhline(baseline["avg_tokens_per_second"], color="red", linestyle="--",
                   linewidth=1.5, label="AR baseline", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Tokens/second")
    ax.set_title("Throughput")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Effect of Threshold on Quality & Speed (max_new_tokens sweep, bs=32, sbs=8, batch=1)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / "02_threshold_vs_quality_mnt.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 3: Batch size effect (batch=1 vs batch=4, bs=32, sbs=8) ────
    log.info("Generating Plot 3: Batch size effect...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    batch_mnt_combos = [
        (1, 256), (4, 256),
        (1, 512), (4, 512),
        (1, 1024), (4, 1024),
    ]
    markers = {1: "o", 4: "s"}
    linestyles = {1: "-", 4: "--"}

    for idx, (metric, label) in enumerate(zip(quality_metrics, quality_labels)):
        ax = axes[idx]
        for bsz, mnt in batch_mnt_combos:
            subset = [r for r in dllm_rows
                      if r["block_size"] == 32 and r["small_block_size"] == 8
                      and r["max_new_tokens"] == mnt and r["batch_size"] == bsz]
            if not subset:
                continue
            subset.sort(key=lambda r: r["threshold"])
            thresholds = [r["threshold"] for r in subset]
            values = [r[metric] for r in subset]
            ax.plot(thresholds, values, f"{markers[bsz]}{linestyles[bsz]}",
                    label=f"b={bsz}, mnt={mnt}", markersize=5)
        if baseline:
            ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=1.5,
                       label="AR baseline", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(label)
        ax.set_title(label)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    ax = axes[8]
    for bsz, mnt in batch_mnt_combos:
        subset = [r for r in dllm_rows
                  if r["block_size"] == 32 and r["small_block_size"] == 8
                  and r["max_new_tokens"] == mnt and r["batch_size"] == bsz]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in subset]
        values = [r["avg_tokens_per_second"] for r in subset]
        ax.plot(thresholds, values, f"{markers[bsz]}{linestyles[bsz]}",
                label=f"b={bsz}, mnt={mnt}", markersize=5)
    if baseline:
        ax.axhline(baseline["avg_tokens_per_second"], color="red", linestyle="--",
                   linewidth=1.5, label="AR baseline", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Tokens/second")
    ax.set_title("Throughput")
    ax.legend(fontsize=7, loc="best")

    fig.suptitle("Effect of Batch Size on Quality & Speed (bs=32, sbs=8)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / "03_batch_size_effect.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 4: Quality vs Speed Pareto front ────────────────────────────
    log.info("Generating Plot 4: Quality vs Speed Pareto...")
    quality_for_pareto = ["avg_meteor", "avg_rougeL", "avg_bleu_code", "avg_cider"]
    qlabels = ["METEOR", "ROUGE-L", "BLEU-CODE", "CIDEr"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (metric, qlabel) in enumerate(zip(quality_for_pareto, qlabels)):
        ax = axes[idx]

        # Color by threshold
        thresholds_all = sorted(set(r["threshold"] for r in dllm_rows))
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin=min(thresholds_all), vmax=max(thresholds_all))

        for r in dllm_rows:
            c = cmap(norm(r["threshold"]))
            ax.scatter(r["avg_tokens_per_second"], r[metric], c=[c],
                      s=30, alpha=0.6, edgecolors="none")

        if baseline:
            ax.scatter(baseline["avg_tokens_per_second"], baseline[metric],
                      c="red", s=120, marker="*", zorder=10, label="AR baseline",
                      edgecolors="black", linewidths=0.5)

        ax.set_xlabel("Tokens/second")
        ax.set_ylabel(qlabel)
        ax.set_title(f"{qlabel} vs Throughput")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label("Threshold")

        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Quality vs Speed Trade-off (colored by threshold)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / "04_quality_vs_speed_pareto.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 5: Tokens per step vs threshold ─────────────────────────────
    log.info("Generating Plot 5: Tokens per step...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg_label, bs, sbs in block_configs:
        subset = [r for r in dllm_rows
                  if r["block_size"] == bs and r["small_block_size"] == sbs
                  and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in subset]
        values = [r["avg_tokens_per_step"] for r in subset]
        ax.plot(thresholds, values, "o-", label=cfg_label, markersize=6)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="AR baseline (1.0)", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Avg Tokens per Step")
    ax.set_title("Tokens Accepted per Diffusion Step vs Threshold (mnt=128, batch=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_tokens_per_step.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 6: Avg generated tokens vs threshold ────────────────────────
    log.info("Generating Plot 6: Generated tokens length...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: block size sweep at mnt=128
    ax = axes[0]
    for cfg_label, bs, sbs in block_configs:
        subset = [r for r in dllm_rows
                  if r["block_size"] == bs and r["small_block_size"] == sbs
                  and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in subset]
        values = [r["avg_generated_tokens"] for r in subset]
        ax.plot(thresholds, values, "o-", label=cfg_label, markersize=5)
    if baseline:
        ax.axhline(baseline["avg_generated_tokens"], color="red", linestyle="--",
                   linewidth=1.5, label="AR baseline", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Avg Generated Tokens")
    ax.set_title("Block Size Sweep (mnt=128, batch=1)")
    ax.legend(fontsize=8)

    # Right: mnt sweep at bs=32, sbs=8, batch=1
    ax = axes[1]
    for mnt, color in zip(mnt_values, colors_mnt):
        subset = [r for r in dllm_rows
                  if r["block_size"] == 32 and r["small_block_size"] == 8
                  and r["max_new_tokens"] == mnt and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in subset]
        values = [r["avg_generated_tokens"] for r in subset]
        ax.plot(thresholds, values, "o-", label=f"mnt={mnt}", color=color, markersize=5)
    if baseline:
        ax.axhline(baseline["avg_generated_tokens"], color="red", linestyle="--",
                   linewidth=1.5, label="AR baseline", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Avg Generated Tokens")
    ax.set_title("Max New Tokens Sweep (bs=32, sbs=8, batch=1)")
    ax.legend(fontsize=8)

    fig.suptitle("Average Generated Token Length", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PLOT_DIR / "06_generated_tokens_length.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 7: Heatmap of METEOR for bs=32, sbs=8 (threshold x max_new_tokens) ─
    log.info("Generating Plot 7: Heatmaps...")
    for bsz in [1, 4]:
        subset = [r for r in dllm_rows
                  if r["block_size"] == 32 and r["small_block_size"] == 8
                  and r["batch_size"] == bsz]
        if not subset:
            continue

        thresholds_u = sorted(set(r["threshold"] for r in subset))
        mnts_u = sorted(set(r["max_new_tokens"] for r in subset))

        for metric, mlabel in [("avg_meteor", "METEOR"), ("avg_rougeL", "ROUGE-L"),
                                ("avg_cider", "CIDEr"), ("avg_tokens_per_second", "Tokens/sec")]:
            grid = np.full((len(mnts_u), len(thresholds_u)), np.nan)
            for r in subset:
                ti = thresholds_u.index(r["threshold"])
                mi = mnts_u.index(r["max_new_tokens"])
                grid[mi, ti] = r[metric]

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(grid, aspect="auto", cmap="YlOrRd" if "token" in metric.lower() else "YlGnBu")
            ax.set_xticks(range(len(thresholds_u)))
            ax.set_xticklabels([f"{t:.1f}" for t in thresholds_u])
            ax.set_yticks(range(len(mnts_u)))
            ax.set_yticklabels([str(m) for m in mnts_u])
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Max New Tokens")
            ax.set_title(f"{mlabel} Heatmap (bs=32, sbs=8, batch={bsz})")
            plt.colorbar(im, ax=ax, label=mlabel)

            # Annotate cells
            for mi in range(len(mnts_u)):
                for ti in range(len(thresholds_u)):
                    val = grid[mi, ti]
                    if not np.isnan(val):
                        fmt = f"{val:.1f}" if val > 10 else f"{val:.3f}"
                        ax.text(ti, mi, fmt, ha="center", va="center", fontsize=8,
                               color="white" if val > np.nanmean(grid) else "black")

            fig.tight_layout()
            safe_mlabel = mlabel.replace("/", "_").replace(" ", "_")
            fig.savefig(PLOT_DIR / f"07_heatmap_{safe_mlabel}_batch{bsz}.png", bbox_inches="tight")
            plt.close(fig)

    # ── PLOT 8: Bar chart comparing best dLLM config vs baseline ─────────
    log.info("Generating Plot 8: Best config comparison bar chart...")

    # Find best by avg_meteor among all dLLM configs
    best_meteor = max(dllm_rows, key=lambda r: r["avg_meteor"])
    # Also find best by speed (most tok/s)
    best_speed = max(dllm_rows, key=lambda r: r["avg_tokens_per_second"])

    compare_configs = {
        f"Best METEOR\n({best_meteor['config'][:30]}...)": best_meteor,
        f"Best Speed\n({best_speed['config'][:30]}...)": best_speed,
    }
    if baseline:
        compare_configs["AR Baseline"] = baseline

    compare_metrics = ["avg_meteor", "avg_rougeL", "avg_rouge1", "avg_bleu4", "avg_bleu_code", "avg_cider"]
    compare_labels = ["METEOR", "ROUGE-L", "ROUGE-1", "BLEU-4", "BLEU-CODE", "CIDEr"]

    x = np.arange(len(compare_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, cfg) in enumerate(compare_configs.items()):
        vals = [cfg[m] for m in compare_metrics]
        offset = (i - len(compare_configs)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(compare_labels)
    ax.set_ylabel("Score")
    ax.set_title("Best dLLM Configurations vs AR Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_best_vs_baseline_bar.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 9: Speedup ratio over AR baseline ──────────────────────────
    log.info("Generating Plot 9: Speedup ratio...")
    if baseline and baseline["avg_tokens_per_second"] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for cfg_label, bs, sbs in block_configs:
            subset = [r for r in dllm_rows
                      if r["block_size"] == bs and r["small_block_size"] == sbs
                      and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
            if not subset:
                continue
            subset.sort(key=lambda r: r["threshold"])
            thresholds = [r["threshold"] for r in subset]
            speedups = [r["avg_tokens_per_second"] / baseline["avg_tokens_per_second"] for r in subset]
            ax.plot(thresholds, speedups, "o-", label=cfg_label, markersize=6)

        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="AR baseline (1×)", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Speedup vs AR Baseline")
        ax.set_title("dLLM Speedup over AR Baseline (mnt=128, batch=1)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "09_speedup_ratio.png", bbox_inches="tight")
        plt.close(fig)

    # ── PLOT 10: Low-mnt ablation (mnt=32, mnt=64) ──────────────────────
    log.info("Generating Plot 10: Low-mnt ablation (mnt=32/64)...")
    low_mnt_rows_plot = [r for r in dllm_rows
                         if r["max_new_tokens"] in (32, 64) and r["batch_size"] == 1]
    if low_mnt_rows_plot:
        low_mnt_rows_plot.sort(key=lambda r: (r["max_new_tokens"], r["block_size"], r["small_block_size"]))

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = axes.flatten()

        plot_metrics = [
            ("avg_meteor", "METEOR"), ("avg_rougeL", "ROUGE-L"), ("avg_rouge1", "ROUGE-1"),
            ("avg_bleu4", "BLEU-4"), ("avg_bleu_code", "BLEU-CODE"), ("avg_cider", "CIDEr"),
            ("avg_tokens_per_second", "Tokens/sec"),
        ]
        config_labels = [f"bs{r['block_size']}_sbs{r['small_block_size']}_mnt{r['max_new_tokens']}"
                         for r in low_mnt_rows_plot]
        x_pos = np.arange(len(low_mnt_rows_plot))
        bar_colors = ["#4C72B0" if r["max_new_tokens"] == 32 else "#55A868" for r in low_mnt_rows_plot]

        for idx, (metric, mlabel) in enumerate(plot_metrics):
            ax = axes[idx]
            vals = [r[metric] for r in low_mnt_rows_plot]
            bars = ax.bar(x_pos, vals, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)

            if baseline:
                ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=1.5,
                           label="AR baseline", alpha=0.7)

            # Also show the best previous mnt=128 config for reference
            best_mnt128 = [r for r in dllm_rows
                           if r["max_new_tokens"] == 128 and r["batch_size"] == 1]
            if best_mnt128:
                best_ref = max(best_mnt128, key=lambda r: r["avg_meteor"])
                ax.axhline(best_ref[metric], color="purple", linestyle=":", linewidth=1.2,
                           label=f"Best mnt=128", alpha=0.7)

            for bar, val in zip(bars, vals):
                fmt = f"{val:.1f}" if val > 1 else f"{val:.3f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=7, rotation=45)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(config_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(mlabel)
            ax.set_title(mlabel)
            if idx == 0:
                # Legend for mnt colors
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor="#4C72B0", label="mnt=32"),
                    Patch(facecolor="#55A868", label="mnt=64"),
                ]
                if baseline:
                    from matplotlib.lines import Line2D
                    legend_elements.append(Line2D([0], [0], color="red", linestyle="--", label="AR baseline"))
                    legend_elements.append(Line2D([0], [0], color="purple", linestyle=":", label="Best mnt=128"))
                ax.legend(handles=legend_elements, fontsize=7, loc="best")

        fig.suptitle("Low Max-New-Tokens Ablation: mnt=32 & mnt=64 at Threshold=0.8",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(PLOT_DIR / "10_low_mnt_ablation.png", bbox_inches="tight")
        plt.close(fig)

    # ── PLOT 11: Generated tokens comparison (low mnt vs others) ─────────
    log.info("Generating Plot 11: Generated tokens comparison...")
    # Compare avg generated tokens across all mnt values at th=0.8, bs=32, sbs=8
    th08_bs32_sbs8 = [r for r in dllm_rows
                      if r["threshold"] == 0.8 and r["block_size"] == 32
                      and r["small_block_size"] == 8 and r["batch_size"] == 1]
    if th08_bs32_sbs8:
        th08_bs32_sbs8.sort(key=lambda r: r["max_new_tokens"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Avg generated tokens vs mnt
        ax = axes[0]
        mnts = [r["max_new_tokens"] for r in th08_bs32_sbs8]
        gen_toks = [r["avg_generated_tokens"] for r in th08_bs32_sbs8]
        ax.bar(range(len(mnts)), gen_toks, color="#4C72B0", alpha=0.85)
        ax.set_xticks(range(len(mnts)))
        ax.set_xticklabels([str(m) for m in mnts])
        ax.set_xlabel("Max New Tokens")
        ax.set_ylabel("Avg Generated Tokens")
        ax.set_title("Output Length vs MNT Cap (bs=32, sbs=8, th=0.8)")
        if baseline:
            ax.axhline(baseline["avg_generated_tokens"], color="red", linestyle="--",
                       linewidth=1.5, label="AR baseline", alpha=0.7)
            ax.legend()
        for i, (m, g) in enumerate(zip(mnts, gen_toks)):
            ax.text(i, g, f"{g:.1f}", ha="center", va="bottom", fontsize=9)

        # Right: Quality (METEOR) vs mnt
        ax = axes[1]
        meteors = [r["avg_meteor"] for r in th08_bs32_sbs8]
        ax.bar(range(len(mnts)), meteors, color="#55A868", alpha=0.85)
        ax.set_xticks(range(len(mnts)))
        ax.set_xticklabels([str(m) for m in mnts])
        ax.set_xlabel("Max New Tokens")
        ax.set_ylabel("METEOR")
        ax.set_title("METEOR vs MNT Cap (bs=32, sbs=8, th=0.8)")
        if baseline:
            ax.axhline(baseline["avg_meteor"], color="red", linestyle="--",
                       linewidth=1.5, label="AR baseline", alpha=0.7)
            ax.legend()
        for i, (m, v) in enumerate(zip(mnts, meteors)):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Effect of Max New Tokens on Output Length & Quality (th=0.8, bs=32, sbs=8)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(PLOT_DIR / "11_mnt_comparison_th08.png", bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # MARKDOWN REPORT
    # ══════════════════════════════════════════════════════════════════════
    log.info("Writing markdown report...")

    # Sort dLLM by avg_meteor descending
    dllm_sorted_meteor = sorted(dllm_rows, key=lambda r: r["avg_meteor"], reverse=True)
    # Sort by throughput
    dllm_sorted_speed = sorted(dllm_rows, key=lambda r: r["avg_tokens_per_second"], reverse=True)

    def md_table_row(cols):
        return "| " + " | ".join(str(c) for c in cols) + " |"

    def md_table_sep(n):
        return "| " + " | ".join(["---"] * n) + " |"

    # ── Helper: subset table for a specific sweep ────────────────────────
    def make_sweep_table(subset_rows, extra_cols=None):
        """Make a markdown table for a list of metric dicts."""
        if not subset_rows:
            return "_No data._\n"
        headers = ["Config", "Threshold", "METEOR", "ROUGE-L", "ROUGE-1",
                   "BLEU-4", "BLEU-CODE", "CIDEr", "Tok/s", "Tok/Step", "Avg Gen Tok"]
        if extra_cols:
            headers = extra_cols + headers[1:]  # replace Config with custom

        lines = [md_table_row(headers), md_table_sep(len(headers))]
        for r in subset_rows:
            row_data = [
                r.get("config", "")[:40],
                f"{r['threshold']:.1f}" if r.get("threshold") is not None else "—",
                f"{r['avg_meteor']:.4f}",
                f"{r['avg_rougeL']:.4f}",
                f"{r['avg_rouge1']:.4f}",
                f"{r['avg_bleu4']:.4f}",
                f"{r['avg_bleu_code']:.4f}",
                f"{r['avg_cider']:.4f}",
                f"{r['avg_tokens_per_second']:.1f}",
                f"{r['avg_tokens_per_step']:.2f}",
                f"{r['avg_generated_tokens']:.1f}",
            ]
            if extra_cols:
                row_data = row_data  # keep as is for now
            lines.append(md_table_row(row_data))
        return "\n".join(lines) + "\n"

    report = []
    report.append("# Ablation Study: Fast-dLLM for Commit Message Generation\n")
    report.append("## Overview\n")
    report.append(f"This report analyzes **{len(dllm_rows)} dLLM configurations** against an "
                  f"**autoregressive (AR) baseline** on a set of **1,000 commit message generation tasks**.\n")
    report.append("The dLLM model is `Efficient-Large-Model/Fast_dLLM_v2_1.5B` using the Fast-dLLM "
                  "speculative/masked diffusion decoding strategy. The AR baseline uses the same architecture "
                  "with standard autoregressive decoding.\n")
    report.append("### Hyperparameters Explored\n")
    report.append("| Parameter | Values |")
    report.append("| --- | --- |")
    report.append("| Block Size (bs) | 16, 32, 64 |")
    report.append("| Small Block Size (sbs) | 4, 8, 16, 32 |")
    report.append("| Confidence Threshold | 0.2, 0.4, 0.6, 0.8, 1.0 |")
    report.append("| Max New Tokens (mnt) | 32, 64, 128, 256, 512, 1024 |")
    report.append("| Batch Size | 1, 4 |")
    report.append("| Cache | True (all configs) |")
    report.append("")

    # ── Baseline section ─────────────────────────────────────────────────
    report.append("## AR Baseline Results\n")
    if baseline:
        report.append("| Metric | Value |")
        report.append("| --- | --- |")
        for k, label in [("avg_meteor", "METEOR"), ("avg_rougeL", "ROUGE-L"),
                         ("avg_rouge1", "ROUGE-1"), ("avg_rouge2", "ROUGE-2"),
                         ("avg_bleu4", "BLEU-4"), ("avg_bleu_norm", "BLEU-NORM"),
                         ("avg_bleu_code", "BLEU-CODE"), ("avg_cider", "CIDEr"),
                         ("avg_tokens_per_second", "Tokens/sec"),
                         ("avg_ms_per_token", "ms/token"),
                         ("avg_generated_tokens", "Avg generated tokens")]:
            report.append(f"| {label} | {baseline[k]} |")
        report.append("")

    # ── Top 10 by METEOR ─────────────────────────────────────────────────
    report.append("## Top 10 dLLM Configurations by METEOR\n")
    report.append(make_sweep_table(dllm_sorted_meteor[:10]))

    # ── Top 10 by Speed ──────────────────────────────────────────────────
    report.append("## Top 10 dLLM Configurations by Throughput\n")
    report.append(make_sweep_table(dllm_sorted_speed[:10]))

    # ── Block size sweep analysis ────────────────────────────────────────
    report.append("## Analysis: Block Size Sweep (mnt=128, batch=1)\n")
    report.append("This section isolates the effect of block size and small block size at a fixed "
                  "max generation length of 128 and batch size of 1.\n")
    report.append("![Threshold vs Quality — Block Size Sweep](plots/01_threshold_vs_quality_blocksize.png)\n")

    for cfg_label, bs, sbs in block_configs:
        subset = [r for r in dllm_rows
                  if r["block_size"] == bs and r["small_block_size"] == sbs
                  and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        report.append(f"### {cfg_label}\n")
        report.append(make_sweep_table(subset))

    # ── Max new tokens sweep ─────────────────────────────────────────────
    report.append("## Analysis: Max New Tokens Sweep (bs=32, sbs=8, batch=1)\n")
    report.append("This section examines how the maximum generation length affects quality and speed.\n")
    report.append("![Threshold vs Quality — MNT Sweep](plots/02_threshold_vs_quality_mnt.png)\n")

    for mnt in mnt_values:
        subset = [r for r in dllm_rows
                  if r["block_size"] == 32 and r["small_block_size"] == 8
                  and r["max_new_tokens"] == mnt and r["batch_size"] == 1]
        if not subset:
            continue
        subset.sort(key=lambda r: r["threshold"])
        report.append(f"### mnt={mnt}\n")
        report.append(make_sweep_table(subset))

    # ── Batch size effect ────────────────────────────────────────────────
    report.append("## Analysis: Batch Size Effect (bs=32, sbs=8)\n")
    report.append("Comparing batch_size=1 vs batch_size=4 across different max_new_tokens values.\n")
    report.append("![Batch Size Effect](plots/03_batch_size_effect.png)\n")

    for mnt in [256, 512, 1024]:
        report.append(f"### mnt={mnt}: batch=1 vs batch=4\n")
        for bsz in [1, 4]:
            subset = [r for r in dllm_rows
                      if r["block_size"] == 32 and r["small_block_size"] == 8
                      and r["max_new_tokens"] == mnt and r["batch_size"] == bsz]
            if not subset:
                continue
            subset.sort(key=lambda r: r["threshold"])
            report.append(f"**Batch size = {bsz}:**\n")
            report.append(make_sweep_table(subset))

    # ── Heatmaps ─────────────────────────────────────────────────────────
    report.append("## Heatmaps (bs=32, sbs=8)\n")
    report.append("These heatmaps show how METEOR, ROUGE-L, and throughput vary across threshold and "
                  "max_new_tokens for the bs=32, sbs=8 configuration.\n")
    for bsz in [1, 4]:
        report.append(f"### Batch size = {bsz}\n")
        report.append(f"![METEOR Heatmap batch={bsz}](plots/07_heatmap_METEOR_batch{bsz}.png)\n")
        report.append(f"![ROUGE-L Heatmap batch={bsz}](plots/07_heatmap_ROUGE-L_batch{bsz}.png)\n")
        report.append(f"![Throughput Heatmap batch={bsz}](plots/07_heatmap_Tokens_sec_batch{bsz}.png)\n")

    # ── New ablation: low mnt (32, 64) at th=0.8 ─────────────────────
    report.append("## Analysis: Low Max-New-Tokens (mnt=32, mnt=64) at Threshold=0.8\n")
    report.append("These experiments test very short generation limits with varying block/sub-block sizes. "
                  "Since commit messages are typically short (~5-20 tokens), constraining the output length "
                  "may reduce verbosity and improve precision-based metrics.\n")
    report.append("![Low MNT Ablation](plots/10_low_mnt_ablation.png)\n")
    report.append("![MNT Comparison](plots/11_mnt_comparison_th08.png)\n")

    low_mnt_rows = [r for r in dllm_rows
                    if r["max_new_tokens"] in (32, 64) and r["batch_size"] == 1]
    if low_mnt_rows:
        low_mnt_rows.sort(key=lambda r: (r["max_new_tokens"], r["block_size"], r["small_block_size"]))
        headers = ["Config", "bs", "sbs", "mnt", "METEOR", "ROUGE-L", "ROUGE-1",
                   "BLEU-4", "BLEU-CODE", "CIDEr", "Tok/s", "Tok/Step", "Avg Gen Tok"]
        report.append(md_table_row(headers))
        report.append(md_table_sep(len(headers)))
        for r in low_mnt_rows:
            report.append(md_table_row([
                r["config"][:40],
                r["block_size"], r["small_block_size"], r["max_new_tokens"],
                f"{r['avg_meteor']:.4f}", f"{r['avg_rougeL']:.4f}", f"{r['avg_rouge1']:.4f}",
                f"{r['avg_bleu4']:.4f}", f"{r['avg_bleu_code']:.4f}", f"{r['avg_cider']:.4f}",
                f"{r['avg_tokens_per_second']:.1f}",
                f"{r['avg_tokens_per_step']:.2f}",
                f"{r['avg_generated_tokens']:.1f}",
            ]))
        report.append("")

        # Compare best low-mnt vs AR baseline and best previous dLLM
        best_low_mnt = max(low_mnt_rows, key=lambda r: r["avg_meteor"])
        report.append("### Comparison: Best Low-MNT vs AR Baseline vs Best Overall dLLM\n")
        comp_headers = ["Model", "mnt", "METEOR", "ROUGE-L", "BLEU-4", "BLEU-CODE", "CIDEr", "Tok/s", "Avg Gen Tok"]
        report.append(md_table_row(comp_headers))
        report.append(md_table_sep(len(comp_headers)))
        for label, cfg, mnt_val in [
            ("AR Baseline", baseline, baseline["max_new_tokens"] if baseline else "—"),
            ("Best overall dLLM", best_meteor, best_meteor.get("max_new_tokens", "—")),
            ("Best low-mnt dLLM", best_low_mnt, best_low_mnt.get("max_new_tokens", "—")),
        ]:
            if cfg:
                report.append(md_table_row([
                    label, mnt_val,
                    f"{cfg['avg_meteor']:.4f}", f"{cfg['avg_rougeL']:.4f}",
                    f"{cfg['avg_bleu4']:.4f}", f"{cfg['avg_bleu_code']:.4f}",
                    f"{cfg['avg_cider']:.4f}",
                    f"{cfg['avg_tokens_per_second']:.1f}",
                    f"{cfg['avg_generated_tokens']:.1f}",
                ]))
        report.append("")

    # ── Quality vs Speed ─────────────────────────────────────────────────
    report.append("## Quality vs Speed Trade-off\n")
    report.append("![Quality vs Speed Pareto](plots/04_quality_vs_speed_pareto.png)\n")
    report.append("Each point is one dLLM configuration, colored by confidence threshold. "
                  "The red star marks the AR baseline. Lower thresholds (blue) accept more tokens "
                  "per step, increasing speed but potentially degrading quality.\n")

    # ── Tokens per step ──────────────────────────────────────────────────
    report.append("## Tokens Accepted per Diffusion Step\n")
    report.append("![Tokens per Step](plots/05_tokens_per_step.png)\n")
    report.append("Lower thresholds allow more tokens to be accepted per diffusion step (the core "
                  "speedup mechanism). The AR baseline always produces exactly 1 token per step.\n")

    # ── Generated token length ───────────────────────────────────────────
    report.append("## Average Generated Token Length\n")
    report.append("![Generated Tokens Length](plots/06_generated_tokens_length.png)\n")
    report.append("dLLM tends to generate more tokens than the AR baseline, especially at lower "
                  "thresholds (more aggressive acceptance). Higher max_new_tokens allows longer outputs.\n")

    # ── Speedup ratio ────────────────────────────────────────────────────
    report.append("## Speedup over AR Baseline\n")
    report.append("![Speedup Ratio](plots/09_speedup_ratio.png)\n")

    if baseline and baseline["avg_tokens_per_second"] > 0:
        # Calculate speedup stats
        speedups = [(r["config"], r["avg_tokens_per_second"] / baseline["avg_tokens_per_second"])
                    for r in dllm_rows]
        speedups.sort(key=lambda x: x[1], reverse=True)
        max_su = speedups[0]
        min_su = speedups[-1]
        avg_su = sum(s for _, s in speedups) / len(speedups)
        report.append(f"- **Max speedup**: {max_su[1]:.2f}× (`{max_su[0]}`)\n")
        report.append(f"- **Min speedup**: {min_su[1]:.2f}× (`{min_su[0]}`)\n")
        report.append(f"- **Average speedup**: {avg_su:.2f}× across all {len(speedups)} configs\n")

    # ── Best config comparison ───────────────────────────────────────────
    report.append("## Best Configurations vs AR Baseline\n")
    report.append("![Best vs Baseline](plots/08_best_vs_baseline_bar.png)\n")

    report.append("### Comparison Table\n")
    headers = ["Model", "METEOR", "ROUGE-L", "ROUGE-1", "BLEU-4", "BLEU-CODE", "CIDEr", "Tok/s", "Speedup"]
    report.append(md_table_row(headers))
    report.append(md_table_sep(len(headers)))

    for label, cfg in [("AR Baseline", baseline),
                       ("Best METEOR dLLM", best_meteor),
                       ("Best Speed dLLM", best_speed)]:
        if not cfg:
            continue
        su = cfg["avg_tokens_per_second"] / baseline["avg_tokens_per_second"] if baseline else 0
        report.append(md_table_row([
            label,
            f"{cfg['avg_meteor']:.4f}",
            f"{cfg['avg_rougeL']:.4f}",
            f"{cfg['avg_rouge1']:.4f}",
            f"{cfg['avg_bleu4']:.4f}",
            f"{cfg['avg_bleu_code']:.4f}",
            f"{cfg['avg_cider']:.4f}",
            f"{cfg['avg_tokens_per_second']:.1f}",
            f"{su:.2f}×",
        ]))
    report.append("")

    # ── Key Findings ─────────────────────────────────────────────────────
    report.append("## Key Findings\n")

    # 1. Threshold effect
    # Find quality at threshold extremes for a representative config
    rep_low = next((r for r in dllm_rows if r["config"] == "bs32_sbs8_th0.2_cacheTrue_batch1_mnt128"), None)
    rep_high = next((r for r in dllm_rows if r["config"] == "bs32_sbs8_th1.0_cacheTrue_batch1_mnt128"), None)

    report.append("### 1. Threshold is the Primary Quality-Speed Lever\n")
    if rep_low and rep_high:
        report.append(f"For the representative config (bs=32, sbs=8, mnt=128, batch=1):\n")
        report.append(f"- **Threshold=0.2**: METEOR={rep_low['avg_meteor']:.4f}, "
                      f"Tok/s={rep_low['avg_tokens_per_second']:.1f}, "
                      f"Tok/step={rep_low['avg_tokens_per_step']:.2f}\n")
        report.append(f"- **Threshold=1.0**: METEOR={rep_high['avg_meteor']:.4f}, "
                      f"Tok/s={rep_high['avg_tokens_per_second']:.1f}, "
                      f"Tok/step={rep_high['avg_tokens_per_step']:.2f}\n")
        report.append(f"- Higher threshold → stricter acceptance → fewer tokens per step → slower but "
                      f"potentially better quality.\n")

    # 2. Block size effect
    report.append("### 2. Block Size Impact\n")
    report.append("Larger block sizes (64 vs 16) generally allow higher throughput at equivalent "
                  "thresholds, as more positions are evaluated in parallel per diffusion step. "
                  "However, quality differences across block sizes are modest compared to the "
                  "threshold effect.\n")

    # 3. Max new tokens
    report.append("### 3. Max New Tokens and Output Length\n")
    report.append("Lower `max_new_tokens` (128) constrains the output and generally results in "
                  "faster generation and shorter, more focused outputs. Higher values (512, 1024) "
                  "allow the model to generate longer messages but may include repetitive or verbose "
                  "content that degrades BLEU/METEOR scores.\n")

    # 4. Batch size
    report.append("### 4. Batch Size Effect\n")
    batch1_speeds = [r["avg_tokens_per_second"] for r in dllm_rows if r["batch_size"] == 1]
    batch4_speeds = [r["avg_tokens_per_second"] for r in dllm_rows if r["batch_size"] == 4]
    if batch1_speeds and batch4_speeds:
        report.append(f"- Average throughput with batch=1: {sum(batch1_speeds)/len(batch1_speeds):.1f} tok/s\n")
        report.append(f"- Average throughput with batch=4: {sum(batch4_speeds)/len(batch4_speeds):.1f} tok/s\n")
    report.append("Batching increases hardware utilization. Quality metrics remain largely unchanged "
                  "between batch sizes since each sample is decoded independently.\n")

    # 5. vs baseline
    report.append("### 5. dLLM vs AR Baseline\n")
    if baseline:
        # Count how many dLLM configs beat baseline on each metric
        better_meteor = sum(1 for r in dllm_rows if r["avg_meteor"] > baseline["avg_meteor"])
        better_cider = sum(1 for r in dllm_rows if r["avg_cider"] > baseline["avg_cider"])
        faster = sum(1 for r in dllm_rows if r["avg_tokens_per_second"] > baseline["avg_tokens_per_second"])
        report.append(f"- **{better_meteor}/{len(dllm_rows)}** dLLM configs achieve higher METEOR than the AR baseline.\n")
        report.append(f"- **{better_cider}/{len(dllm_rows)}** dLLM configs achieve higher CIDEr than the AR baseline.\n")
        report.append(f"- **{faster}/{len(dllm_rows)}** dLLM configs are faster (higher tok/s) than the AR baseline.\n")
        report.append("The dLLM approach offers a clear **speed advantage** due to parallel token generation. "
                      "Quality is competitive with or exceeds the AR baseline for well-tuned threshold values.\n")

    report.append("---\n")
    report.append("*Report generated automatically by `50_analyze_ablation.py`.*\n")

    # Write report
    report_path = OUTPUT_DIR / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log.info(f"Wrote report → {report_path}")

    log.info("Done! All outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
