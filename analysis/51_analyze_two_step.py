#!/usr/bin/env python3
"""
Two-Step CMG Pipeline Analysis
================================

Compares all four experimental conditions for the two-step CMG experiment:

  A: AR-Direct      — LLM generates CMG from full diff          (results_llm_baseline/)
  B: dLLM-Direct    — dLLM generates CMG from full diff         (fresh_ablation_results/<best_config>/)
  C: LLM-TwoStep    — LLM summarises each file → LLM CMG        (results_two_step_llm/)
  D: dLLM-TwoStep   — dLLM summarises each file → LLM CMG       (results_two_step_dllm/<config>/)

Generates:
  - Condition comparison bar chart (quality metrics)
  - Pipeline timing breakdown chart (summary vs CMG step times)
  - Per-file-count stratification plots (1 file / 2 files / 3+ files)
  - Statistical significance test (Wilcoxon signed-rank on BLEU-4)
  - Summary CSV for all conditions
  - Markdown report

Usage:
    python 51_analyze_two_step.py
    python 51_analyze_two_step.py --tasks build_tasks/tasks_tags.jsonl
    python 51_analyze_two_step.py --dllm-config bs32_sbs8_th0.4_smnt128_cmnt128
"""

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Reuse metric helpers from 50_analyze_ablation.py ────────────────────────

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    from nltk.stem import PorterStemmer
except ImportError:
    sys.exit("nltk is required: pip install nltk")

try:
    from rouge_score import rouge_scorer
except ImportError:
    sys.exit("rouge-score is required: pip install rouge-score")

for resource in ["wordnet", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource == "wordnet" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("nltk").setLevel(logging.WARNING)

# ─── Text helpers ─────────────────────────────────────────────────────────────

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
    return _NONALPHA_RE.sub(" ", text.lower()).split()


def split_identifiers(text):
    t = _CAMEL_RE1.sub(r"\1 \2", text)
    t = _CAMEL_RE2.sub(r"\1 \2", t)
    return _SNAKE_RE.sub(" ", t)


def tokenize_code(text):
    return tokenize_norm(split_identifiers(text))


def compute_bleu4(hyp, ref):
    if not hyp or not ref:
        return 0.0
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=_SMOOTH)


def compute_rouge(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


def compute_meteor(hypothesis, reference):
    try:
        hyp_tok = nltk.word_tokenize(hypothesis.lower())
        ref_tok = nltk.word_tokenize(reference.lower())
        if not hyp_tok or not ref_tok:
            return 0.0
        return float(_nltk_meteor([ref_tok], hyp_tok))
    except Exception:
        return 0.0


def _ngrams(tokens, n):
    from collections import Counter
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _build_doc_freq(refs_tok, max_n=4):
    from collections import Counter
    df = Counter()
    for ref in refs_tok:
        seen = set()
        for n in range(1, max_n + 1):
            for ng in _ngrams(ref, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    return df


def _tfidf_vec(tokens, df, num_docs, max_n=4):
    from collections import Counter
    vec = Counter()
    length = len(tokens)
    for n in range(1, max_n + 1):
        for ng, count in _ngrams(tokens, n).items():
            tf = count / max(length - n + 1, 1)
            idf = math.log(max(1.0, num_docs) / max(1.0, df.get(ng, 0)))
            vec[ng] = tf * idf
    return vec


def _cos_sim(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def compute_cider_corpus(hyps_tok, refs_tok):
    num_docs = len(refs_tok)
    df = _build_doc_freq(refs_tok)
    return [10.0 * _cos_sim(_tfidf_vec(h, df, num_docs), _tfidf_vec(r, df, num_docs))
            for h, r in zip(hyps_tok, refs_tok)]


# ─── Result loading ───────────────────────────────────────────────────────────

def load_results_from_dir(directory: Path) -> list[dict]:
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


# ─── Per-sample metrics ───────────────────────────────────────────────────────

def per_sample_scores(results: list[dict]) -> list[dict]:
    """Compute per-sample quality metrics for a list of result dicts."""
    rows = []
    for res in results:
        raw_gen = strip_msg_tags(res.get("generated", ""))
        raw_gen = extract_first_sentence(raw_gen)
        raw_ref = res.get("label", "")
        if not raw_ref:
            continue

        hyp = tokenize_simple(raw_gen)
        ref = tokenize_simple(raw_ref)
        rouge = compute_rouge(raw_gen, raw_ref)
        stats = res.get("stats", {})

        rows.append({
            "task_id": res.get("task_id", ""),
            "num_files": stats.get("num_files", 1),
            "bleu4": compute_bleu4(hyp, ref),
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "meteor": compute_meteor(raw_gen, raw_ref),
            "t_total": stats.get("t_total_seconds", stats.get("wall_seconds", 0.0)),
            "t_summary": stats.get("t_summary_seconds", 0.0),
            "t_cmg": stats.get("t_cmg_seconds", 0.0),
            "tps": stats.get("batch_tokens_per_second", stats.get("tokens_per_second", 0.0)),
            "generated_tokens": stats.get("generated_tokens", 0),
        })
    return rows


def aggregate_scores(sample_rows: list[dict]) -> dict:
    """Aggregate per-sample scores into corpus-level metrics."""
    if not sample_rows:
        return {}

    n = len(sample_rows)
    avg = lambda key: sum(r[key] for r in sample_rows) / n

    all_hyp = [tokenize_simple(strip_msg_tags(r.get("generated", ""))) for r in []]
    # For corpus BLEU we'd need to re-tokenize, but per-sample BLEU avg is fine for comparison
    return {
        "n": n,
        "avg_bleu4": round(avg("bleu4"), 4),
        "avg_rouge1": round(avg("rouge1"), 4),
        "avg_rouge2": round(avg("rouge2"), 4),
        "avg_rougeL": round(avg("rougeL"), 4),
        "avg_meteor": round(avg("meteor"), 4),
        "avg_t_total": round(avg("t_total"), 4),
        "avg_t_summary": round(avg("t_summary"), 4),
        "avg_t_cmg": round(avg("t_cmg"), 4),
        "avg_tps": round(avg("tps"), 2),
        "avg_generated_tokens": round(avg("generated_tokens"), 2),
    }


# ─── Statistical significance ─────────────────────────────────────────────────

def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.
    Returns {'statistic': ..., 'p_value': ..., 'significant': bool}.
    Requires scipy.
    """
    try:
        from scipy.stats import wilcoxon
        # Align by shorter list
        n = min(len(scores_a), len(scores_b))
        stat, p = wilcoxon(scores_a[:n], scores_b[:n], alternative="two-sided")
        return {"statistic": round(float(stat), 4), "p_value": round(float(p), 6),
                "significant": p < 0.05, "n": n}
    except ImportError:
        return {"error": "scipy not installed; run: pip install scipy"}
    except Exception as e:
        return {"error": str(e)}


# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path("f:/local-dllm-exploration")
OUTPUTS = ROOT / "outputs"
OUTPUT_DIR = OUTPUTS / "analysis_output"
PLOT_DIR = OUTPUT_DIR / "plots"
BASELINE_DIR = OUTPUTS / "results_llm_baseline"
ABLATION_DIR = OUTPUTS / "fresh_ablation_results"
TWO_STEP_DLLM_DIR = OUTPUTS / "results_two_step_dllm"
TWO_STEP_LLM_DIR = OUTPUTS / "results_two_step_llm"


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-step CMG experiment analysis")
    p.add_argument("--tasks", default="build_tasks/tasks_tags.jsonl",
                   help="Tasks JSONL for per-file-count stratification")
    p.add_argument("--dllm-config", default=None,
                   help="Specific dLLM-TwoStep config subdirectory name to highlight "
                        "(default: all configs in results_two_step_dllm/ are included)")
    p.add_argument("--best-dllm-direct", default="bs32_sbs8_th0.4_cacheTrue_batch1_mnt128",
                   help="Config name of the best dLLM-Direct baseline from ablation results")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # ── Load per-file-count mapping from tasks file ───────────────────────
    task_file_counts: dict[str, int] = {}
    tasks_path = Path(args.tasks)
    if tasks_path.exists():
        with open(tasks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                tid = task.get("task_id", "")
                nf = len(task.get("files", []))
                if tid:
                    task_file_counts[tid] = nf
        log.info(f"Loaded file-count mapping for {len(task_file_counts)} tasks.")
    else:
        log.warning(f"Tasks file not found: {tasks_path} — skipping per-file-count stratification.")

    # ── Load results for each condition ──────────────────────────────────
    conditions: dict[str, list[dict]] = {}

    # Condition A: AR-Direct
    if BASELINE_DIR.exists():
        results_a = load_results_from_dir(BASELINE_DIR)
        if results_a:
            conditions["A: AR-Direct"] = results_a
            log.info(f"Condition A: {len(results_a)} results loaded from {BASELINE_DIR}")

    # Condition B: dLLM-Direct (best config from ablation)
    best_dllm_dir = ABLATION_DIR / args.best_dllm_direct
    if best_dllm_dir.exists():
        results_b = load_results_from_dir(best_dllm_dir)
        if results_b:
            conditions[f"B: dLLM-Direct\n({args.best_dllm_direct[:30]})"] = results_b
            log.info(f"Condition B: {len(results_b)} results loaded from {best_dllm_dir}")
    else:
        log.warning(f"Condition B directory not found: {best_dllm_dir}")

    # Condition C: LLM-TwoStep
    if TWO_STEP_LLM_DIR.exists() and any(TWO_STEP_LLM_DIR.glob("*.json")):
        results_c = load_results_from_dir(TWO_STEP_LLM_DIR)
        if results_c:
            conditions["C: LLM-Summary\n→LLM-CMG"] = results_c
            log.info(f"Condition C: {len(results_c)} results loaded from {TWO_STEP_LLM_DIR}")

    # Condition D: dLLM-TwoStep (one or all configs)
    if TWO_STEP_DLLM_DIR.exists():
        if args.dllm_config:
            ts_dirs = [TWO_STEP_DLLM_DIR / args.dllm_config]
        else:
            ts_dirs = sorted(d for d in TWO_STEP_DLLM_DIR.iterdir() if d.is_dir())

        for ts_dir in ts_dirs:
            if not ts_dir.exists():
                log.warning(f"dLLM-TwoStep config not found: {ts_dir}")
                continue
            results_d = load_results_from_dir(ts_dir)
            if results_d:
                label = f"D: dLLM-Summary\n→LLM-CMG\n({ts_dir.name[:25]})"
                conditions[label] = results_d
                log.info(f"Condition D ({ts_dir.name}): {len(results_d)} results loaded")

    if not conditions:
        log.error("No results found for any condition. Run the eval scripts first.")
        return

    # ── Compute per-sample scores for all conditions ──────────────────────
    sample_scores: dict[str, list[dict]] = {}
    agg_scores: dict[str, dict] = {}

    for label, results in conditions.items():
        # Inject num_files from task metadata if not in stats
        for r in results:
            tid = r.get("task_id", "")
            if "num_files" not in r.get("stats", {}) and tid in task_file_counts:
                r.setdefault("stats", {})["num_files"] = task_file_counts[tid]

        samples = per_sample_scores(results)
        sample_scores[label] = samples
        agg_scores[label] = aggregate_scores(samples)
        log.info(f"{label!r}: n={len(samples)}, avg_meteor={agg_scores[label].get('avg_meteor', 'N/A')}")

    # ── Save summary CSV ──────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "two_step_summary.csv"
    fieldnames = ["condition", "n", "avg_bleu4", "avg_rouge1", "avg_rouge2", "avg_rougeL",
                  "avg_meteor", "avg_t_total", "avg_t_summary", "avg_t_cmg", "avg_tps",
                  "avg_generated_tokens"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label, agg in agg_scores.items():
            row = {"condition": label.replace("\n", " ")}
            row.update(agg)
            writer.writerow(row)
    log.info(f"Wrote summary CSV → {csv_path}")

    # ── Statistical significance tests ────────────────────────────────────
    sig_results = {}
    ref_label = "A: AR-Direct"
    if ref_label in sample_scores:
        ref_bleu = [r["bleu4"] for r in sample_scores[ref_label]]
        for label, samples in sample_scores.items():
            if label == ref_label:
                continue
            cand_bleu = [r["bleu4"] for r in samples]
            sig = wilcoxon_test(ref_bleu, cand_bleu)
            sig_results[label] = sig
            p_str = f"p={sig.get('p_value', '?')}"
            sig_str = "SIGNIFICANT" if sig.get("significant") else "not significant"
            log.info(f"Wilcoxon A vs {label!r}: {p_str} ({sig_str})")

    # ── PLOT 12: Condition comparison bar chart ───────────────────────────
    log.info("Plot 12: Condition comparison bar chart ...")
    metrics = ["avg_bleu4", "avg_rouge1", "avg_rougeL", "avg_meteor"]
    labels_m = ["BLEU-4", "ROUGE-1", "ROUGE-L", "METEOR"]

    cond_labels = list(agg_scores.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(cond_labels), 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(cond_labels)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (cond, agg) in enumerate(agg_scores.items()):
        vals = [agg.get(m, 0.0) for m in metrics]
        offset = (i - len(cond_labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cond.replace("\n", " "), color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_m)
    ax.set_ylabel("Score")
    ax.set_title("Two-Step CMG: Quality Comparison Across Conditions (A / B / C / D)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_twostep_condition_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ── PLOT 13: Pipeline timing breakdown ────────────────────────────────
    log.info("Plot 13: Pipeline timing breakdown ...")
    # Only two-step conditions have meaningful t_summary / t_cmg breakdown
    two_step_labels = [k for k in agg_scores if "TwoStep" in k or "Summary" in k or "C:" in k or "D:" in k]
    has_timing = any(agg_scores[k].get("avg_t_summary", 0) > 0 for k in two_step_labels if k in agg_scores)

    if has_timing:
        fig, ax = plt.subplots(figsize=(10, 5))
        ts_conds = [k for k in cond_labels if agg_scores[k].get("avg_t_total", 0) > 0]

        for i, cond in enumerate(ts_conds):
            agg = agg_scores[cond]
            t_sum = agg.get("avg_t_summary", 0)
            t_cmg = agg.get("avg_t_cmg", 0)
            t_other = max(agg.get("avg_t_total", 0) - t_sum - t_cmg, 0)
            cond_short = cond.replace("\n", " ")
            ax.bar(i, t_sum, color="#4C72B0", label="Summary step" if i == 0 else "")
            ax.bar(i, t_cmg, bottom=t_sum, color="#55A868", label="CMG step" if i == 0 else "")
            if t_other > 0:
                ax.bar(i, t_other, bottom=t_sum + t_cmg, color="#C44E52",
                       label="Other" if i == 0 else "")
            ax.text(i, agg.get("avg_t_total", t_sum + t_cmg) + 0.01,
                    f"{agg.get('avg_t_total', t_sum + t_cmg):.2f}s",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(len(ts_conds)))
        ax.set_xticklabels([c.replace("\n", " ") for c in ts_conds], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Average seconds per task")
        ax.set_title("Pipeline Timing Breakdown per Task")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "13_twostep_timing_breakdown.png", bbox_inches="tight")
        plt.close(fig)
    else:
        log.info("No timing data available yet — Plot 13 skipped.")

    # ── PLOT 14: Per-file-count stratification ────────────────────────────
    if task_file_counts:
        log.info("Plot 14: Per-file-count stratification ...")
        file_count_groups = {1: "1 file", 2: "2 files", "3+": "3+ files"}

        def get_group(n):
            if n == 1:
                return 1
            if n == 2:
                return 2
            return "3+"

        # For each condition, bucket per-sample scores by file count
        stratified: dict[str, dict] = {}  # condition → group → list of meteor scores
        for cond, samples in sample_scores.items():
            stratified[cond] = defaultdict(list)
            for s in samples:
                nf = s.get("num_files", 1)
                g = get_group(nf)
                stratified[cond][g].append(s["meteor"])

        groups_order = [1, 2, "3+"]
        cond_list = list(stratified.keys())
        n_groups = len(groups_order)
        n_conds = len(cond_list)
        colors_strat = plt.cm.tab10(np.linspace(0, 0.9, n_conds))

        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), sharey=True)
        if n_groups == 1:
            axes = [axes]

        for gi, group in enumerate(groups_order):
            ax = axes[gi]
            for ci, cond in enumerate(cond_list):
                scores = stratified[cond].get(group, [])
                if not scores:
                    continue
                avg_m = sum(scores) / len(scores)
                ax.bar(ci, avg_m, color=colors_strat[ci], alpha=0.85,
                       label=cond.replace("\n", " ") if gi == 0 else "")
                ax.text(ci, avg_m + 0.001, f"{avg_m:.3f}\n(n={len(scores)})",
                        ha="center", va="bottom", fontsize=7)

            ax.set_title(file_count_groups[group])
            ax.set_xticks(range(n_conds))
            ax.set_xticklabels([c.replace("\n", " ") for c in cond_list],
                               rotation=20, ha="right", fontsize=7)
            ax.set_ylabel("Avg METEOR" if gi == 0 else "")

        handles, labs = axes[0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="upper center", fontsize=8,
                   bbox_to_anchor=(0.5, 1.02), ncol=min(n_conds, 4))
        fig.suptitle("METEOR by Number of Changed Files per Commit", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(PLOT_DIR / "14_twostep_per_file_count.png", bbox_inches="tight")
        plt.close(fig)
    else:
        log.info("No task file-count data — Plot 14 skipped.")

    # ── PLOT 15: Quality vs End-to-End Speed scatter ──────────────────────
    log.info("Plot 15: Quality vs end-to-end speed ...")
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap_scatter = plt.cm.tab10(np.linspace(0, 0.9, len(cond_labels)))

    for i, (cond, agg) in enumerate(agg_scores.items()):
        tps = agg.get("avg_tps", 0)
        meteor = agg.get("avg_meteor", 0)
        if tps == 0 and meteor == 0:
            continue
        ax.scatter(tps, meteor, s=180, color=cmap_scatter[i], zorder=5,
                   edgecolors="black", linewidths=0.5,
                   label=cond.replace("\n", " "))
        ax.annotate(cond.replace("\n", " "), (tps, meteor),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.8)

    ax.set_xlabel("Avg End-to-End Throughput (tok/s, measured on CMG output)")
    ax.set_ylabel("Avg METEOR")
    ax.set_title("Quality vs Speed: All Conditions")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "15_twostep_quality_vs_speed.png", bbox_inches="tight")
    plt.close(fig)

    # ── Markdown report ───────────────────────────────────────────────────
    log.info("Writing two-step report ...")
    report = []
    report.append("# Two-Step CMG Experiment Report\n")
    report.append("## Experimental Conditions\n")
    report.append("| Condition | Step 1 | Step 2 | Results Dir |")
    report.append("| --- | --- | --- | --- |")
    report.append("| A: AR-Direct | — | LLM(full diff) | `results_llm_baseline/` |")
    report.append("| B: dLLM-Direct | — | dLLM(full diff) | `fresh_ablation_results/<best_config>/` |")
    report.append("| C: LLM-TwoStep | LLM per file (sequential) | LLM(summaries) | `results_two_step_llm/` |")
    report.append("| D: dLLM-TwoStep | dLLM per file (batch≤4) | LLM(summaries) | `results_two_step_dllm/<config>/` |")
    report.append("")

    report.append("## Hypotheses\n")
    report.append("| # | Hypothesis | Test |")
    report.append("| --- | --- | --- |")
    report.append("| H1 | Two-step improves quality | C/D vs A/B on BLEU-4, ROUGE-L, METEOR |")
    report.append("| H2 | dLLM summary is faster than LLM summary | t_summary(D) vs t_summary(C) |")
    report.append("| H3 | Two-step total pipeline is faster than AR-Direct | t_total(D) vs t_total(A) |")
    report.append("| H4 | Quality benefit scales with file count | Stratify H1 by #files |")
    report.append("| H5 | dLLM summary quality ≥ LLM summary quality | D vs C on final CMG metrics |")
    report.append("")

    report.append("## Aggregate Quality Results\n")
    headers = ["Condition", "N", "BLEU-4", "ROUGE-1", "ROUGE-L", "METEOR",
               "Avg t_total (s)", "Avg t_summary (s)", "Avg t_cmg (s)", "Tok/s"]
    report.append("| " + " | ".join(headers) + " |")
    report.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for cond, agg in agg_scores.items():
        cond_clean = cond.replace("\n", " ")
        report.append("| " + " | ".join([
            cond_clean,
            str(agg.get("n", "")),
            f"{agg.get('avg_bleu4', 0):.4f}",
            f"{agg.get('avg_rouge1', 0):.4f}",
            f"{agg.get('avg_rougeL', 0):.4f}",
            f"{agg.get('avg_meteor', 0):.4f}",
            f"{agg.get('avg_t_total', 0):.3f}",
            f"{agg.get('avg_t_summary', 0):.3f}",
            f"{agg.get('avg_t_cmg', 0):.3f}",
            f"{agg.get('avg_tps', 0):.1f}",
        ]) + " |")
    report.append("")

    report.append("## Statistical Significance (vs Condition A: AR-Direct)\n")
    if sig_results:
        report.append("Wilcoxon signed-rank test on per-sample BLEU-4 scores (two-sided, α=0.05).\n")
        report.append("| vs Condition | Statistic | p-value | Significant? |")
        report.append("| --- | --- | --- | --- |")
        for cond, res in sig_results.items():
            cond_clean = cond.replace("\n", " ")
            if "error" in res:
                report.append(f"| {cond_clean} | — | — | Error: {res['error']} |")
            else:
                sig_str = "YES ✓" if res.get("significant") else "No"
                report.append(f"| {cond_clean} | {res.get('statistic', '?')} | "
                               f"{res.get('p_value', '?')} | {sig_str} |")
        report.append("")
    else:
        report.append("_Condition A not available — cannot run significance tests._\n")

    report.append("## Plots\n")
    report.append("![Condition Comparison](plots/12_twostep_condition_comparison.png)\n")
    report.append("![Timing Breakdown](plots/13_twostep_timing_breakdown.png)\n")
    report.append("![Per-File-Count Stratification](plots/14_twostep_per_file_count.png)\n")
    report.append("![Quality vs Speed](plots/15_twostep_quality_vs_speed.png)\n")

    report.append("## H1 — Does two-step improve quality?\n")
    report.append("Compare Conditions C and D vs A and B in the table above.\n")
    if sig_results:
        better = {k: v for k, v in sig_results.items() if v.get("significant")}
        if better:
            report.append(f"**Significant differences found** for: {', '.join(better.keys())}\n")
        else:
            report.append("**No statistically significant quality difference** detected at α=0.05.\n")

    report.append("## H2 — Is dLLM summarisation faster than LLM?\n")
    c_sum = agg_scores.get("C: LLM-Summary\n→LLM-CMG", {}).get("avg_t_summary", None)
    d_keys = [k for k in agg_scores if "D:" in k]
    d_sum = agg_scores[d_keys[0]].get("avg_t_summary", None) if d_keys else None
    if c_sum and d_sum:
        speedup = c_sum / d_sum if d_sum > 0 else None
        report.append(f"- Avg summary time (C): {c_sum:.3f}s\n")
        report.append(f"- Avg summary time (D): {d_sum:.3f}s\n")
        if speedup:
            report.append(f"- **Speedup**: {speedup:.2f}× (D vs C in summary step)\n")
    else:
        report.append("_Timing data not yet available (run evaluation scripts first)._\n")

    report.append("## H3 — Is total two-step pipeline faster than AR-Direct?\n")
    a_t = agg_scores.get("A: AR-Direct", {}).get("avg_t_total", None)
    d_t = agg_scores[d_keys[0]].get("avg_t_total", None) if d_keys else None
    if a_t and d_t:
        su_total = a_t / d_t if d_t > 0 else None
        report.append(f"- Avg total time (A — AR-Direct): {a_t:.3f}s\n")
        report.append(f"- Avg total time (D — dLLM-TwoStep): {d_t:.3f}s\n")
        if su_total:
            report.append(f"- **End-to-end speedup**: {su_total:.2f}× (D vs A)\n")
    else:
        report.append("_Timing data not yet available._\n")

    report.append("---\n")
    report.append("*Generated by `51_analyze_two_step.py`*\n")

    report_path = OUTPUT_DIR / "two_step_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log.info(f"Wrote report → {report_path}")
    log.info("Done! Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
