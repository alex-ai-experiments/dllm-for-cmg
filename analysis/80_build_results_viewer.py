#!/usr/bin/env python3
"""Build an interactive HTML results viewer with all ablation data embedded."""

import json
import glob
import os
import re
import html as html_mod
from pathlib import Path

FRESH_DIR = "outputs/fresh_ablation_results"
NEW_DIR = "outputs/new_ablation_64_32"
BASELINE_DIR = "outputs/results_llm_baseline"
METRICS_FILE = "outputs/analysis_output/all_metrics.json"
OUTPUT_HTML = "outputs/analysis_output/results_viewer.html"


def clean_generated(text: str) -> str:
    """Strip <msg> tags and whitespace from generated text."""
    text = re.sub(r"</?msg>", "", text).strip()
    return text


def parse_config_name(name: str):
    """Extract hyperparameters from config directory name."""
    m = re.match(
        r"bs(\d+)_sbs(\d+)_th([\d.]+)_cache(True|False)_batch(\d+)_mnt(\d+)", name
    )
    if m:
        return {
            "block_size": int(m.group(1)),
            "sub_block_size": int(m.group(2)),
            "threshold": float(m.group(3)),
            "cache": m.group(4) == "True",
            "batch_size": int(m.group(5)),
            "max_new_tokens": int(m.group(6)),
        }
    return {}


def load_summary_jsonl(path: str):
    """Load a _results_summary.jsonl and return list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    print("Loading aggregate metrics...")
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)
    print(f"  {len(all_metrics)} configs in aggregate metrics")

    # ── Collect per-sample data ──────────────────────────────────────────
    # We store: shared task list (task_id + label) once,
    # then per-config: list of {generated, tok_s, gen_tok, steps, tok_per_step}
    # indexed in same order as task list.

    # First, load baseline to establish task order
    print("Loading AR baseline samples...")
    bl_samples = load_summary_jsonl(
        os.path.join(BASELINE_DIR, "_results_summary.jsonl")
    )
    task_order = [s["task_id"] for s in bl_samples]
    task_id_to_idx = {tid: i for i, tid in enumerate(task_order)}

    # Shared task list
    tasks = []
    for s in bl_samples:
        tasks.append(
            {"id": s["task_id"], "label": s["label"], "prompt_tok": s["stats"].get("prompt_tokens", 0)}
        )

    # Per-config sample data
    configs_samples = {}

    # Baseline
    bl_per_sample = [None] * len(task_order)
    for s in bl_samples:
        idx = task_id_to_idx[s["task_id"]]
        bl_per_sample[idx] = {
            "g": clean_generated(s["generated"]),
            "ts": round(s["stats"]["tokens_per_second"], 2),
            "gt": s["stats"]["generated_tokens"],
            "st": s["stats"]["generated_tokens"],  # AR: steps = tokens
            "tps": 1.0,
        }
    configs_samples["AR_baseline"] = bl_per_sample

    # dLLM configs
    summary_files = sorted(
        glob.glob(os.path.join(FRESH_DIR, "*", "_results_summary.jsonl"))
        + glob.glob(os.path.join(NEW_DIR, "*", "_results_summary.jsonl"))
    )
    print(f"Loading {len(summary_files)} dLLM config summaries...")

    for sf in summary_files:
        config_dir = os.path.basename(os.path.dirname(sf))
        samples = load_summary_jsonl(sf)
        per_sample = [None] * len(task_order)
        for s in samples:
            idx = task_id_to_idx.get(s["task_id"])
            if idx is None:
                continue
            st = s["stats"]
            per_sample[idx] = {
                "g": clean_generated(s["generated"]),
                "ts": round(st.get("batch_tokens_per_second", st.get("tokens_per_second", 0)), 2),
                "gt": st.get("generated_tokens", st.get("batch_total_tokens", 0)),
                "st": st.get("batch_total_steps", st.get("generated_tokens", 0)),
                "tps": round(st.get("tokens_per_step", 1.0), 2),
            }
        configs_samples[config_dir] = per_sample
        print(f"  {config_dir}: {sum(1 for x in per_sample if x)} samples")

    # ── Build compact JSON payload ───────────────────────────────────────
    payload = {
        "metrics": all_metrics,
        "tasks": tasks,
        "samples": configs_samples,
    }

    payload_json = json.dumps(payload, separators=(",", ":"))
    print(f"Payload size: {len(payload_json) / 1024 / 1024:.1f} MB")

    # ── Generate HTML ────────────────────────────────────────────────────
    html_content = build_html(payload_json)

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Written: {OUTPUT_HTML} ({os.path.getsize(OUTPUT_HTML) / 1024 / 1024:.1f} MB)")


def build_html(payload_json: str) -> str:
    return (
        HTML_HEAD
        + f"\n<script>const DATA = {payload_json};</script>\n"
        + HTML_BODY
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML Template
# ═══════════════════════════════════════════════════════════════════════════

HTML_HEAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>dLLM Ablation Results Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff; --accent2: #3fb950;
    --accent3: #f0883e; --accent4: #bc8cff; --red: #f85149;
    --radius: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }
  a { color: var(--accent); text-decoration: none; }

  /* Layout */
  .app { display: flex; flex-direction: column; min-height: 100vh; }
  .topbar { background: var(--bg2); border-bottom: 1px solid var(--border);
            padding: 12px 24px; display: flex; align-items: center; gap: 24px; position: sticky; top: 0; z-index: 100; }
  .topbar h1 { font-size: 18px; font-weight: 600; white-space: nowrap; }
  .tabs { display: flex; gap: 4px; }
  .tab { padding: 6px 16px; border-radius: var(--radius); cursor: pointer;
         background: transparent; border: 1px solid transparent; color: var(--text2);
         font-size: 14px; transition: all .15s; }
  .tab:hover { background: var(--bg3); color: var(--text); }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .content { flex: 1; padding: 24px; max-width: 1600px; margin: 0 auto; width: 100%; }
  .panel { display: none; }
  .panel.active { display: block; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
          padding: 16px; }
  .card .label { font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: .5px; }
  .card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .card .sub { font-size: 12px; color: var(--text2); margin-top: 2px; }

  /* Tables */
  .table-wrap { overflow-x: auto; margin-bottom: 24px; border: 1px solid var(--border); border-radius: var(--radius); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { background: var(--bg2); color: var(--text2); font-weight: 600; position: sticky; top: 0;
       cursor: pointer; user-select: none; }
  th:hover { color: var(--text); }
  th .sort-arrow { margin-left: 4px; font-size: 10px; }
  tr:hover td { background: var(--bg3); }
  tr.highlight td { background: rgba(88,166,255,.12); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }

  /* Controls */
  .controls { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; align-items: center; }
  select, input[type="text"], input[type="number"] {
    background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    color: var(--text); padding: 6px 12px; font-size: 13px; outline: none; }
  select:focus, input:focus { border-color: var(--accent); }
  select { cursor: pointer; }
  button { background: var(--accent); color: #fff; border: none; border-radius: var(--radius);
           padding: 6px 16px; font-size: 13px; cursor: pointer; transition: opacity .15s; }
  button:hover { opacity: .85; }
  button.secondary { background: var(--bg3); color: var(--text); border: 1px solid var(--border); }
  label { font-size: 13px; color: var(--text2); }

  /* Charts */
  .chart-container { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
                     padding: 16px; margin-bottom: 24px; position: relative; }
  .chart-container canvas { max-height: 450px; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  @media (max-width: 1000px) { .chart-row { grid-template-columns: 1fr; } }

  /* Sample browser */
  .sample-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
                 padding: 16px; margin-bottom: 12px; }
  .sample-card .meta { font-size: 12px; color: var(--text2); margin-bottom: 8px; display: flex; gap: 16px; flex-wrap: wrap; }
  .sample-card .meta span { display: inline-flex; align-items: center; gap: 4px; }
  .sample-card .texts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 800px) { .sample-card .texts { grid-template-columns: 1fr; } }
  .text-block { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
  .text-block .heading { font-size: 11px; color: var(--text2); text-transform: uppercase;
                         letter-spacing: .5px; margin-bottom: 6px; }
  .text-block .content-text { font-size: 14px; word-break: break-word; white-space: pre-wrap; }
  .text-block .content-text.label-text { color: var(--accent2); }
  .text-block .content-text.gen-text { color: var(--accent); }
  .text-block .content-text.ar-text { color: var(--accent3); }

  .pagination { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }
  .pagination .info { font-size: 13px; color: var(--text2); }

  /* Comparison checkboxes */
  .config-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .chip { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px;
          border-radius: 20px; font-size: 12px; cursor: pointer; border: 1px solid var(--border);
          background: var(--bg2); transition: all .15s; user-select: none; }
  .chip.selected { background: var(--accent); color: #fff; border-color: var(--accent); }
  .chip.ar { border-color: var(--accent3); }
  .chip.ar.selected { background: var(--accent3); border-color: var(--accent3); }

  /* Heatmap */
  .heatmap-cell { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
                  font-size: 12px; font-weight: 600; }

  /* Misc */
  .section-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: var(--text); }
  .muted { color: var(--text2); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-ar { background: rgba(240,136,62,.2); color: var(--accent3); }
  .badge-dllm { background: rgba(88,166,255,.2); color: var(--accent); }
  .hidden { display: none !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--border); }

  .multi-select-wrap { position: relative; }
  .multi-select-btn { min-width: 220px; text-align: left; display: flex; justify-content: space-between; align-items: center; }
  .multi-select-btn::after { content: '▾'; margin-left: 8px; }
  .multi-select-dropdown { position: absolute; top: 100%; left: 0; min-width: 350px; max-height: 400px;
    overflow-y: auto; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    z-index: 200; display: none; margin-top: 4px; box-shadow: 0 8px 24px rgba(0,0,0,.4); }
  .multi-select-dropdown.open { display: block; }
  .ms-item { padding: 6px 12px; cursor: pointer; font-size: 13px; display: flex; align-items: center; gap: 8px; }
  .ms-item:hover { background: var(--bg3); }
  .ms-item input { accent-color: var(--accent); }
  .ms-group-label { padding: 8px 12px 4px; font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: .5px; }

  .diff-highlight { background: rgba(63,185,80,.15); padding: 1px 2px; border-radius: 2px; }
</style>
</head>
<body>
"""

HTML_BODY = r"""
<div class="app">
  <div class="topbar">
    <h1>🔬 dLLM Ablation Results</h1>
    <div class="tabs">
      <div class="tab active" data-tab="overview">Overview</div>
      <div class="tab" data-tab="compare">Compare</div>
      <div class="tab" data-tab="samples">Sample Browser</div>
      <div class="tab" data-tab="hparams">Hyperparams</div>
    </div>
  </div>

  <div class="content">
    <!-- ═══ OVERVIEW TAB ═══ -->
    <div class="panel active" id="panel-overview">
      <div id="overview-cards" class="cards"></div>
      <div class="chart-row">
        <div class="chart-container">
          <div class="section-title">Quality vs Speed (all configs)</div>
          <canvas id="chart-pareto"></canvas>
        </div>
        <div class="chart-container">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
            <span class="section-title" style="margin-bottom:0">Metric by Configuration (top 20)</span>
            <select id="overview-metric-sel" style="margin-left:auto">
              <option value="avg_meteor">METEOR</option>
              <option value="avg_rougeL">ROUGE-L</option>
              <option value="avg_rouge1">ROUGE-1</option>
              <option value="avg_rouge2">ROUGE-2</option>
              <option value="avg_bleu4">BLEU-4</option>
              <option value="avg_bleu_code">BLEU-CODE</option>
              <option value="avg_cider">CIDEr</option>
              <option value="avg_tokens_per_second">Tokens/sec</option>
              <option value="avg_tokens_per_step">Tokens/step</option>
              <option value="avg_generated_tokens">Avg Gen Tokens</option>
              <option value="avg_diffusion_steps">Avg Steps</option>
            </select>
          </div>
          <canvas id="chart-meteor-bar"></canvas>
        </div>
      </div>
      <div class="section-title">All Configurations</div>
      <div class="controls">
        <input type="text" id="overview-search" placeholder="Filter configs..." style="width:250px">
        <label><input type="checkbox" id="overview-dllm-only"> dLLM only</label>
      </div>
      <div class="table-wrap" style="max-height:600px; overflow-y:auto;">
        <table id="overview-table">
          <thead><tr id="overview-thead"></tr></thead>
          <tbody id="overview-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- ═══ COMPARE TAB ═══ -->
    <div class="panel" id="panel-compare">
      <div class="section-title">Select Configurations to Compare</div>
      <div class="controls">
        <div class="multi-select-wrap">
          <button class="secondary multi-select-btn" id="compare-select-btn">Select configs...</button>
          <div class="multi-select-dropdown" id="compare-dropdown"></div>
        </div>
        <button id="compare-clear">Clear All</button>
        <button id="compare-presets" class="secondary">Best Presets</button>
      </div>
      <div class="config-chips" id="compare-chips"></div>
      <div class="chart-row">
        <div class="chart-container">
          <div class="section-title">Quality Metrics Comparison</div>
          <canvas id="chart-compare-quality"></canvas>
        </div>
        <div class="chart-container">
          <div class="section-title">Speed Metrics Comparison</div>
          <canvas id="chart-compare-speed"></canvas>
        </div>
      </div>
      <div class="chart-container">
        <div class="section-title">Radar Chart</div>
        <canvas id="chart-radar" style="max-height:500px"></canvas>
      </div>
      <div class="section-title">Side-by-Side Table</div>
      <div class="table-wrap">
        <table id="compare-table">
          <thead><tr id="compare-thead"></tr></thead>
          <tbody id="compare-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- ═══ SAMPLES TAB ═══ -->
    <div class="panel" id="panel-samples">
      <div class="controls">
        <label>Config:
          <select id="sample-config" style="min-width:300px"></select>
        </label>
        <label>Compare with:
          <select id="sample-compare" style="min-width:200px">
            <option value="">None</option>
          </select>
        </label>
        <input type="text" id="sample-search" placeholder="Search task ID or text..." style="width:250px">
        <label>Sort:
          <select id="sample-sort">
            <option value="idx">Default Order</option>
            <option value="ts_desc">Speed ↓</option>
            <option value="ts_asc">Speed ↑</option>
            <option value="gt_desc">Gen Tokens ↓</option>
            <option value="gt_asc">Gen Tokens ↑</option>
            <option value="label_short">Label Shortest</option>
            <option value="label_long">Label Longest</option>
          </select>
        </label>
      </div>
      <div class="pagination">
        <button id="sample-prev" class="secondary">← Prev</button>
        <span class="info" id="sample-page-info"></span>
        <button id="sample-next" class="secondary">Next →</button>
        <label style="margin-left:auto">Per page:
          <select id="sample-perpage">
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </label>
      </div>
      <div id="sample-list"></div>
      <div class="pagination" style="margin-top:12px">
        <button id="sample-prev2" class="secondary">← Prev</button>
        <span class="info" id="sample-page-info2"></span>
        <button id="sample-next2" class="secondary">Next →</button>
      </div>
    </div>

    <!-- ═══ HYPERPARAMS TAB ═══ -->
    <div class="panel" id="panel-hparams">
      <div class="controls">
        <label>X-axis:
          <select id="hp-x">
            <option value="threshold">Threshold</option>
            <option value="block_size">Block Size</option>
            <option value="small_block_size">Sub-Block Size</option>
            <option value="max_new_tokens">Max New Tokens</option>
            <option value="batch_size">Batch Size</option>
          </select>
        </label>
        <label>Y-axis:
          <select id="hp-y">
            <option value="avg_meteor">METEOR</option>
            <option value="avg_rougeL">ROUGE-L</option>
            <option value="avg_rouge1">ROUGE-1</option>
            <option value="avg_bleu4">BLEU-4</option>
            <option value="avg_bleu_code">BLEU-CODE</option>
            <option value="avg_cider">CIDEr</option>
            <option value="avg_tokens_per_second">Tokens/sec</option>
            <option value="avg_tokens_per_step">Tokens/step</option>
            <option value="avg_generated_tokens">Avg Gen Tokens</option>
          </select>
        </label>
        <label>Color by:
          <select id="hp-color">
            <option value="threshold">Threshold</option>
            <option value="block_size">Block Size</option>
            <option value="max_new_tokens">Max New Tokens</option>
            <option value="batch_size">Batch Size</option>
          </select>
        </label>
        <label>Filter bs:
          <select id="hp-filter-bs"><option value="">All</option></select>
        </label>
        <label>Filter sbs:
          <select id="hp-filter-sbs"><option value="">All</option></select>
        </label>
        <label>Filter mnt:
          <select id="hp-filter-mnt"><option value="">All</option></select>
        </label>
      </div>
      <div class="chart-container">
        <canvas id="chart-hparams" style="max-height:500px"></canvas>
      </div>
      <div class="chart-row">
        <div class="chart-container">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
            <span class="section-title" style="margin-bottom:0">Threshold Effect (grouped)</span>
            <select id="hp-th-metric" style="margin-left:auto">
              <option value="avg_meteor">METEOR</option>
              <option value="avg_rougeL">ROUGE-L</option>
              <option value="avg_rouge1">ROUGE-1</option>
              <option value="avg_rouge2">ROUGE-2</option>
              <option value="avg_bleu4">BLEU-4</option>
              <option value="avg_bleu_code">BLEU-CODE</option>
              <option value="avg_cider">CIDEr</option>
              <option value="avg_tokens_per_second">Tokens/sec</option>
              <option value="avg_tokens_per_step">Tokens/step</option>
              <option value="avg_generated_tokens">Avg Gen Tokens</option>
            </select>
          </div>
          <canvas id="chart-threshold-line"></canvas>
        </div>
        <div class="chart-container">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
            <span class="section-title" style="margin-bottom:0">Block Size Effect</span>
            <select id="hp-bs-metric" style="margin-left:auto">
              <option value="avg_meteor">METEOR</option>
              <option value="avg_rougeL">ROUGE-L</option>
              <option value="avg_rouge1">ROUGE-1</option>
              <option value="avg_rouge2">ROUGE-2</option>
              <option value="avg_bleu4">BLEU-4</option>
              <option value="avg_bleu_code">BLEU-CODE</option>
              <option value="avg_cider">CIDEr</option>
              <option value="avg_tokens_per_second">Tokens/sec</option>
              <option value="avg_tokens_per_step">Tokens/step</option>
              <option value="avg_generated_tokens">Avg Gen Tokens</option>
            </select>
          </div>
          <canvas id="chart-bs-box"></canvas>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
// DATA ACCESS HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const M = DATA.metrics;
const T = DATA.tasks;
const S = DATA.samples;

const configNames = Object.keys(M).sort((a, b) => {
  if (a === 'AR_baseline') return -1;
  if (b === 'AR_baseline') return 1;
  return a.localeCompare(b);
});
const dllmConfigs = configNames.filter(c => M[c].model_type === 'dLLM');

// Color palette
const COLORS = [
  '#58a6ff','#3fb950','#f0883e','#bc8cff','#f85149','#79c0ff','#56d364',
  '#e3b341','#db61a2','#7ee787','#a5d6ff','#ffa657','#d2a8ff','#ff7b72',
  '#b1bac4','#238636','#1f6feb','#8957e5','#da3633','#d29922'
];
function getColor(i) { return COLORS[i % COLORS.length]; }

// ═══════════════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═══════════════════════════════════════════════════════════════════════════
function initOverview() {
  // Summary cards
  const ar = M['AR_baseline'];
  const bestMeteor = dllmConfigs.reduce((best, c) => M[c].avg_meteor > M[best].avg_meteor ? c : best, dllmConfigs[0]);
  const bestSpeed = dllmConfigs.reduce((best, c) => M[c].avg_tokens_per_second > M[best].avg_tokens_per_second ? c : best, dllmConfigs[0]);
  const avgSpeedup = dllmConfigs.reduce((s, c) => s + M[c].avg_tokens_per_second / ar.avg_tokens_per_second, 0) / dllmConfigs.length;

  const cardsHtml = `
    <div class="card"><div class="label">Total Configs</div><div class="value">${configNames.length}</div><div class="sub">${dllmConfigs.length} dLLM + 1 AR baseline</div></div>
    <div class="card"><div class="label">Best METEOR (dLLM)</div><div class="value" style="color:var(--accent2)">${M[bestMeteor].avg_meteor.toFixed(4)}</div><div class="sub">${bestMeteor}</div></div>
    <div class="card"><div class="label">AR Baseline METEOR</div><div class="value" style="color:var(--accent3)">${ar.avg_meteor.toFixed(4)}</div><div class="sub">Greedy, mnt=1024</div></div>
    <div class="card"><div class="label">Best Speed (dLLM)</div><div class="value" style="color:var(--accent)">${M[bestSpeed].avg_tokens_per_second.toFixed(1)} t/s</div><div class="sub">${bestSpeed}</div></div>
    <div class="card"><div class="label">AR Baseline Speed</div><div class="value">${ar.avg_tokens_per_second.toFixed(1)} t/s</div><div class="sub">Reference</div></div>
    <div class="card"><div class="label">Avg Speedup</div><div class="value">${avgSpeedup.toFixed(2)}×</div><div class="sub">Over AR baseline</div></div>
  `;
  document.getElementById('overview-cards').innerHTML = cardsHtml;

  // Pareto chart
  const paretoCtx = document.getElementById('chart-pareto').getContext('2d');
  const thresholds = [...new Set(dllmConfigs.map(c => M[c].threshold))].sort();
  const thColors = {};
  thresholds.forEach((t, i) => thColors[t] = getColor(i));

  const paretoDatasets = thresholds.map(th => ({
    label: `th=${th}`,
    data: dllmConfigs.filter(c => M[c].threshold === th).map(c => ({
      x: M[c].avg_tokens_per_second,
      y: M[c].avg_meteor,
      config: c
    })),
    backgroundColor: thColors[th] + '99',
    borderColor: thColors[th],
    pointRadius: 5,
    pointHoverRadius: 8,
  }));
  // AR baseline point
  paretoDatasets.push({
    label: 'AR Baseline',
    data: [{ x: ar.avg_tokens_per_second, y: ar.avg_meteor, config: 'AR_baseline' }],
    backgroundColor: '#f85149',
    borderColor: '#f85149',
    pointRadius: 10,
    pointStyle: 'star',
  });

  new Chart(paretoCtx, {
    type: 'scatter',
    data: { datasets: paretoDatasets },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              return `${d.config}: METEOR=${d.y.toFixed(4)}, ${d.x.toFixed(1)} tok/s`;
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Tokens/sec', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { title: { display: true, text: 'METEOR', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      }
    }
  });

  // Metric bar chart (top 20 + AR) — reactive to dropdown
  let overviewBarChart = null;
  function renderOverviewBar() {
    const metricKey = document.getElementById('overview-metric-sel').value;
    const metricLabel = document.getElementById('overview-metric-sel').selectedOptions[0].text;
    const sorted = [...dllmConfigs].sort((a, b) => M[b][metricKey] - M[a][metricKey]).slice(0, 20);
    sorted.push('AR_baseline');
    if (overviewBarChart) overviewBarChart.destroy();
    overviewBarChart = new Chart(document.getElementById('chart-meteor-bar'), {
      type: 'bar',
      data: {
        labels: sorted.map(c => c.replace(/_cacheTrue/g, '').replace(/_batch1/g, '')),
        datasets: [{
          data: sorted.map(c => M[c][metricKey]),
          backgroundColor: sorted.map(c => c === 'AR_baseline' ? '#f0883e' : '#58a6ff'),
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: metricLabel, color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
          y: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { display: false } }
        }
      }
    });
  }
  document.getElementById('overview-metric-sel').addEventListener('change', renderOverviewBar);
  renderOverviewBar();

  // Overview table
  const cols = [
    { key: 'config', label: 'Config', fmt: v => v },
    { key: 'model_type', label: 'Type', fmt: v => `<span class="badge badge-${v.toLowerCase()}">${v}</span>` },
    { key: 'block_size', label: 'BS', fmt: v => v ?? '-', cls: 'num' },
    { key: 'small_block_size', label: 'SBS', fmt: v => v ?? '-', cls: 'num' },
    { key: 'threshold', label: 'Thresh', fmt: v => v?.toFixed(1) ?? '-', cls: 'num' },
    { key: 'max_new_tokens', label: 'MNT', fmt: v => v, cls: 'num' },
    { key: 'batch_size', label: 'Batch', fmt: v => v, cls: 'num' },
    { key: 'avg_meteor', label: 'METEOR', fmt: v => v.toFixed(4), cls: 'num' },
    { key: 'avg_rougeL', label: 'ROUGE-L', fmt: v => v.toFixed(4), cls: 'num' },
    { key: 'avg_rouge1', label: 'ROUGE-1', fmt: v => v.toFixed(4), cls: 'num' },
    { key: 'avg_bleu4', label: 'BLEU-4', fmt: v => v.toFixed(4), cls: 'num' },
    { key: 'avg_bleu_code', label: 'BLEU-CODE', fmt: v => v.toFixed(4), cls: 'num' },
    { key: 'avg_cider', label: 'CIDEr', fmt: v => v.toFixed(4), cls: 'num' },
    { key: 'avg_tokens_per_second', label: 'Tok/s', fmt: v => v.toFixed(2), cls: 'num' },
    { key: 'avg_tokens_per_step', label: 'Tok/step', fmt: v => v.toFixed(2), cls: 'num' },
    { key: 'avg_generated_tokens', label: 'Avg Gen Tok', fmt: v => v.toFixed(1), cls: 'num' },
    { key: 'avg_diffusion_steps', label: 'Avg Steps', fmt: v => v.toFixed(1), cls: 'num' },
  ];

  // Build thead
  const thead = document.getElementById('overview-thead');
  thead.innerHTML = cols.map(c => `<th data-key="${c.key}">${c.label}<span class="sort-arrow"></span></th>`).join('');

  let sortKey = 'avg_meteor', sortDir = -1;
  let filterText = '', dllmOnly = false;

  function renderOverviewTable() {
    let rows = configNames.filter(c => {
      if (dllmOnly && M[c].model_type !== 'dLLM') return false;
      if (filterText && !c.toLowerCase().includes(filterText)) return false;
      return true;
    });
    rows.sort((a, b) => {
      let va = M[a][sortKey], vb = M[b][sortKey];
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
      if (typeof va === 'string') return sortDir * va.localeCompare(vb);
      return sortDir * (va - vb);
    });
    const tbody = document.getElementById('overview-tbody');
    tbody.innerHTML = rows.map(c => {
      const m = M[c];
      return '<tr>' + cols.map(col => {
        const cls = col.cls ? ` class="${col.cls}"` : '';
        return `<td${cls}>${col.fmt(m[col.key])}</td>`;
      }).join('') + '</tr>';
    }).join('');

    // Update sort arrows
    thead.querySelectorAll('th').forEach(th => {
      const arrow = th.querySelector('.sort-arrow');
      arrow.textContent = th.dataset.key === sortKey ? (sortDir === 1 ? '▲' : '▼') : '';
    });
  }

  thead.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      if (sortKey === th.dataset.key) sortDir *= -1;
      else { sortKey = th.dataset.key; sortDir = -1; }
      renderOverviewTable();
    });
  });

  document.getElementById('overview-search').addEventListener('input', e => {
    filterText = e.target.value.toLowerCase();
    renderOverviewTable();
  });
  document.getElementById('overview-dllm-only').addEventListener('change', e => {
    dllmOnly = e.target.checked;
    renderOverviewTable();
  });

  renderOverviewTable();
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPARE TAB
// ═══════════════════════════════════════════════════════════════════════════
let compareSelected = new Set();
let compareCharts = {};

function initCompare() {
  const dropdown = document.getElementById('compare-dropdown');
  const btn = document.getElementById('compare-select-btn');

  // Group configs
  const groups = {};
  configNames.forEach(c => {
    const m = M[c];
    const group = m.model_type === 'AR' ? 'AR Baseline' :
      `bs=${m.block_size} sbs=${m.small_block_size} mnt=${m.max_new_tokens}`;
    if (!groups[group]) groups[group] = [];
    groups[group].push(c);
  });

  dropdown.innerHTML = Object.entries(groups).map(([g, configs]) =>
    `<div class="ms-group-label">${g}</div>` +
    configs.map(c => `<div class="ms-item" data-config="${c}"><input type="checkbox" ${compareSelected.has(c)?'checked':''}><span>${c}</span></div>`).join('')
  ).join('');

  btn.addEventListener('click', e => {
    e.stopPropagation();
    dropdown.classList.toggle('open');
  });
  document.addEventListener('click', () => dropdown.classList.remove('open'));
  dropdown.addEventListener('click', e => e.stopPropagation());

  dropdown.querySelectorAll('.ms-item').forEach(item => {
    item.addEventListener('click', () => {
      const c = item.dataset.config;
      const cb = item.querySelector('input');
      if (compareSelected.has(c)) { compareSelected.delete(c); cb.checked = false; }
      else { compareSelected.add(c); cb.checked = true; }
      renderCompare();
    });
  });

  document.getElementById('compare-clear').addEventListener('click', () => {
    compareSelected.clear();
    dropdown.querySelectorAll('input').forEach(cb => cb.checked = false);
    renderCompare();
  });

  document.getElementById('compare-presets').addEventListener('click', () => {
    compareSelected.clear();
    // Add AR + best METEOR + fastest + a few interesting ones
    compareSelected.add('AR_baseline');
    const bestM = dllmConfigs.reduce((b, c) => M[c].avg_meteor > M[b].avg_meteor ? c : b, dllmConfigs[0]);
    const bestS = dllmConfigs.reduce((b, c) => M[c].avg_tokens_per_second > M[b].avg_tokens_per_second ? c : b, dllmConfigs[0]);
    compareSelected.add(bestM);
    compareSelected.add(bestS);
    // Add one config at each unique threshold with bs=32, sbs=8, mnt=128
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(th => {
      const c = dllmConfigs.find(c => M[c].threshold === th && M[c].block_size === 32 && M[c].small_block_size === 8 && M[c].max_new_tokens === 128);
      if (c) compareSelected.add(c);
    });
    dropdown.querySelectorAll('input').forEach(cb => {
      cb.checked = compareSelected.has(cb.parentElement.dataset.config);
    });
    renderCompare();
  });

  renderCompare();
}

function renderCompare() {
  const sel = [...compareSelected];
  // Chips
  document.getElementById('compare-chips').innerHTML = sel.map((c, i) =>
    `<div class="chip selected" style="background:${getColor(i)};border-color:${getColor(i)}" data-config="${c}">
      ${c.replace(/_cacheTrue/g,'').replace(/_batch1/g,'')} ×
    </div>`
  ).join('');

  document.querySelectorAll('#compare-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      compareSelected.delete(chip.dataset.config);
      document.querySelector(`#compare-dropdown .ms-item[data-config="${chip.dataset.config}"] input`).checked = false;
      renderCompare();
    });
  });

  btn_text = sel.length ? `${sel.length} selected` : 'Select configs...';
  document.getElementById('compare-select-btn').childNodes[0].textContent = btn_text;

  if (sel.length === 0) return;

  // Quality chart
  const qMetrics = ['avg_meteor','avg_rougeL','avg_rouge1','avg_bleu4','avg_bleu_code','avg_cider'];
  const qLabels = ['METEOR','ROUGE-L','ROUGE-1','BLEU-4','BLEU-CODE','CIDEr'];
  if (compareCharts.quality) compareCharts.quality.destroy();
  compareCharts.quality = new Chart(document.getElementById('chart-compare-quality'), {
    type: 'bar',
    data: {
      labels: qLabels,
      datasets: sel.map((c, i) => ({
        label: c.replace(/_cacheTrue/g,'').replace(/_batch1/g,''),
        data: qMetrics.map(k => M[c][k]),
        backgroundColor: getColor(i) + 'cc',
        borderColor: getColor(i),
        borderWidth: 1,
        borderRadius: 3,
      }))
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true }
      }
    }
  });

  // Speed chart
  const sMetrics = ['avg_tokens_per_second','avg_tokens_per_step','avg_generated_tokens'];
  const sLabels = ['Tok/sec','Tok/step','Avg Gen Tokens'];
  if (compareCharts.speed) compareCharts.speed.destroy();
  compareCharts.speed = new Chart(document.getElementById('chart-compare-speed'), {
    type: 'bar',
    data: {
      labels: sLabels,
      datasets: sel.map((c, i) => ({
        label: c.replace(/_cacheTrue/g,'').replace(/_batch1/g,''),
        data: sMetrics.map(k => M[c][k]),
        backgroundColor: getColor(i) + 'cc',
        borderColor: getColor(i),
        borderWidth: 1,
        borderRadius: 3,
      }))
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true }
      }
    }
  });

  // Radar chart
  const radarMetrics = ['avg_meteor','avg_rougeL','avg_bleu4','avg_bleu_code','avg_cider','avg_tokens_per_second','avg_tokens_per_step'];
  const radarLabels = ['METEOR','ROUGE-L','BLEU-4','BLEU-CODE','CIDEr','Tok/sec','Tok/step'];
  // Normalize to 0-1 range
  const radarMaxes = radarMetrics.map(k => Math.max(...configNames.map(c => M[c][k])));
  if (compareCharts.radar) compareCharts.radar.destroy();
  compareCharts.radar = new Chart(document.getElementById('chart-radar'), {
    type: 'radar',
    data: {
      labels: radarLabels,
      datasets: sel.map((c, i) => ({
        label: c.replace(/_cacheTrue/g,'').replace(/_batch1/g,''),
        data: radarMetrics.map((k, j) => M[c][k] / radarMaxes[j]),
        borderColor: getColor(i),
        backgroundColor: getColor(i) + '22',
        pointBackgroundColor: getColor(i),
        borderWidth: 2,
      }))
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        r: {
          ticks: { color: '#8b949e', backdropColor: 'transparent' },
          grid: { color: '#30363d' },
          pointLabels: { color: '#e6edf3', font: { size: 12 } },
          suggestedMin: 0, suggestedMax: 1
        }
      }
    }
  });

  // Table
  const allMetricKeys = ['model_type','block_size','small_block_size','threshold','max_new_tokens','batch_size',
    'avg_meteor','avg_rougeL','avg_rouge1','avg_rouge2','avg_bleu4','avg_bleu_code','avg_cider',
    'avg_tokens_per_second','avg_ms_per_token','avg_tokens_per_step','avg_diffusion_steps','avg_generated_tokens','num_samples'];
  const metricLabels = ['Type','BS','SBS','Thresh','MNT','Batch',
    'METEOR','ROUGE-L','ROUGE-1','ROUGE-2','BLEU-4','BLEU-CODE','CIDEr',
    'Tok/s','ms/tok','Tok/step','Avg Steps','Avg Gen Tok','Samples'];

  document.getElementById('compare-thead').innerHTML = '<th>Metric</th>' + sel.map((c, i) =>
    `<th style="color:${getColor(i)}">${c.replace(/_cacheTrue/g,'').replace(/_batch1/g,'')}</th>`
  ).join('');

  document.getElementById('compare-tbody').innerHTML = allMetricKeys.map((k, ki) => {
    const vals = sel.map(c => M[c][k]);
    const isNum = typeof vals.find(v => v != null) === 'number';
    const best = isNum ? Math.max(...vals.filter(v => v != null)) : null;
    return '<tr><td><strong>' + metricLabels[ki] + '</strong></td>' +
      sel.map((c, i) => {
        let v = M[c][k];
        let display = v == null ? '-' : (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v);
        let style = isNum && v === best ? 'font-weight:700;color:var(--accent2)' : '';
        return `<td class="num" style="${style}">${display}</td>`;
      }).join('') + '</tr>';
  }).join('');
}

// ═══════════════════════════════════════════════════════════════════════════
// SAMPLES TAB
// ═══════════════════════════════════════════════════════════════════════════
let samplePage = 0;
let sampleIndices = [];

function initSamples() {
  const configSelect = document.getElementById('sample-config');
  const compareSelect = document.getElementById('sample-compare');

  configNames.forEach(c => {
    configSelect.innerHTML += `<option value="${c}">${c}</option>`;
    compareSelect.innerHTML += `<option value="${c}">${c}</option>`;
  });

  configSelect.addEventListener('change', () => { samplePage = 0; renderSamples(); });
  compareSelect.addEventListener('change', renderSamples);
  document.getElementById('sample-search').addEventListener('input', () => { samplePage = 0; renderSamples(); });
  document.getElementById('sample-sort').addEventListener('change', () => { samplePage = 0; renderSamples(); });
  document.getElementById('sample-perpage').addEventListener('change', () => { samplePage = 0; renderSamples(); });

  ['sample-prev','sample-prev2'].forEach(id => document.getElementById(id).addEventListener('click', () => {
    if (samplePage > 0) { samplePage--; renderSamples(); }
  }));
  ['sample-next','sample-next2'].forEach(id => document.getElementById(id).addEventListener('click', () => {
    const pp = parseInt(document.getElementById('sample-perpage').value);
    if ((samplePage + 1) * pp < sampleIndices.length) { samplePage++; renderSamples(); }
  }));

  renderSamples();
}

function renderSamples() {
  const config = document.getElementById('sample-config').value;
  const compareConfig = document.getElementById('sample-compare').value;
  const searchText = document.getElementById('sample-search').value.toLowerCase();
  const sortMode = document.getElementById('sample-sort').value;
  const perPage = parseInt(document.getElementById('sample-perpage').value);

  const samples = S[config];
  if (!samples) return;

  // Build filtered indices
  sampleIndices = [];
  for (let i = 0; i < T.length; i++) {
    if (!samples[i]) continue;
    if (searchText) {
      const tid = T[i].id.toLowerCase();
      const label = T[i].label.toLowerCase();
      const gen = samples[i].g.toLowerCase();
      if (!tid.includes(searchText) && !label.includes(searchText) && !gen.includes(searchText)) continue;
    }
    sampleIndices.push(i);
  }

  // Sort
  sampleIndices.sort((a, b) => {
    switch (sortMode) {
      case 'ts_desc': return (samples[b]?.ts || 0) - (samples[a]?.ts || 0);
      case 'ts_asc': return (samples[a]?.ts || 0) - (samples[b]?.ts || 0);
      case 'gt_desc': return (samples[b]?.gt || 0) - (samples[a]?.gt || 0);
      case 'gt_asc': return (samples[a]?.gt || 0) - (samples[b]?.gt || 0);
      case 'label_short': return T[a].label.length - T[b].label.length;
      case 'label_long': return T[b].label.length - T[a].label.length;
      default: return a - b;
    }
  });

  const total = sampleIndices.length;
  const start = samplePage * perPage;
  const end = Math.min(start + perPage, total);
  const pageIndices = sampleIndices.slice(start, end);

  const pageInfo = `Showing ${start + 1}–${end} of ${total}`;
  document.getElementById('sample-page-info').textContent = pageInfo;
  document.getElementById('sample-page-info2').textContent = pageInfo;

  const compareSamples = compareConfig ? S[compareConfig] : null;
  const m = M[config];

  let html = '';
  for (const idx of pageIndices) {
    const task = T[idx];
    const s = samples[idx];
    if (!s) continue;

    const cs = compareSamples ? compareSamples[idx] : null;

    html += `<div class="sample-card">
      <div class="meta">
        <span><strong>${task.id}</strong></span>
        <span>⚡ ${s.ts} tok/s</span>
        <span>📝 ${s.gt} gen tokens</span>
        <span>🔄 ${s.st} steps</span>
        <span>📊 ${s.tps} tok/step</span>
        <span>📏 Prompt: ${task.prompt_tok} tok</span>
        ${cs ? `<span style="color:var(--accent4)">Compare: ⚡${cs.ts} tok/s, 📝${cs.gt} tok</span>` : ''}
      </div>
      <div class="texts" style="${cs ? 'grid-template-columns:1fr 1fr 1fr' : ''}">
        <div class="text-block">
          <div class="heading">Ground Truth</div>
          <div class="content-text label-text">${escHtml(task.label)}</div>
        </div>
        <div class="text-block">
          <div class="heading">${config.replace(/_cacheTrue/g,'').replace(/_batch1/g,'')}</div>
          <div class="content-text gen-text">${escHtml(s.g)}</div>
        </div>
        ${cs ? `<div class="text-block">
          <div class="heading" style="color:var(--accent3)">${compareConfig.replace(/_cacheTrue/g,'').replace(/_batch1/g,'')}</div>
          <div class="content-text ar-text">${escHtml(cs.g)}</div>
        </div>` : ''}
      </div>
    </div>`;
  }

  document.getElementById('sample-list').innerHTML = html || '<p class="muted">No samples match your filter.</p>';
}

function escHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════════════
// HYPERPARAMS TAB
// ═══════════════════════════════════════════════════════════════════════════
let hpChart = null, thresholdChart = null, bsChart = null;

function initHparams() {
  // Populate filter dropdowns
  const bsVals = [...new Set(dllmConfigs.map(c => M[c].block_size))].sort((a,b)=>a-b);
  const sbsVals = [...new Set(dllmConfigs.map(c => M[c].small_block_size))].sort((a,b)=>a-b);
  const mntVals = [...new Set(dllmConfigs.map(c => M[c].max_new_tokens))].sort((a,b)=>a-b);

  bsVals.forEach(v => document.getElementById('hp-filter-bs').innerHTML += `<option value="${v}">${v}</option>`);
  sbsVals.forEach(v => document.getElementById('hp-filter-sbs').innerHTML += `<option value="${v}">${v}</option>`);
  mntVals.forEach(v => document.getElementById('hp-filter-mnt').innerHTML += `<option value="${v}">${v}</option>`);

  ['hp-x','hp-y','hp-color','hp-filter-bs','hp-filter-sbs','hp-filter-mnt'].forEach(id => {
    document.getElementById(id).addEventListener('change', renderHparams);
  });
  document.getElementById('hp-th-metric').addEventListener('change', renderThresholdLine);
  document.getElementById('hp-bs-metric').addEventListener('change', renderBsChart);

  renderHparams();
  renderThresholdLine();
  renderBsChart();
}

function renderHparams() {
  const xKey = document.getElementById('hp-x').value;
  const yKey = document.getElementById('hp-y').value;
  const colorKey = document.getElementById('hp-color').value;
  const filterBs = document.getElementById('hp-filter-bs').value;
  const filterSbs = document.getElementById('hp-filter-sbs').value;
  const filterMnt = document.getElementById('hp-filter-mnt').value;

  let configs = dllmConfigs.filter(c => {
    if (filterBs && M[c].block_size != filterBs) return false;
    if (filterSbs && M[c].small_block_size != filterSbs) return false;
    if (filterMnt && M[c].max_new_tokens != filterMnt) return false;
    return true;
  });

  const colorVals = [...new Set(configs.map(c => M[c][colorKey]))].sort((a,b)=>a-b);
  const colorMap = {};
  colorVals.forEach((v, i) => colorMap[v] = getColor(i));

  const datasets = colorVals.map(cv => ({
    label: `${colorKey}=${cv}`,
    data: configs.filter(c => M[c][colorKey] === cv).map(c => ({
      x: M[c][xKey],
      y: M[c][yKey],
      config: c
    })),
    backgroundColor: colorMap[cv] + 'bb',
    borderColor: colorMap[cv],
    pointRadius: 7,
    pointHoverRadius: 10,
  }));

  // AR baseline reference line
  const arY = M['AR_baseline'][yKey];
  const arX = M['AR_baseline'][xKey];

  if (hpChart) hpChart.destroy();
  hpChart = new Chart(document.getElementById('chart-hparams'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        tooltip: { callbacks: { label: ctx => `${ctx.raw.config}: ${ctx.raw.y.toFixed(4)}` } },
        annotation: arY != null ? {
          annotations: {
            arLine: { type: 'line', yMin: arY, yMax: arY, borderColor: '#f0883e', borderDash: [6,3], borderWidth: 2,
              label: { display: true, content: `AR baseline: ${arY.toFixed(4)}`, position: 'start', color: '#f0883e', font: { size: 11 } } }
          }
        } : {}
      },
      scales: {
        x: { title: { display: true, text: xKey, color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { title: { display: true, text: yKey, color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      }
    }
  });
}

function renderThresholdLine() {
  const metricKey = document.getElementById('hp-th-metric').value;
  const metricLabel = document.getElementById('hp-th-metric').selectedOptions[0].text;
  // Group by (bs, sbs, mnt, batch) and plot metric vs threshold
  const groups = {};
  dllmConfigs.forEach(c => {
    const m = M[c];
    const key = `bs${m.block_size}_sbs${m.small_block_size}_mnt${m.max_new_tokens}_b${m.batch_size}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(c);
  });

  // Only show groups with >= 3 thresholds
  const validGroups = Object.entries(groups).filter(([k, v]) => v.length >= 3);
  validGroups.sort((a,b) => a[0].localeCompare(b[0]));

  const datasets = validGroups.map(([key, configs], i) => {
    configs.sort((a,b) => M[a].threshold - M[b].threshold);
    return {
      label: key,
      data: configs.map(c => ({ x: M[c].threshold, y: M[c][metricKey] })),
      borderColor: getColor(i),
      backgroundColor: getColor(i) + '44',
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 5,
      fill: false,
    };
  });

  if (thresholdChart) thresholdChart.destroy();
  thresholdChart = new Chart(document.getElementById('chart-threshold-line'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'Threshold', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, type: 'linear', min: 0, max: 1.1 },
        y: { title: { display: true, text: metricLabel, color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      }
    }
  });
}

function renderBsChart() {
  const metricKey = document.getElementById('hp-bs-metric').value;
  const metricLabel = document.getElementById('hp-bs-metric').selectedOptions[0].text;
  // Group by block_size, show distribution of chosen metric
  const bsGroups = {};
  dllmConfigs.forEach(c => {
    const bs = M[c].block_size;
    if (!bsGroups[bs]) bsGroups[bs] = [];
    bsGroups[bs].push(M[c][metricKey]);
  });

  const labels = Object.keys(bsGroups).sort((a,b) => a-b);
  const means = labels.map(bs => bsGroups[bs].reduce((s,v) => s+v, 0) / bsGroups[bs].length);
  const maxes = labels.map(bs => Math.max(...bsGroups[bs]));
  const mins = labels.map(bs => Math.min(...bsGroups[bs]));

  if (bsChart) bsChart.destroy();
  bsChart = new Chart(document.getElementById('chart-bs-box'), {
    type: 'bar',
    data: {
      labels: labels.map(bs => `BS=${bs}`),
      datasets: [
        { label: `Mean ${metricLabel}`, data: means, backgroundColor: '#58a6ff99', borderColor: '#58a6ff', borderWidth: 1, borderRadius: 3 },
        { label: 'Max', data: maxes, backgroundColor: '#3fb95066', borderColor: '#3fb950', borderWidth: 1, borderRadius: 3 },
        { label: 'Min', data: mins, backgroundColor: '#f8514966', borderColor: '#f85149', borderWidth: 1, borderRadius: 3 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { title: { display: true, text: metricLabel, color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  initOverview();
  initCompare();
  initSamples();
  initHparams();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
