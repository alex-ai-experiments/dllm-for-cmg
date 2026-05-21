# Research Questions — Diffusion LLMs for Commit Message Generation

## Thesis

We investigate whether **discrete diffusion language models (dLLMs)** can improve
automated commit message generation (CMG) — both in quality and speed — compared
to conventional autoregressive LLMs of the same scale.

The central pipeline under study is a **two-step** architecture:

1. **Summarisation** — Each changed file's diff is independently summarised
   (dLLM or LLM).
2. **Generation** — A single LLM synthesises the per-file summaries into a
   final commit message.

We compare this against a direct (one-step) baseline where the LLM/dLLM
generates the commit message straight from the raw diff.

---

## Overview of Pipelines

| Mode       | Summariser     | Generator     | Description                        |
|------------|----------------|---------------|------------------------------------|
| `dllm_llm` | dLLM (Fast-dLLM v2) | LLM (Qwen 2.5) | Primary experimental pipeline |
| `llm_llm`  | LLM (Qwen 2.5)      | LLM (Qwen 2.5) | Two-step autoregressive baseline |
| `llm_only` | —              | LLM (Qwen 2.5) | One-step: raw diff → commit msg    |
| `dllm_only`| —              | dLLM (Fast-dLLM v2) | One-step dLLM: raw diff → msg |

Models: **Qwen/Qwen2.5-7B-Instruct** (LLM), **Efficient-Large-Model/Fast_dLLM_v2_7B** (dLLM), INT8 quantisation, 2×T4 GPUs (16 GB each), model-parallel.

---

## RQ-a: Can dLLMs Replace the Autoregressive Summariser in a Two-Step CMG Pipeline Without Sacrificing Quality?

### Motivation

The two-step CMG pipeline (`Σ → Gen`) relies on an intermediate summarisation
stage that processes each changed file independently.  If a dLLM can produce
summaries of comparable quality, it opens the door to parallel decoding
advantages that autoregressive models fundamentally cannot exploit.

### Comparison

`dllm_llm` vs `llm_llm` — identical two-step architecture, only the summariser
differs.

### Metrics & Analysis

| Category | Metrics |
|----------|---------|
| **Quality** | BLEU-4, BLEU-NORM, BLEU-CODE, ROUGE-1/2/L, METEOR, CIDEr |
| **Speed** | `summary_wall_s`, `total_wall_s`, overall `tokens_per_second` |
| **Combined** | QS-Score = quality × log₂(1 + tok/s) |

### Preliminary Evidence (n = 1 200 tasks, 7B models)

| Metric | dllm_llm | llm_llm | Δ |
|--------|----------|---------|---|
| BLEU-4 (corpus) | 0.0085 | 0.0092 | −7.6% |
| BLEU-CODE | 0.1861 | 0.1544 | **+20.5%** |
| ROUGE-L | 0.152 | 0.156 | −2.8% |
| METEOR | 0.120 | 0.128 | −6.4% |
| CIDEr | 0.301 | 0.310 | −2.8% |
| Avg summary wall (s) | 25.73 | 15.90 | **1.62× slower** |
| Avg total wall (s) | 30.18 | 19.99 | 1.51× slower |
| QS-METEOR | 0.103 | 0.148 | −30.0% |

**Key observation:** dLLM summaries produce commit messages that are *better at
preserving code identifiers* (BLEU-CODE +20.5%) despite marginally lower scores
on surface n-gram metrics.  However, the dLLM summariser is currently **1.6×
slower** than the autoregressive one — making speed optimisation essential.

### What to Report

- Per-sample paired analysis (Wilcoxon signed-rank test, paired bootstrap).
- Qualitative error analysis: when does dLLM produce better/worse summaries?
- Impact of summary quality on downstream CMG quality (correlation).

---

## RQ-b: Does a Two-Step (Summary-Based) Pipeline Outperform Direct Diff-to-Message Generation?

### Motivation

Generating commit messages directly from the full diff avoids the
summarisation overhead but forces the model to handle long, noisy
inputs.  Two-step pipelines trade extra inference time for
information-dense, structured input to the generator.

### Comparison

Two-step modes (`dllm_llm`, `llm_llm`) vs one-step modes (`llm_only`, `dllm_only`).

### Metrics & Analysis

| Category | Metrics |
|----------|---------|
| **Quality** | Same full quality suite as RQ-a |
| **Speed** | `total_wall_s` — the one-step pipeline has zero `summary_wall_s` |
| **Combined** | QS-Score comparison |

### What to Report

- Quality delta between one-step and two-step, controlling for the same generator model.
- Break-even analysis: at what commit size (number of files / diff length) does the
  two-step pipeline overtake one-step on combined QS-Score?
- Robustness: does the two-step pipeline degrade more gracefully on very large diffs?

---

## RQ-c: How Do dLLM Generation Parameters Affect the Quality–Speed Trade-Off?

### Motivation

The block-diffusion architecture (Fast-dLLM v2) exposes several parameters that
directly control the quality–speed trade-off:

| Parameter | Role | Impact |
|-----------|------|--------|
| `block_size` | Size of each diffusion block (tokens decoded per block pass) | Larger → fewer blocks → fewer cache updates, but may hurt quality |
| `small_block_size` | Sub-block granularity for iterative refinement | Smaller → finer unmasking → more steps but potentially higher quality |
| `threshold` | Probability threshold for accepting an unmasked token | Lower → more tokens accepted per step → fewer steps → faster, but noisier |
| `use_block_cache` | Whether to reuse KV cache across small blocks | On → faster but uses more memory |
| `max_new_tokens` | Maximum generation length for summaries | Larger → allows longer summaries, more memory, more steps |
| `batch_size` | Number of file-diffs summarised in one dLLM forward pass | Larger → better GPU utilisation but more padding waste |

### Experiment Design

Grid search over key parameters using the 250-task dLLM evaluation set:

```
block_size:       [32, 64]
small_block_size: [8, 16]
threshold:        [0.8, 0.6]
use_block_cache:  [true, false]
max_new_tokens:   [256, 512, 1024]
batch_size:       [4, 8]
```

→ **192 configurations** × 50 sampled tasks = 9 600 runs.

### Metrics & Analysis

- **Primary**: Pareto frontier of (QS-Score, total_wall_s) across configurations.
- **Secondary**: Per-parameter sensitivity analysis — how much does each parameter
  contribute to quality and speed independently?
- **Derived**: `tokens_per_step` — measures effective parallelism.  The current
  average of **1.67 tok/step** suggests the dLLM is operating far below its
  theoretical parallel bandwidth.
- **Constraint**: GPU memory consumption — which configurations OOM on T4 16 GB?

### What to Report

- Pareto-optimal parameter sets.
- Sensitivity plots: each parameter vs quality, each parameter vs speed.
- Recommended "fast" configuration and "quality" configuration.
- Memory budget table per configuration.

---

## RQ-d: How Does Commit Complexity (Number of Changed Files) Affect Relative dLLM vs LLM Performance?

### Motivation

dLLMs use **block-parallel diffusion** — all masked tokens within a block can be
predicted simultaneously.  In the two-step pipeline, the dLLM summariser is
called once **per batch of files**.  Unlike autoregressive LLMs (which must
process each file sequentially), the dLLM's `batch_sample` can process multiple
files in a single forward pass, padding shorter prompts with MASK tokens.

**Hypothesis:** As the number of changed files in a commit grows, the dLLM's
batching ability should provide increasing throughput advantages because:
1. More files per batch → better GPU utilisation.
2. Batched diffusion amortises the iterative refinement cost.
3. Autoregressive LLMs scale linearly with file count (sequential calls).

The critical question is whether this theoretical advantage materialises in
practice, or whether padding waste and the per-step overhead of iterative
refinement negate it.

### Experiment Design

Stratify the evaluation set by commit complexity (number of changed files):

| Stratum | File count | Expected behaviour |
|---------|------------|-------------------|
| Small | 3–4 files | LLM likely faster (low batching benefit) |
| Medium | 5–8 files | Crossover region |
| Large | 9–16 files | dLLM batching should start to help |
| Very large | 17–35 files | Strong dLLM advantage expected |

For each stratum, compare:
- `dllm_llm` vs `llm_llm` summary wall time.
- Speedup ratio as a function of file count.
- Quality metrics — does quality degrade differently under high file count?

### Metrics & Analysis

- **Primary**: `summary_wall_s` speedup ratio (dLLM / LLM) stratified by file count.
- **Secondary**: Quality metrics per stratum (ensure quality parity holds at scale).
- **Derived**: `tokens_per_step` vs file count — does parallelism improve with batch fill?
- **Visualisation**: Speedup curve plotted against file count, with 1.0× reference line.

### What to Report

- Crossover point: at how many files does dLLM become faster than LLM?
- Scaling curve with confidence intervals.
- Per-stratum quality comparison tables.
- Discussion: practical implications (what fraction of real-world commits have enough files?).

---

## RQ-e: Can Block Diffusion's Parallel Decoding Achieve Wall-Clock Speedup Over Autoregressive Generation for Code Summarisation?

### Motivation

The Fast-dLLM v2 architecture uses **block diffusion** (see `modeling.py`):
- Tokens are decoded in blocks of `block_size` tokens.
- Within each block, tokens are iteratively unmasked across `small_block_size`
  sub-blocks via a confidence-based schedule.
- The model uses a specialised attention mask (`eval_block_diff_mask`) that
  enforces block-causal attention — each block can attend to all previous blocks
  but only within its own block internally.
- `use_block_cache` enables KV-cache reuse across sub-block refinement steps,
  avoiding redundant attention computation.

Theoretically, this should be faster than autoregressive generation because
multiple tokens are produced per forward pass.  However, preliminary results show
the dLLM averaging only **1.67 tokens/step** (with `threshold=0.8`,
`block_size=32`), which is far from the theoretical maximum of `small_block_size
= 8` tokens/step.

### Experiment Design

**Direct head-to-head timing benchmark** of dLLM `batch_sample` vs LLM
`model.generate()` on identical summarisation prompts.  This is a controlled
micro-benchmark, *not* an end-to-end pipeline comparison:

1. Select tasks stratified by file count (4, 8, 16, 20+ files).
2. For each task, time:
   - **dLLM batch**: all file-diffs in one `batch_sample` call (or batched into groups).
   - **dLLM sequential**: each file-diff in a separate `batch_sample(batch_size=1)` call.
   - **LLM sequential**: each file-diff in a separate `model.generate()` call.
3. Sweep dLLM parameters (`block_size`, `small_block_size`, `threshold`) to find
   the configuration that maximises speedup over the LLM baseline.

### Key Architecture Details (from `modeling.py`)

```
Block Diffusion Attention:
  M_BD  — Block-diagonal: self-attention within noised blocks
  M_OBC — Offset block-causal: cross-attention for conditional context
  M_BC  — Block-causal: attention to update x₀

Inference path (eval mode):
  eval_block_diff_mask:  block_q >= block_kv  (block-causal)
  Uses SDPA attention with sliding_window support
  block_past_key_values for cross-block KV cache reuse
```

### Metrics & Analysis

| Metric | Description |
|--------|-------------|
| `wall_s` (total) | Wall-clock time for all files in a task |
| `wall_s / file` | Amortised per-file time |
| `tokens_per_step` | Effective parallelism (dLLM only) |
| `tokens_per_second` | Raw throughput for both models |
| `speedup` | `LLM_sequential_wall / dLLM_batch_wall` |
| `padding_ratio` | Wasted MASK tokens from length mismatch in dLLM batch |
| `GPU memory (peak)` | Memory high-water mark |

### What to Report

- Speedup table: dLLM-batch vs LLM-sequential across file counts and parameter configs.
- Breakdown: how much of the dLLM's time is spent on useful unmasking vs overhead?
- Optimal dLLM configuration per file-count stratum.
- `tokens_per_step` distribution — identify whether the threshold or block_size
  is the bottleneck for parallelism.
- Practical recommendation: when to use dLLM vs LLM for the summarisation stage.

---

## RQ-f: Does the dLLM Summariser Produce More Code-Aware Summaries?

### Motivation

The preliminary results show a striking **+20.5% improvement in BLEU-CODE** for
`dllm_llm` over `llm_llm`, despite marginal losses on surface-level n-gram
metrics (BLEU-4, ROUGE-L, METEOR).  BLEU-CODE uses a tokenisation scheme that
preserves code identifiers (camelCase splitting, snake_case splitting, etc.),
suggesting that dLLM summaries better retain the *code-relevant content* that
matters for developer-facing commit messages.

### Hypothesis

The block-diffusion architecture's ability to revise all tokens within a block
simultaneously may help it maintain coherence over code-like tokens (which are
often long, compositional identifiers like `TaskSchedulerManager` or
`handleAuthCallback`).  Autoregressive models must commit to each sub-token
sequentially, risking early-commitment errors on long identifiers.

### Analysis

- BLEU-CODE breakdown per programming language (Java, C++, Python, etc.).
- Identifier preservation rate: what fraction of code identifiers from the diff
  appear (exact or stemmed match) in the generated summary?
- Qualitative examples: pairs where dLLM preserves identifiers that LLM drops.
- Correlation between BLEU-CODE improvement and diff code-density.

### What to Report

- Per-language BLEU-CODE comparison.
- Identifier preservation analysis with statistical testing.
- Concrete examples showing code-aware vs surface-level summarisation.
- Discussion: why might block diffusion be better at code token generation?

---

## Summary of All Research Questions

| RQ | Question | Key Comparison |
|----|----------|---------------|
| **RQ-a** | Can dLLMs replace the AR summariser without quality loss? | `dllm_llm` vs `llm_llm` |
| **RQ-b** | Does two-step (summary) outperform one-step (raw diff)? | 2-step vs 1-step modes |
| **RQ-c** | How do dLLM parameters affect the quality–speed trade-off? | Grid search over 192 configs |
| **RQ-d** | How does commit complexity affect relative dLLM vs LLM perf? | Stratified by file count |
| **RQ-e** | Can block diffusion achieve wall-clock speedup over AR? | Micro-benchmark: dLLM vs LLM |
| **RQ-f** | Does the dLLM produce more code-aware summaries? | BLEU-CODE + identifier analysis |

---

## Evaluation Framework

### Quality Metrics

| Metric | Description |
|--------|-------------|
| BLEU-4 | Standard 4-gram precision with brevity penalty (smoothed) |
| BLEU-NORM | BLEU-4 on normalised (lowercased, stemmed) tokens |
| BLEU-CODE | BLEU-4 with code-identifier-aware tokenisation (camelCase/snake_case split) |
| ROUGE-1/2/L | Recall-oriented unigram/bigram/longest-common-subsequence overlap |
| METEOR | Alignment-based score with stemming and synonym matching |
| CIDEr | TF-IDF weighted cosine similarity × 10 (corpus-aware) |

### Speed Metrics

| Metric | Description |
|--------|-------------|
| `total_wall_s` | End-to-end wall time per task |
| `summary_wall_s` | Time spent in the summarisation stage |
| `cmg_wall_s` | Time spent generating the final commit message |
| `tokens_per_second` | Raw throughput (gen tokens / wall time) |
| `tokens_per_step` | dLLM-specific: tokens unmasked per diffusion step |

### Combined Metric

**QS-Score** (Quality–Speed Score):

$$\text{QS}(q, s) = q \times \log_2(1 + s)$$

where $q$ is any quality metric and $s$ is tokens per second.  This rewards
models that achieve high quality *and* high throughput.

---

## Experimental Setup

- **Dataset**: 5 000 tasks from ApacheCM (multi-file commits, 3–35 files)
- **Main evaluation**: 1 200 overlapping tasks between machine 1 and machine 2
- **Grid search**: 250 selected dLLM-favourable tasks, 50-task sample per config
- **Hardware**: 2× NVIDIA T4 (16 GB) per machine, model-parallel
- **Models**: 7B parameter scale (Qwen 2.5-7B-Instruct + Fast-dLLM v2 7B, INT8)
- **Reproducibility**: Fixed seeds, `temperature=0.0`, `do_sample=False`
