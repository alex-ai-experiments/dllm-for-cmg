"""
Analyze tasks_profiling.jsonl and select:
  - 50 best tasks for profiling (speed benchmarking)
  - 500 best tasks for evaluation (quality + speed)

Selection criteria:
  - Profiling: maximize file-count diversity, prefer balanced per-file diff lengths
  - Evaluation: broad coverage, good label quality, diverse repos
"""
import json
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from diff_utils import get_per_file_diffs

# ── Load ──────────────────────────────────────────────────────────────────────
tasks = [json.loads(l) for l in open("build_tasks/tasks_profiling.jsonl") if l.strip()]
print(f"Total tasks loaded: {len(tasks)}")

# ── Compute per-task features ─────────────────────────────────────────────────
task_features = []
for t in tasks:
    file_diffs = get_per_file_diffs(t)
    n_files = len(file_diffs)
    per_file_lens = [len(fd) for _, fd in file_diffs]
    
    # Label quality heuristics
    label = t["label"]
    label_words = len(label.split())
    
    # Extract repo from task_id (apache_<project>_<sha>)
    parts = t["task_id"].split("_")
    repo = parts[1] if len(parts) >= 3 else "unknown"
    
    # Per-file length balance (lower CV = more balanced = better for batching)
    if n_files > 1 and statistics.mean(per_file_lens) > 0:
        cv = statistics.stdev(per_file_lens) / statistics.mean(per_file_lens)
    else:
        cv = 0.0
    
    task_features.append({
        "task_id": t["task_id"],
        "n_files": n_files,
        "diff_length": t["diff_length"],
        "per_file_lens": per_file_lens,
        "mean_file_len": statistics.mean(per_file_lens) if per_file_lens else 0,
        "cv_file_len": cv,  # coefficient of variation
        "label": label,
        "label_words": label_words,
        "repo": repo,
        "files": t.get("files", []),
    })

# ── Distribution analysis ─────────────────────────────────────────────────────
fc = Counter(tf["n_files"] for tf in task_features)
print("\n═══ FILE COUNT DISTRIBUTION ═══")
for k in sorted(fc.keys()):
    bar = "█" * min(fc[k] // 10, 50)
    print(f"  {k:>2} files: {fc[k]:>4} tasks  {bar}")

# Available for profiling strata
print("\n═══ REALISTIC PROFILING STRATA ═══")
print("Given the distribution, realistic strata for profiling (≥5 tasks each):")
strata_candidates = [k for k in sorted(fc.keys()) if fc[k] >= 5]
print(f"  Available: {strata_candidates}")
print(f"  Max file count with ≥5 tasks: {max(strata_candidates) if strata_candidates else 0}")

# Propose strata: cover the range as evenly as possible
# We'll use: 1, 2, 3, 4, 5, 6, 7, 8, 9-10, 11+
# But for profiling SPEED (batching), single-file tasks aren't interesting
# Focus on 2+ files since batching only helps with multiple files
proposed_profiling_strata = []
# Must have at least 5 tasks
for n in sorted(fc.keys()):
    if n >= 2 and fc[n] >= 5:
        proposed_profiling_strata.append(n)

# For high file counts with < 5 tasks, group them
high_file_group = [tf for tf in task_features if tf["n_files"] >= 8]
print(f"\n  Tasks with ≥8 files: {len(high_file_group)}")
for tf in sorted(high_file_group, key=lambda x: x["n_files"]):
    print(f"    {tf['task_id']} → {tf['n_files']} files  (diff_len={tf['diff_length']}, CV={tf['cv_file_len']:.2f})")

# ── SCORING for profiling candidates ──────────────────────────────────────────
# For profiling, we want:
# 1. Low CV (balanced per-file lengths → less padding waste)
# 2. Moderate diff_length (not extremes) 
# 3. Meaningful label (≥3 words)
# 4. Repo diversity within each stratum

def profiling_score(tf):
    """Higher = better candidate for profiling."""
    score = 0.0
    # Prefer balanced file lengths (low CV)
    score -= tf["cv_file_len"] * 2.0
    # Prefer mid-range diff lengths (penalize extremes)
    mid = 5500
    score -= abs(tf["diff_length"] - mid) / 3000.0
    # Prefer meaningful labels
    if tf["label_words"] >= 4:
        score += 0.5
    elif tf["label_words"] >= 3:
        score += 0.3
    # Slight bonus for more files (rarer, more interesting for batching)
    score += tf["n_files"] * 0.1
    return score

# ── SELECT PROFILING TASKS (50) ───────────────────────────────────────────────
print("\n═══ SELECTING 50 PROFILING TASKS ═══")

# Strategy: even spread across file counts to measure speedup scaling
# Strata: 2, 3, 4, 5, 6, 7, 8, 9, 10+, extreme(13-21)
# 5 per stratum for 2-9, then fill remaining from 10+ and extreme

profiling_set = []
used_ids = set()

# Define strata: each gets 5 tasks (or fewer if not enough available)
profiling_strata = [2, 3, 4, 5, 6, 7, 8, 9]

for n_files in profiling_strata:
    candidates = [tf for tf in task_features 
                  if tf["n_files"] == n_files and tf["task_id"] not in used_ids]
    candidates.sort(key=profiling_score, reverse=True)
    
    # Select with repo diversity
    n_select = min(5, len(candidates))
    selected = []
    repos_used = set()
    
    # First pass: one per repo (diversity)
    for tf in candidates:
        if len(selected) >= n_select:
            break
        if tf["repo"] not in repos_used:
            selected.append(tf)
            repos_used.add(tf["repo"])
            used_ids.add(tf["task_id"])
    
    # Second pass: fill remaining from best scores
    for tf in candidates:
        if len(selected) >= n_select:
            break
        if tf["task_id"] not in used_ids:
            selected.append(tf)
            used_ids.add(tf["task_id"])
    
    profiling_set.extend(selected)
    print(f"  {n_files} files: selected {len(selected)} tasks (from {len(candidates)+len(selected)} available, {len(repos_used)} repos)")

# Now add the 10+ file tasks (these are the rarest / most valuable for batching)
high_strata = [10, 11, 12, 13, 15, 17, 18, 21]
remaining_slots = 50 - len(profiling_set)
high_candidates = [tf for tf in task_features 
                   if tf["n_files"] >= 10 and tf["task_id"] not in used_ids]
high_candidates.sort(key=profiling_score, reverse=True)

# Take all of them (they're rare) up to remaining slots
for tf in high_candidates[:remaining_slots]:
    profiling_set.append(tf)
    used_ids.add(tf["task_id"])
print(f"  10+ files: selected {min(len(high_candidates), remaining_slots)} tasks (from {len(high_candidates)} available)")

print(f"\n  TOTAL PROFILING: {len(profiling_set)} tasks")
print(f"  File count breakdown: {Counter(tf['n_files'] for tf in profiling_set)}")

# ── SELECT EVALUATION TASKS (500, includes the 50 profiling ones) ─────────────
print("\n═══ SELECTING 500 EVALUATION TASKS ═══")

# For evaluation we want:
# 1. Good label quality (meaningful commit messages to compare against)
# 2. Diverse repos and file counts
# 3. Moderate complexity (not trivially short diffs)
# 4. Include all 50 profiling tasks

def eval_score(tf):
    """Higher = better candidate for evaluation."""
    score = 0.0
    # Label quality is paramount for evaluation
    if tf["label_words"] >= 5:
        score += 2.0
    elif tf["label_words"] >= 4:
        score += 1.5
    elif tf["label_words"] >= 3:
        score += 1.0
    elif tf["label_words"] <= 1:
        score -= 2.0  # single-word labels are poor evaluation targets
    
    # Prefer moderate diff lengths
    if 3000 <= tf["diff_length"] <= 7000:
        score += 0.5
    
    # Slight preference for multi-file tasks (more realistic commits)
    score += min(tf["n_files"], 5) * 0.2
    
    # Penalize very short labels that might be generic
    if len(tf["label"]) < 10:
        score -= 1.0
    
    return score

# Start with profiling set
eval_ids = set(tf["task_id"] for tf in profiling_set)
eval_set = list(profiling_set)

# Score remaining tasks
remaining = [tf for tf in task_features if tf["task_id"] not in eval_ids]
remaining.sort(key=eval_score, reverse=True)

# Select with stratified sampling to maintain file-count diversity
# Target distribution for eval: proportional but boost rare high-file-count tasks
target_eval = 500
slots_left = target_eval - len(eval_set)

# Reserve slots for each file count proportionally (but minimum 5 for rare ones)
fc_remaining = Counter(tf["n_files"] for tf in remaining)
total_remaining = len(remaining)
file_count_quotas = {}
for n_files in sorted(fc_remaining.keys()):
    # Proportional + bonus for multi-file
    base = max(5, int(slots_left * fc_remaining[n_files] / total_remaining))
    # Boost multi-file tasks
    if n_files >= 4:
        base = int(base * 1.5)
    file_count_quotas[n_files] = min(base, fc_remaining[n_files])

# Normalize to fit in slots_left
total_quota = sum(file_count_quotas.values())
if total_quota > slots_left:
    scale = slots_left / total_quota
    file_count_quotas = {k: max(1, int(v * scale)) for k, v in file_count_quotas.items()}

# Fill by stratum
for n_files in sorted(file_count_quotas.keys()):
    quota = file_count_quotas[n_files]
    candidates = [tf for tf in remaining if tf["n_files"] == n_files and tf["task_id"] not in eval_ids]
    candidates.sort(key=eval_score, reverse=True)
    
    # Diversify repos
    selected = []
    repos_used = set()
    for tf in candidates:
        if len(selected) >= quota:
            break
        if tf["repo"] not in repos_used or len(selected) >= quota // 2:
            selected.append(tf)
            repos_used.add(tf["repo"])
            eval_ids.add(tf["task_id"])
    
    eval_set.extend(selected)

# If we're short, fill from best remaining
if len(eval_set) < target_eval:
    extra_needed = target_eval - len(eval_set)
    extra_candidates = [tf for tf in remaining if tf["task_id"] not in eval_ids]
    extra_candidates.sort(key=eval_score, reverse=True)
    for tf in extra_candidates[:extra_needed]:
        eval_set.append(tf)
        eval_ids.add(tf["task_id"])

# If we're over, trim from lowest-scored single-file tasks
while len(eval_set) > target_eval:
    # Remove lowest-scored 1-file task
    single_file = [(i, tf) for i, tf in enumerate(eval_set) 
                   if tf["n_files"] == 1 and tf["task_id"] not in set(p["task_id"] for p in profiling_set)]
    if single_file:
        single_file.sort(key=lambda x: eval_score(x[1]))
        idx = single_file[0][0]
        eval_set.pop(idx)
    else:
        eval_set.pop()

print(f"  TOTAL EVALUATION: {len(eval_set)} tasks")
eval_fc = Counter(tf["n_files"] for tf in eval_set)
print(f"  File count breakdown:")
for k in sorted(eval_fc.keys()):
    print(f"    {k:>2} files: {eval_fc[k]:>3} tasks")

# Verify profiling tasks are in eval set
prof_ids = set(tf["task_id"] for tf in profiling_set)
eval_ids_final = set(tf["task_id"] for tf in eval_set)
assert prof_ids.issubset(eval_ids_final), "Profiling tasks must be in eval set!"
print(f"\n  ✓ All {len(profiling_set)} profiling tasks included in evaluation set")

# ── Repo diversity stats ──────────────────────────────────────────────────────
prof_repos = Counter(tf["repo"] for tf in profiling_set)
eval_repos = Counter(tf["repo"] for tf in eval_set)
print(f"\n  Profiling repos ({len(prof_repos)}): {dict(prof_repos.most_common(10))}")
print(f"  Evaluation repos ({len(eval_repos)}): top 10 = {dict(eval_repos.most_common(10))}")

# ── Print profiling task details ──────────────────────────────────────────────
print("\n═══ PROFILING TASKS (50) — DETAILS ═══")
profiling_set.sort(key=lambda tf: (tf["n_files"], tf["task_id"]))
for tf in profiling_set:
    print(f"  {tf['task_id']:<40} {tf['n_files']:>2} files  diff={tf['diff_length']:>5}  "
          f"CV={tf['cv_file_len']:.2f}  label=\"{tf['label'][:60]}\"")

# ── Save outputs ──────────────────────────────────────────────────────────────
# Save profiling task IDs
profiling_ids = [tf["task_id"] for tf in profiling_set]
eval_task_ids = [tf["task_id"] for tf in eval_set]

output = {
    "profiling_task_ids": profiling_ids,
    "eval_task_ids": eval_task_ids,
    "stats": {
        "total_source": len(tasks),
        "profiling_count": len(profiling_set),
        "eval_count": len(eval_set),
        "profiling_file_counts": dict(Counter(tf["n_files"] for tf in profiling_set)),
        "eval_file_counts": dict(Counter(tf["n_files"] for tf in eval_set)),
    }
}

with open("build_tasks/selected_tasks.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: build_tasks/selected_tasks.json")

# Also save the actual JSONL subsets
task_lookup = {t["task_id"]: t for t in tasks}

with open("build_tasks/tasks_profiling_50.jsonl", "w", encoding="utf-8") as f:
    for tid in profiling_ids:
        f.write(json.dumps(task_lookup[tid]) + "\n")
print(f"Saved: build_tasks/tasks_profiling_50.jsonl ({len(profiling_ids)} tasks)")

with open("build_tasks/tasks_eval_500.jsonl", "w", encoding="utf-8") as f:
    for tid in eval_task_ids:
        f.write(json.dumps(task_lookup[tid]) + "\n")
print(f"Saved: build_tasks/tasks_eval_500.jsonl ({len(eval_task_ids)} tasks)")
