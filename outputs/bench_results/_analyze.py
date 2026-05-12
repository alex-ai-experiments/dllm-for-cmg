import json

data = json.load(open('outputs/bench_results/grid_3tasks.json', encoding='utf-8'))
print(f'Total entries: {len(data)}\n')

hdr = f"  {'task_id':<35} {'bs':>4} {'sbs':>4} {'th':>5} {'mnt':>5}  {'sp@2':>7} {'sp@4':>7}  {'seq_avg_s':>10} {'steps@2':>8} {'steps@4':>8}"
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
for r in data:
    b2 = r['batches'].get('2', {})
    b4 = r['batches'].get('4', {})
    sp2 = 'OOM' if b2.get('oom') else (f"{b2['speedup']:.2f}x" if b2.get('speedup') else '-')
    sp4 = 'OOM' if b4.get('oom') else (f"{b4['speedup']:.2f}x" if b4.get('speedup') else '-')
    steps2 = b2.get('steps', '-')
    steps4 = b4.get('steps', '-')
    seq_avg = r['seq_total'] / len(r['seq_times']) if r['seq_times'] else 0
    print(f"  {r['task_id']:<35} {r['block_size']:>4} {r['small_block_size']:>4} {r['threshold']:>5.2f} {r['max_new_tokens']:>5}  {sp2:>7} {sp4:>7}  {seq_avg:>10.2f}s {str(steps2):>8} {str(steps4):>8}")

print()
# Best per task
print('--- Best speedup @ bs=4 per task ---')
by_task = {}
for r in data:
    tid = r['task_id']
    sp4 = r['batches'].get('4', {}).get('speedup') or 0
    if tid not in by_task or sp4 > by_task[tid][0]:
        by_task[tid] = (sp4, r['block_size'], r['small_block_size'], r['threshold'], r['max_new_tokens'])
for tid, (sp, bs, sbs, th, mnt) in by_task.items():
    print(f"  {tid:<35}  {sp:.3f}x  bs={bs} sbs={sbs} th={th} mnt={mnt}")

print()
# Quality sample: first seq text from first entry per task
print('--- Sample sequential summary (first file, best-speed config per task) ---')
best_configs = {}
for r in data:
    tid = r['task_id']
    sp4 = r['batches'].get('4', {}).get('speedup') or 0
    if tid not in best_configs or sp4 > best_configs[tid][0]:
        best_configs[tid] = (sp4, r)
for tid, (sp, r) in best_configs.items():
    print(f"\n  [{tid}] (bs={r['block_size']} sbs={r['small_block_size']} th={r['threshold']})")
    print(f"  {r['seq_texts'][0][:300]}")
