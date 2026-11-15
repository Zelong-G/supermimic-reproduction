"""Evaluate the new model; compare with archived exp006 without rerunning it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'my_work/outputs/exp_007_full_training/full_seed0/evaluation'
BASE = ROOT / 'my_work/outputs/exp_006_bones_benchmark'
CKPT = 'my_work/outputs/exp_007_full_training/full_seed0/ckpt_latest.pt'
SETS = {'rand2000': (2000, '28dfbc14f02146cc05fe62c794d89f01'),
        'hard': (130, 'b931bb29b8c786bfdfa34b57c0832caf')}


def digest(path, algorithm):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def analyze():
    rows = []
    failures = []
    for stage in SETS:
        path = OUT / f'{stage}.json'
        if not path.exists():
            continue
        baseline = json.loads((BASE / f'{stage}.json').read_text())
        trained = json.loads(path.read_text())
        a = {c['name']: c for c in baseline['clips']}
        b = {c['name']: c for c in trained['clips']}
        if a.keys() != b.keys() or any(a[k]['frames'] != b[k]['frames'] for k in a):
            raise ValueError(f'{stage}: evaluation units differ')
        for model, result in [('community_exp006', baseline), ('trained_exp007', trained)]:
            s = result['summary']
            rows.append(dict(stage=stage, model=model, clips=s['clips'],
                             tracking_score=s['mean_norm_return'], failed=s['failed'],
                             failure_rate=s['failed']/s['clips'],
                             overflow_clips=sum(c.get('overflow_steps', 0)>0 for c in result['clips']),
                             overflow_steps=sum(c.get('overflow_steps', 0) for c in result['clips'])))
            failures += [dict(stage=stage, model=model, motion=c['name'],
                              fail_step=c['fail_step'], overflow_steps=c.get('overflow_steps', 0))
                         for c in result['clips'] if c['failed']]
    if not rows:
        raise SystemExit('No completed new-model benchmark results.')
    for name, items, fields in [('bones_comparison.csv', rows, list(rows[0])),
                                 ('bones_failures.csv', failures,
                                  ['stage','model','motion','fail_step','overflow_steps'])]:
        with (OUT/name).open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(items)
    report = ['# Experiment 007: BONES-SEED Benchmark', '',
              'Community results are reused from Experiment 006; community weights were not rerun.',
              'BONES-SEED is training-related data, not held-out generalization. Raw policy weights; same community evaluator.', '',
              '| Set | Model | Tracking score | Failed / clips | Failure rate | Overflow clips / steps |',
              '|---|---|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"| {r['stage']} | {r['model']} | {r['tracking_score']:.6f} | {r['failed']} / {r['clips']} | {r['failure_rate']:.4%} | {r['overflow_clips']} / {r['overflow_steps']} |")
    report += ['', 'Clip names and frame counts match the archived baselines. Solver capacity overflow and run-to-run variation limit causal interpretation. The new runs share the GPU with another evaluation (recorded in run metadata). No full-set benchmark was started.', '',
               'Exact commands, list/checkpoint hashes, timestamps, code commit, GPU processes and exit codes are in each *_run.json. See bones_failures.csv for failed motions.']
    if failures:
        report += ['', '## Failed motions', '']
        report += [f"- {r['stage']} / {r['model']}: {r['motion']}, failure step {r['fail_step']}, overflow steps {r['overflow_steps']}." for r in failures]
    (OUT/'BONES_REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


def run(stage):
    out = OUT / f'{stage}.json'
    log = OUT / f'{stage}.log'
    meta_path = OUT / f'{stage}_run.json'
    if any(p.exists() for p in (out, log, meta_path)):
        raise SystemExit(f'Refusing to overwrite existing {stage} output/log/metadata.')
    expected_count, expected_md5 = SETS[stage]
    clip_list = ROOT / f'data/bench/{stage}.txt'
    names = clip_list.read_text().splitlines()
    if len(names) != expected_count or digest(clip_list, 'md5') != expected_md5:
        raise SystemExit('Benchmark list count or MD5 mismatch.')
    baseline = json.loads((BASE/f'{stage}.json').read_text())
    if {c['name'] for c in baseline['clips']} != set(names):
        raise SystemExit('Baseline clip names do not match list.')
    data = ROOT / 'data/npz/bones_seed'
    if any(not (data/f'{name}.npz').is_file() for name in names):
        raise SystemExit('Missing NPZ data; not starting evaluation.')
    command = [sys.executable, '-u', 'scripts/eval_benchmark.py', '--ckpt', CKPT,
               '--npz-dir', 'data/npz/bones_seed', '--clip-list', str(clip_list.relative_to(ROOT)),
               '--num-envs', '1024', '--chunk-size', '2000', '--device', 'cuda:0',
               '--out', str(out.relative_to(ROOT))]
    meta = dict(status='running', command=command, started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                checkpoint_sha256=digest(ROOT/CKPT, 'sha256'), list_md5=expected_md5,
                community_evaluator_sha256=digest(ROOT/'scripts/eval_benchmark.py', 'sha256'),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                gpu_before=subprocess.check_output(['nvidia-smi'],text=True), community_rerun=False)
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    start=time.monotonic()
    print('Starting', stage, flush=True)
    with log.open('x') as f:
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    meta.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,
                elapsed_seconds=time.monotonic()-start,
                finished_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    if result.returncode:
        raise SystemExit(f'Evaluation failed; see {log}')
    analyze()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=[*SETS, 'analyze'])
    args=parser.parse_args()
    if args.stage=='analyze':
        analyze()
    else:
        run(args.stage)
