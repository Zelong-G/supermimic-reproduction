"""Analyze existing LAFAN1 results; no simulator, checkpoint loading or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'my_work'
DEFAULT_FULL = WORK / 'outputs/exp_001_lafan1_full/lafan1_full.json'
DEFAULT_SEG = WORK / 'outputs/exp_002_lafan1_seg10/lafan1_seg10.json'
DEFAULT_OUT = WORK / 'outputs/exp_003_failure_analysis'
SEGMENT = re.compile(r'(.+)__seg(\d+)$')
MOTION = re.compile(r'([A-Za-z]+)\d+_subject\d+$')
LABELS = {'dance': 'Dance', 'fallAndGetUp': 'Fall and get up', 'fight': 'Fight',
          'fightAndSports': 'Fight and sports', 'jumps': 'Jumps', 'run': 'Run',
          'sprint': 'Sprint', 'walk': 'Walk'}


def category(name):
    match = MOTION.fullmatch(name)
    if not match or match[1] not in LABELS:
        raise ValueError(f'Unrecognized LAFAN1 motion: {name}')
    return match[1]


def split_segment(name):
    match = SEGMENT.fullmatch(name)
    if not match:
        raise ValueError(f'Invalid segment name: {name}')
    return match[1], int(match[2])


def aggregate(clips):
    n = len(clips)
    frames = sum(c['frames'] for c in clips)
    return {'clips': n, 'failed': sum(c['failed'] for c in clips),
            'failure_rate': sum(c['failed'] for c in clips) / n,
            'mean_norm_return': sum(c['norm_return'] for c in clips) / n,
            'step_weighted_norm_return': sum(c['norm_return'] * c['frames'] for c in clips) / frames,
            'mean_norm_return_survived_v1': sum(c['norm_return_survived'] for c in clips) / n,
            'failed_at_end': sum(c['failed'] and c['fail_step'] >= c['frames'] - 3 for c in clips),
            'frames': frames, 'overflow_steps': sum(c['overflow_steps'] for c in clips),
            'overflow_clips': sum(c['overflow_steps'] > 0 for c in clips)}


def load_results(path):
    data = json.loads(path.read_text())
    clips = data['clips']
    if not clips or len({c['name'] for c in clips}) != len(clips):
        raise ValueError(f'{path}: empty or duplicate clips')
    for c in clips:
        if type(c['failed']) is not bool or not 0 < c['steps'] <= c['frames']:
            raise ValueError(f'Invalid result: {c["name"]}')
        if c['failed'] and c['fail_step'] != c['steps']:
            raise ValueError(f'Invalid failure step: {c["name"]}')
        if not c['failed'] and (c['fail_step'] != -1 or c['steps'] != c['frames']):
            raise ValueError(f'Invalid completion: {c["name"]}')
        for key in ('norm_return', 'norm_return_survived'):
            if not math.isfinite(c[key]):
                raise ValueError(f'Nonfinite {key}: {c["name"]}')
    stats = aggregate(clips)
    for key in ('clips', 'failed', 'failed_at_end', 'mean_norm_return',
                'step_weighted_norm_return', 'mean_norm_return_survived_v1'):
        if not math.isclose(stats[key], data['summary'][key], abs_tol=1e-10):
            raise ValueError(f'{path}: stored summary differs from clip data: {key}')
    return data


def match_segments(full, segments, segment_frames=500):
    originals = {c['name']: c for c in full}
    grouped = defaultdict(list)
    for c in segments:
        source, index = split_segment(c['name'])
        if source not in originals:
            raise ValueError(f'Unmatched source: {source}')
        grouped[source].append((index, c))
    rows, segment_rows = [], []
    for name, original in originals.items():
        ordered = sorted(grouped[name])
        expected = [(i, min(segment_frames, original['frames'] - start))
                    for i, start in enumerate(range(0, original['frames'], segment_frames))
                    if min(segment_frames, original['frames'] - start) >= segment_frames // 2]
        if [(i, c['frames']) for i, c in ordered] != expected:
            raise ValueError(f'{name}: missing/duplicate segments or unexpected lengths')
        cs = [c for _, c in ordered]
        stats = aggregate(cs)
        rows.append({'name': name, 'category': category(name),
                     'full_frames': original['frames'], 'full_failed': original['failed'],
                     'full_fail_step': original['fail_step'], 'full_score': original['norm_return'],
                     'full_overflow_steps': original['overflow_steps'],
                     'segment_overflow_steps': stats['overflow_steps'],
                     'segment_overflow_clips': stats['overflow_clips'],
                     'segments': len(cs), 'segment_failures': stats['failed'],
                     'segment_failure_rate': stats['failure_rate'],
                     'segment_mean_score': stats['mean_norm_return'],
                     'segment_frame_weighted_score': stats['step_weighted_norm_return'],
                     'score_difference': stats['step_weighted_norm_return'] - original['norm_return'],
                     'covered_frames': stats['frames'],
                     'omitted_tail_frames': original['frames'] - stats['frames']})
        for index, c in ordered:
            segment_rows.append({'name': c['name'], 'source_motion': name, 'category': category(name),
                                 'segment_index': index, 'source_start_frame': index * segment_frames,
                                 'source_stop_frame_exclusive': index * segment_frames + c['frames'],
                                 'frames': c['frames'], 'failed': c['failed'],
                                 'fail_step': c['fail_step'], 'norm_return': c['norm_return'],
                                 'overflow_steps': c['overflow_steps'],
                                 'source_step_offset_at_failure': index * segment_frames + c['fail_step']
                                 if c['failed'] else None})
    return rows, segment_rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_all(full, seg, rows, categories, out):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fs, ss = aggregate(full), aggregate(seg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    metrics = ['mean_norm_return', 'step_weighted_norm_return']
    for offset, stats, label, color in [(-.18, fs, f'Full sequences (n={fs["clips"]})', '#3b6ea8'),
                                       (.18, ss, f'Independent segments (n={ss["clips"]})', '#e39b36')]:
        bars = axes[0].bar([i + offset for i in range(2)], [stats[k] for k in metrics], .36,
                           label=label, color=color)
        axes[0].bar_label(bars, fmt='%.3f', padding=3)
    axes[0].set(xticks=[0, 1], xticklabels=['Equal clip weight', 'Reference-frame weight'],
                ylim=(0, 1.12), ylabel='Normalized return', title='Tracking score (different protocols)')
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(['Full sequences', '10-second segments'],
                       [100 * fs['failure_rate'], 100 * ss['failure_rate']], color=['#3b6ea8', '#e39b36'])
    axes[1].bar_label(bars, labels=[f'{s["failed"]}/{s["clips"]} ({100*s["failure_rate"]:.2f}%)'
                                  for s in [fs, ss]], padding=4)
    axes[1].set(ylim=(0, max(fs['failure_rate'], ss['failure_rate'])*100+10), ylabel='Failed evaluation units (%)', title='Sequence and segment units differ')
    save_figure(fig, out, '01_protocol_comparison')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout='constrained')
    y = list(range(len(categories)))
    for offset, prefix, label, color in [(-.18, 'full', 'Full sequences', '#3b6ea8'),
                                       (.18, 'segment', '10-second segments', '#e39b36')]:
        counts = [r[f'{prefix}_samples'] for r in categories]
        failed = [r[f'{prefix}_failures'] for r in categories]
        rates = [100 * f/n for f, n in zip(failed, counts)]
        for ax, values in zip(axes, [counts, failed, rates]):
            bars = ax.barh([i + offset for i in y], values, .36, color=color, label=label)
            ax.bar_label(bars, fmt='%.1f' if ax is axes[2] else '%d', padding=2, fontsize=8)
    for ax, title in zip(axes, ['Sample counts', 'Failure counts', 'Failure rates (%)']):
        ax.set(yticks=y, yticklabels=[LABELS[r['category']] for r in categories], title=title)
        ax.invert_yaxis()
        ax.margins(x=.22)
    axes[0].legend(fontsize=8)
    fig.suptitle('Name-derived LAFAN1 categories; each protocol has a different evaluation unit')
    save_figure(fig, out, '02_category_distribution')

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for failed, marker, label in [(False, 'o', 'Full sequence completed'), (True, 'X', 'Full sequence failed')]:
        chosen = [r for r in rows if r['full_failed'] == failed]
        points = ax.scatter([r['full_score'] for r in chosen],
                            [r['segment_frame_weighted_score'] for r in chosen],
                            c=[100*r['segment_failure_rate'] for r in chosen],
                            norm=Normalize(0, 100), cmap='viridis_r', marker=marker, s=70, label=label)
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Full-sequence normalized return',
           ylabel='Within-motion frame-weighted segment return', title=f'Matched source motions (n={len(rows)})')
    ax.legend(loc='lower right', fontsize=8)
    fig.colorbar(points, ax=ax, label='Failed segments within motion (%)')
    save_figure(fig, out, '03_matched_motion_scores')

    # A missing trailing cell is not a passing segment; show it in gray.
    import numpy as np
    ordered = sorted(rows, key=lambda r: (r['category'], r['name']))
    width = max(r['segments'] for r in rows)
    values = np.full((len(ordered), width), np.nan)
    failures = []
    lookup = {r['name']: i for i, r in enumerate(ordered)}
    for c in seg:
        name, index = split_segment(c['name']); row = lookup[name]
        values[row, index] = c['norm_return']
        if c['failed']: failures.append((index, row))
    fig, ax = plt.subplots(figsize=(12, 12), layout='constrained')
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
    im = ax.imshow(values, vmin=0, vmax=1, aspect='auto', cmap=cmap)
    ax.scatter([x for x, _ in failures], [y for _, y in failures], marker='x', color='red', s=20,
               linewidths=.9, label='Failed segment')
    ax.set(yticks=range(len(ordered)), yticklabels=[r['name'] for r in ordered],
           xlabel='Segment index (start time = index x 10 seconds)',
           title='Segment scores and failures by source motion (gray = no segment)')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, label='Segment normalized return', shrink=.65)
    save_figure(fig, out, '04_segment_score_map')


def select_cases(rows, segments):
    cases = []
    groups = [
        ('full_success_all_segments_success', [r for r in rows if not r['full_failed'] and not r['segment_failures']],
         lambda r: -r['full_score'], 'Highest full score among motions passing both protocols.'),
        ('full_failure_all_segments_success', [r for r in rows if r['full_failed'] and not r['segment_failures']],
         lambda r: -r['score_difference'], 'Largest score difference among full failures with zero segment failures.'),
        ('persistent_segment_failures', [r for r in rows if r['full_failed'] and r['segment_failures']],
         lambda r: -r['segment_failure_rate'], 'Highest segment failure rate among motions failing the full sequence.'),
        ('full_success_segment_failure', [r for r in rows if not r['full_failed'] and r['segment_failures']],
         lambda r: -r['segment_failure_rate'], 'Full completion coexists with a failed independent segment; inspect overflow flags before interpretation.')]
    for kind, pool, key, reason in groups:
        if not pool: continue
        motion = sorted(pool, key=lambda r: (key(r), r['name']))[0]
        candidates = [c for c in segments if c['source_motion'] == motion['name']]
        segment = min(candidates, key=lambda c: (not c['failed'], c['norm_return'], c['name']))
        cases.append({'selection': kind, 'reason': reason, 'motion': motion,
                      'segment': segment,
                      'segment_selection': 'Lowest-return failing segment, or lowest-return passing segment if none failed.',
                      'full_npz': f'data/npz/lafan1/{motion["name"]}.npz',
                      'segment_npz': f'data/npz/lafan1_seg10/{segment["name"]}.npz',
                      'replay_status': 'candidate only; no new simulation or rendering performed'})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', type=Path, default=DEFAULT_FULL)
    parser.add_argument('--segmented', type=Path, default=DEFAULT_SEG)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    inputs = [args.full, args.segmented]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    full, seg = [load_results(p) for p in inputs]
    # Comparing different checkpoints would confound this analysis further.
    if full['checkpoint'] != seg['checkpoint']:
        raise ValueError('Checkpoint metadata differ between evaluations')
    rows, segment_rows = match_segments(full['clips'], seg['clips'])
    categories = []
    for group in sorted({r['category'] for r in rows}):
        f = aggregate([c for c in full['clips'] if category(c['name']) == group])
        s = aggregate([c for c in seg['clips'] if category(split_segment(c['name'])[0]) == group])
        categories.append({'category': group, 'full_samples': f['clips'], 'full_failures': f['failed'],
                           'full_failure_rate': f['failure_rate'], 'full_mean_score': f['mean_norm_return'],
                           'segment_samples': s['clips'], 'segment_failures': s['failed'],
                           'segment_failure_rate': s['failure_rate'], 'segment_mean_score': s['mean_norm_return']})
    cases = select_cases(rows, segment_rows)
    fs, ss = aggregate(full['clips']), aggregate(seg['clips'])
    groups = Counter(('full_failed' if r['full_failed'] else 'full_passed') +
                     ('_some_segments_failed' if r['segment_failures'] else '_all_segments_passed') for r in rows)
    summary = {'full': fs, 'segmented': ss, 'matched_motions': len(rows),
               'source_outcome_groups': dict(sorted(groups.items())),
               'source_balanced_segment_mean': sum(r['segment_frame_weighted_score'] for r in rows)/len(rows),
               'omitted_tail_frames': sum(r['omitted_tail_frames'] for r in rows),
               'segment_length_frames': 500, 'fps': 50,
               'categories': categories,
               'overflow_affected_units': {
                   'full': [c['name'] for c in full['clips'] if c['overflow_steps']],
                   'segmented': [c['name'] for c in seg['clips'] if c['overflow_steps']]},
               'limitations': ['Capacity overflows drop contacts/constraints: affected results need simulation verification; they are retained without correction.',
                              'Descriptive comparison of different protocols, not a causal drift experiment.',
                              'Each segment starts independently from its reference state; short tails under 250 frames are omitted.',
                              'No repeated seeds or uncertainty estimate; sequence and segment failure rates have different denominators.',
                              'Aggregate JSON does not contain trajectories: no joint RMSE, drift curve or causal failure diagnosis.',
                              'Categories are inferred from LAFAN1 filename prefixes, not paper BONES categories.']}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'motion_comparison.csv', rows)
    write_csv(args.out/'segment_mapping.csv', segment_rows)
    write_csv(args.out/'category_summary.csv', categories)
    for name, obj in [('summary.json', summary), ('replay_candidates.json', cases)]:
        (args.out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    plot_all(full['clips'], seg['clips'], rows, categories, args.out)
    good = sorted([r for r in rows if r['full_failed'] and not r['segment_failures']],
                  key=lambda r: -r['score_difference'])
    persistent = sorted([r for r in rows if r['segment_failures']], key=lambda r: -r['segment_failure_rate'])
    lines = ['# Experiment 003: LAFAN1 failure analysis', '',
             'Analysis of existing evaluations only; no training or simulation was run.', '',
             '| Protocol | Units | Failed | Failure rate | Mean normalized return | Frame-weighted return |',
             '|---|---:|---:|---:|---:|---:|']
    for label, s in [('Full sequences', fs), ('Independent 10-second segments', ss)]:
        lines.append(f'| {label} | {s["clips"]} | {s["failed"]} | {s["failure_rate"]:.2%} | {s["mean_norm_return"]:.6f} | {s["step_weighted_norm_return"]:.6f} |')
    lines += ['', f'All {len(rows)} original motions matched. Omitted short tails: {summary["omitted_tail_frames"]} frames '
              f'({summary["omitted_tail_frames"]/50:.2f} seconds in total).',
              f'Source-balanced mean of within-motion frame-weighted segment scores: {summary["source_balanced_segment_mean"]:.6f}.',
              f'Overflow steps: full={fs["overflow_steps"]} across {fs["overflow_clips"]} sequences; '
              f'segments={ss["overflow_steps"]} across {ss["overflow_clips"]} segments. '
              'See summary.json for affected names and CSV files for flags. Counts are simulator diagnostics, not failure counts.', '',
              '## Matched outcomes', '', '| Full outcome / segment outcome | Source motions |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in sorted(groups.items())]
    lines += ['', '## Full failures with all independently initialized segments passing', '',
              'This is a strict descriptive definition of better segment performance (zero segment failures), not proof of drift.', '',
              '| Motion | Full score | Segment frame-weighted score | Segments |', '|---|---:|---:|---:|']
    lines += [f'| {r["name"]} | {r["full_score"]:.4f} | {r["segment_frame_weighted_score"]:.4f} | {r["segments"]} |' for r in good]
    lines += ['', '## Motions with remaining segment failures', '',
              '| Motion | Failed segments / total | Full failed | Segment frame-weighted score |', '|---|---:|---|---:|']
    lines += [f'| {r["name"]} | {r["segment_failures"]}/{r["segments"]} | {r["full_failed"]} | {r["segment_frame_weighted_score"]:.4f} |' for r in persistent]
    lines += ['', '## Category distribution', '',
              'Categories remove the take number and subject suffix from LAFAN1 names; they are not semantic annotations.', '',
              '| Category | Full failed / samples | Segment failed / samples |', '|---|---:|---:|']
    lines += [f'| {LABELS[r["category"]]} | {r["full_failures"]}/{r["full_samples"]} | {r["segment_failures"]}/{r["segment_samples"]} |' for r in categories]
    lines += ['', '## Replay candidates', '', 'These are data-selected candidates; no replay has been performed.', '']
    lines += [f'- **{c["selection"]}**: `{c["motion"]["name"]}`; segment `{c["segment"]["name"]}`. {c["reason"]} '
              f'Overflow steps: full motion={c["motion"]["full_overflow_steps"]}, '
              f'all its segments={c["motion"]["segment_overflow_steps"]}, '
              f'selected segment={c["segment"]["overflow_steps"]}.' for c in cases]
    lines += ['', '## Interpretation limits', ''] + ['- '+s for s in summary['limitations']]
    lines += ['', 'Normalized return divides reward by `0.1 * max(frames, steps, 1)`; early termination reduces the score. '
              'Frame-weighted return uses complete reference frames, not survived steps. '
              '`norm_return_survived` is a legacy diagnostic, not the primary tracking score.', '',
              'A segment failure offset is `index * 500 + fail_step` in control steps, not an exact recorded trajectory frame. '
              'Boundary effects and reinitialization can expose failures even in a motion that completed in full.', '',
              '## Reproduction', '', 'From the repository root:', '', '```bash',
              '.venv/bin/python my_work/experiments/exp_003_failure_analysis/analyze.py', '```', '',
              'See `provenance.json` for input hashes and software versions; CSV files retain unrounded derived metrics.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    provenance = {'inputs': [{'path': str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve()),
                               'sha256': digest} for p,digest in zip(inputs,before)],
                  'checkpoint': full['checkpoint'],
                  'versions': {p: importlib.metadata.version(p) for p in ['matplotlib', 'numpy']},
                  'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'segment_contract': 'tools/segment_clips.py defaults: fps=50, seconds=10, min_frames=250',
                  'historical_command_line': 'Not stored in source JSON; cannot verify historical num_envs/device flags.'}
    (args.out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    print(json.dumps(summary, indent=2))
    print(f'Analysis saved to {args.out}')


if __name__ == '__main__':
    main()
