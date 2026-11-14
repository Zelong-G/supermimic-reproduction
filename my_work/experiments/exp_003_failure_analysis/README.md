# Experiment 003: Failure Analysis

Purpose: compare existing full-sequence and independently initialized 10-second LAFAN1
benchmark results from the same released community checkpoint. **No training, inference,
new simulation or trajectory rendering is performed.**

## Inputs and outputs

- `my_work/outputs/exp_001_lafan1_full/lafan1_full.json`
- `my_work/outputs/exp_002_lafan1_seg10/lafan1_seg10.json`
- All derived files: `my_work/outputs/exp_003_failure_analysis/`.

From the repository root, using the existing environment:

```bash
.venv/bin/python my_work/experiments/exp_003_failure_analysis/analyze.py
.venv/bin/python -m unittest discover -s my_work/experiments/exp_003_failure_analysis -p 'test_*.py'
```

For new rerun results, explicitly select them and a different output directory:

```bash
.venv/bin/python my_work/experiments/exp_003_failure_analysis/analyze.py \
  --full my_work/outputs/exp_001_lafan1_full/lafan1_full_rerun.json \
  --segmented my_work/outputs/exp_002_lafan1_seg10/lafan1_seg10_rerun.json \
  --out my_work/outputs/exp_003_failure_analysis/rerun
```

## Method and validation

Read definitions in `src/supermimic/eval/benchmark.py`, CLI in `scripts/eval_benchmark.py`,
and segmentation in `tools/segment_clips.py` before implementing this analysis. Recompute
stored summaries from clip rows and reject inconsistent values. Require equal checkpoint
metadata, unique clip names, valid completion/failure fields, all sources matched, contiguous
segment indices and the exact expected 500-frame / minimum-250-frame tail lengths.

The mapping uses `<source>__seg<index>`; start offset is `index * 500` at 50 Hz. Categories
come from the LAFAN1 filename prefix after removing take and subject numbers. Do not apply
the community BONES Table A.8 groups to LAFAN1. Report clip averages, reference-frame-weighted
averages, and within-motion matched segment scores; preserve full precision in CSV/JSON.

Outputs include three CSV tables (categories, motion comparisons, every segment mapping),
a machine-readable summary, data-selected replay candidates, an English report, input hashes
and software provenance. Four English figures are written as both PNG and PDF. The heatmap
shows absent trailing segments in gray and failures with red crosses.

## Observed results

- Full: 14/40 failures; mean normalized return 0.659233.
- Segmented: 37/882 failures; mean normalized return 0.828295.
- All 40 source motions matched; 3309 short-tail frames omitted (66.18 s total).
- Four full failures have zero failing segments: `dance2_subject4`, `fightAndSports1_subject1`,
  `walk3_subject1`, `jumps1_subject2`.
- Eleven source motions still have failing segments. Fall-and-get-up accounts for 30/37
  segment failures; all six full fall-and-get-up sequences fail.
- `dance1_subject2` completes in full but fails in segment 003. That failed segment also
  records a capacity overflow, so it requires verification before interpreting its cause.
- Full/segment overflow diagnostics total 14/18 steps across 11/17 evaluation units;
  originals remain unchanged and all affected units are flagged in the outputs.

See [REPORT.md](../../outputs/exp_003_failure_analysis/REPORT.md) for complete results,
representative replay candidates and selection criteria. Candidates have not been replayed.

## Interpretation limits

Independent resets, endpoint coverage, duration weighting and simulator overflows all affect
comparability. Segment failure rate and sequence failure rate have different denominators.
There are no repeated-seed estimates. Aggregate rewards cannot yield drift trajectories,
joint RMSE or causal failure diagnoses. Differences must not be attributed entirely to drift.
