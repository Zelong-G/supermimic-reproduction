# Experiment 003: LAFAN1 failure analysis

Analysis of existing evaluations only; no training or simulation was run.

| Protocol | Units | Failed | Failure rate | Mean normalized return | Frame-weighted return |
|---|---:|---:|---:|---:|---:|
| Full sequences | 40 | 14 | 35.00% | 0.659233 | 0.690772 |
| Independent 10-second segments | 882 | 37 | 4.20% | 0.828295 | 0.827734 |

All 40 original motions matched. Omitted short tails: 3309 frames (66.18 seconds in total).
Source-balanced mean of within-motion frame-weighted segment scores: 0.813400.
Overflow steps: full=14 across 11 sequences; segments=18 across 17 segments. See summary.json for affected names and CSV files for flags. Counts are simulator diagnostics, not failure counts.

## Matched outcomes

| Full outcome / segment outcome | Source motions |
|---|---:|
| full_failed_all_segments_passed | 4 |
| full_failed_some_segments_failed | 10 |
| full_passed_all_segments_passed | 25 |
| full_passed_some_segments_failed | 1 |

## Full failures with all independently initialized segments passing

This is a strict descriptive definition of better segment performance (zero segment failures), not proof of drift.

| Motion | Full score | Segment frame-weighted score | Segments |
|---|---:|---:|---:|
| dance2_subject4 | 0.2124 | 0.8292 | 23 |
| fightAndSports1_subject1 | 0.1933 | 0.7939 | 25 |
| walk3_subject1 | 0.7044 | 0.8723 | 25 |
| jumps1_subject2 | 0.7058 | 0.7740 | 24 |

## Motions with remaining segment failures

| Motion | Failed segments / total | Full failed | Segment frame-weighted score |
|---|---:|---|---:|
| fallAndGetUp3_subject1 | 6/10 | True | 0.5186 |
| fallAndGetUp1_subject1 | 8/17 | True | 0.6063 |
| fallAndGetUp2_subject3 | 6/16 | True | 0.5171 |
| fallAndGetUp1_subject5 | 5/17 | True | 0.5386 |
| fallAndGetUp1_subject4 | 3/17 | True | 0.5948 |
| dance1_subject1 | 2/13 | True | 0.7930 |
| fallAndGetUp2_subject2 | 2/16 | True | 0.6242 |
| fight1_subject3 | 2/24 | True | 0.7992 |
| dance1_subject2 | 1/13 | False | 0.7703 |
| walk2_subject3 | 1/24 | True | 0.9098 |
| walk3_subject3 | 1/25 | True | 0.8600 |

## Category distribution

Categories remove the take number and subject suffix from LAFAN1 names; they are not semantic annotations.

| Category | Full failed / samples | Segment failed / samples |
|---|---:|---:|
| Dance | 2/8 | 3/154 |
| Fall and get up | 6/6 | 30/93 |
| Fight | 1/3 | 2/72 |
| Fight and sports | 1/2 | 0/50 |
| Jumps | 1/3 | 0/72 |
| Run | 0/4 | 0/96 |
| Sprint | 0/2 | 0/54 |
| Walk | 3/12 | 2/291 |

## Replay candidates

These are data-selected candidates; no replay has been performed.

- **full_success_all_segments_success**: `walk1_subject1`; segment `walk1_subject1__seg025`. Highest full score among motions passing both protocols. Overflow steps: full motion=0, all its segments=0, selected segment=0.
- **full_failure_all_segments_success**: `dance2_subject4`; segment `dance2_subject4__seg005`. Largest score difference among full failures with zero segment failures. Overflow steps: full motion=0, all its segments=1, selected segment=0.
- **persistent_segment_failures**: `fallAndGetUp3_subject1`; segment `fallAndGetUp3_subject1__seg009`. Highest segment failure rate among motions failing the full sequence. Overflow steps: full motion=0, all its segments=1, selected segment=0.
- **full_success_segment_failure**: `dance1_subject2`; segment `dance1_subject2__seg003`. Full completion coexists with a failed independent segment; inspect overflow flags before interpretation. Overflow steps: full motion=0, all its segments=3, selected segment=1.

## Interpretation limits

- Capacity overflows drop contacts/constraints: affected results need simulation verification; they are retained without correction.
- Descriptive comparison of different protocols, not a causal drift experiment.
- Each segment starts independently from its reference state; short tails under 250 frames are omitted.
- No repeated seeds or uncertainty estimate; sequence and segment failure rates have different denominators.
- Aggregate JSON does not contain trajectories: no joint RMSE, drift curve or causal failure diagnosis.
- Categories are inferred from LAFAN1 filename prefixes, not paper BONES categories.

Normalized return divides reward by `0.1 * max(frames, steps, 1)`; early termination reduces the score. Frame-weighted return uses complete reference frames, not survived steps. `norm_return_survived` is a legacy diagnostic, not the primary tracking score.

A segment failure offset is `index * 500 + fail_step` in control steps, not an exact recorded trajectory frame. Boundary effects and reinitialization can expose failures even in a motion that completed in full.

## Reproduction

From the repository root:

```bash
.venv/bin/python my_work/experiments/exp_003_failure_analysis/analyze.py
```

See `provenance.json` for input hashes and software versions; CSV files retain unrounded derived metrics.
