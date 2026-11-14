# Experiment 002: LAFAN1 10-Second Benchmark

Purpose: evaluate the released community policy on independently initialized LAFAN1 segments.
Status: existing completed evaluation, imported without rerunning. **No training or fine-tuning was performed in this experiment.**

## Inputs and attribution

- Community checkpoint: `weights/realmimic_g1_86k.pt`, released by realmimic (Hugging Face `kevinhenryz/realmimic`).
- JSON metadata: iteration 86000, run `release`, robot `g1`, raw `policy` weights, `freeze_clip_end=true`.
- The checkpoint's `train_seconds` describes upstream training, not compute performed by this experiment.
- Dataset: held-out LAFAN1 G1-retargeted motions, converted to 50 Hz; local path `data/npz/lafan1_seg10/`.
- Original output (retained): `runs/lafan1_seg10.json`.
- Archived output: `my_work/outputs/exp_002_lafan1_seg10/lafan1_seg10.json`.
- Original/copy SHA-256: `b4330fa407aa12c2d6fd0bcfe1c844930d185b93e6e78ace72abaffec2d7ede2`.

## Commands and provenance

The original full command line, device flag and parallel environment count are **not recorded**
in the JSON, and no matching shell-history entry was available. They cannot be reported as verified
historical facts. The following is a reproducible invocation using the observed checkpoint/data paths
and the community README's suggested environment count; changing parallelism can change outcomes.
This organization task did not rerun the evaluation.

Segment preparation (already completed locally):

```bash
.venv/bin/python tools/segment_clips.py data/npz/lafan1 data/npz/lafan1_seg10 --seconds 10
```

At 50 Hz, segments are up to 500 frames. Final tails of at least 250 frames are retained;
shorter tails are omitted. Each segment independently starts from its reference state.
The existing converted dataset has 882 segment files corresponding to the 882 JSON records.

Run from the repository root:

```bash
bash my_work/experiments/exp_002_lafan1_seg10/run.sh
```

The runner executes:

```bash
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt weights/realmimic_g1_86k.pt --npz-dir data/npz/lafan1_seg10 \
  --num-envs 1024 \
  --out my_work/outputs/exp_002_lafan1_seg10/lafan1_seg10_rerun.json
```

CUDA/GPU required. The runner refuses to overwrite an existing rerun output. The archived
JSON is never overwritten. Re-evaluation results are new observations, not guaranteed bitwise replicas.

## Observed archived results

| Metric | Value |
|---|---:|
| Evaluation units | 882 |
| Failed units | 37 |
| Failure rate | 4.195011% |
| End-of-clip failures (last 3 frames) | 1 |
| Mean normalized return | 0.828295180696 |
| Reference-frame-weighted normalized return | 0.827733830380 |
| Legacy survived-step mean (diagnostic only) | 0.838010095986 |
| Simulator capacity overflow steps | 18 |
| Units with capacity overflow | 17 |

Metrics follow `src/supermimic/eval/benchmark.py`: primary normalized return includes
early-termination penalties; it is not joint-position RMSE. Failure means the height/tilt
termination fires; thresholds are 0.25 m and 0.8 respectively (tilt is the community termination
measure, not an angle in radians). Capacity overflows can drop contacts/constraints and require
verification before physical conclusions. No original values were changed.

Full-sequence and independent-segment benchmarks use different protocols and evaluation units.
Their differences do not by themselves prove long-term drift. See Experiment 003 for the matched analysis.
