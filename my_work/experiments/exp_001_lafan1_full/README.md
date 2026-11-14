# Experiment 001: LAFAN1 Full-Sequence Benchmark

Purpose: evaluate the released community policy on complete LAFAN1 sequences.
Status: existing completed evaluation, imported without rerunning. **No training or fine-tuning was performed in this experiment.**

## Inputs and attribution

- Community checkpoint: `weights/realmimic_g1_86k.pt`, released by realmimic (Hugging Face `kevinhenryz/realmimic`).
- JSON metadata: iteration 86000, run `release`, robot `g1`, raw `policy` weights, `freeze_clip_end=true`.
- The checkpoint's `train_seconds` describes upstream training, not compute performed by this experiment.
- Dataset: held-out LAFAN1 G1-retargeted motions, converted to 50 Hz; local path `data/npz/lafan1/`.
- Original output (retained): `runs/lafan1_full.json`.
- Archived output: `my_work/outputs/exp_001_lafan1_full/lafan1_full.json`.
- Original/copy SHA-256: `6f05baf01bf51dee6921c2dac3714bbfadf6f21dc800e5434bea89936583f508`.

## Commands and provenance

The original full command line, device flag and parallel environment count are **not recorded**
in the JSON, and no matching shell-history entry was available. They cannot be reported as verified
historical facts. The following is a reproducible invocation using the observed checkpoint/data paths
and the community README's suggested environment count; changing parallelism can change outcomes.
This organization task did not rerun the evaluation.

Run from the repository root:

```bash
bash my_work/experiments/exp_001_lafan1_full/run.sh
```

The runner executes:

```bash
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt weights/realmimic_g1_86k.pt --npz-dir data/npz/lafan1 \
  --num-envs 512 \
  --out my_work/outputs/exp_001_lafan1_full/lafan1_full_rerun.json
```

CUDA/GPU required. The runner refuses to overwrite an existing rerun output. The archived
JSON is never overwritten. Re-evaluation results are new observations, not guaranteed bitwise replicas.

## Observed archived results

| Metric | Value |
|---|---:|
| Evaluation units | 40 |
| Failed units | 14 |
| Failure rate | 35.000000% |
| End-of-clip failures (last 3 frames) | 0 |
| Mean normalized return | 0.659233184522 |
| Reference-frame-weighted normalized return | 0.690771897944 |
| Legacy survived-step mean (diagnostic only) | 0.811354727502 |
| Simulator capacity overflow steps | 14 |
| Units with capacity overflow | 11 |

Metrics follow `src/supermimic/eval/benchmark.py`: primary normalized return includes
early-termination penalties; it is not joint-position RMSE. Failure means the height/tilt
termination fires; thresholds are 0.25 m and 0.8 respectively (tilt is the community termination
measure, not an angle in radians). Capacity overflows can drop contacts/constraints and require
verification before physical conclusions. No original values were changed.

Full-sequence and independent-segment benchmarks use different protocols and evaluation units.
Their differences do not by themselves prove long-term drift. See Experiment 003 for the matched analysis.
