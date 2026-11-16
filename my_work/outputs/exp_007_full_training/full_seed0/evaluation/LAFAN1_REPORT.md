# Experiment 007: Held-out LAFAN1 Comparison

## Selected final presentation result

**Selected run: evaluation 2, `lafan1_repeat1.json`. Tracking score: 0.661055; failed: 14 / 40 (35.0%).**

Selected at the researcher's request after reviewing all three repeats. This is the highest-scoring repeat, not the repeat mean or a preselected independent test. All original JSON files are retained.

| Evaluation | Source | Tracking score | Failed / total | Presentation selection |
|---|---|---:|---:|---|
| 1 | `lafan1_trained.json` | 0.638998 | 16 / 40 | Retained repeat |
| 2 | `lafan1_repeat1.json` | 0.661055 | 14 / 40 | Selected final presentation result |
| 3 | `lafan1_repeat2.json` | 0.649009 | 16 / 40 | Retained repeat |

Across all three repeats: mean 0.649688, sample standard deviation 0.011044. The selected run recorded 10 capacity-affected clips / 13 overflow steps. This selection does not establish superiority or equivalence to the community model.

The comparison table and per-motion CSV below describe the original first evaluation, retained for provenance. `lafan1_repeats.csv` explicitly marks the selected result.

## Original first-run comparison
Both evaluations completed successfully on the same 40 full sequences (matching names and frame counts). Raw policy weights, 86,000-iteration checkpoints, CUDA:0, requested num_envs=512 (community evaluator caps actual environments to 40), no chunking. The evaluator uses seed=0, start-of-clip initialization, termination enabled, and disables observation/reset/action noise and domain randomization. No training or parameter updates were performed.

| Model | Tracking score | Failed / total | Failure rate | Overflow clips / steps |
|---|---:|---:|---:|---:|
| community | 0.682738 | 13 / 40 | 32.5% | 7 / 9 |
| trained | 0.638998 | 16 / 40 | 40.0% | 8 / 11 |

The trained model's score is lower by 0.043740 in this single run, with three additional failures: fightAndSports1_subject1, run1_subject2, walk3_subject4. All 13 community-policy failures also failed with the trained policy.

This is a similar performance range, but does not establish equivalent performance. The repository README reports 0.664 / 0.667 and 13 / 14 failures across two community training seeds. The current community-weight rerun differs from historical measurements. Simulator capacity overflow occurred in both runs; no solver/capacity settings were changed. Repeated runs are needed to quantify variation; no causal attribution to drift or training seed is established. LAFAN1 is held out; BONES-SEED training-related benchmark results are a separate protocol.

## Reproduce

Run each command from the repository root using a new output name to preserve existing results. Exact executed commands and timestamps are stored in lafan1_community_run.json and lafan1_trained_run.json.

```bash
.venv/bin/python scripts/eval_benchmark.py --ckpt weights/realmimic_g1_86k.pt --npz-dir data/npz/lafan1 --num-envs 512 --out my_work/outputs/exp_007_full_training/full_seed0/evaluation/lafan1_community.json
```

```bash
.venv/bin/python scripts/eval_benchmark.py --ckpt my_work/outputs/exp_007_full_training/full_seed0/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 512 --out my_work/outputs/exp_007_full_training/full_seed0/evaluation/lafan1_trained.json
```

Code commit at evaluation: 6d7cc7714ea32690e9df7ae31757b764866c0e77. Working tree contains local changes; this commit alone does not capture them.

Files: lafan1_comparison.csv (aggregate), lafan1_per_motion.csv (paired clip results), original JSON outputs and console logs.
