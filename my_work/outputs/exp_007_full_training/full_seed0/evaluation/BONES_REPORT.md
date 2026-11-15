# Experiment 007: BONES-SEED Benchmark

Community results are reused from Experiment 006; community weights were not rerun.
BONES-SEED is training-related data, not held-out generalization. Raw policy weights; same community evaluator.

| Set | Model | Tracking score | Failed / clips | Failure rate | Overflow clips / steps |
|---|---|---:|---:|---:|---:|
| rand2000 | community_exp006 | 0.956083 | 0 / 2000 | 0.0000% | 13 / 15 |
| rand2000 | trained_exp007 | 0.956671 | 0 / 2000 | 0.0000% | 21 / 21 |
| hard | community_exp006 | 0.744589 | 1 / 130 | 0.7692% | 3 / 3 |
| hard | trained_exp007 | 0.747267 | 4 / 130 | 3.0769% | 2 / 2 |

Clip names and frame counts match the archived baselines. Solver capacity overflow and run-to-run variation limit causal interpretation. The new runs share the GPU with another evaluation (recorded in run metadata). No full-set benchmark was started.

Exact commands, list/checkpoint hashes, timestamps, code commit, GPU processes and exit codes are in each *_run.json. See bones_failures.csv for failed motions.

## Failed motions

- hard / community_exp006: crossed_legs_R_001__A430, failure step 545, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A429, failure step 186, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A430_M, failure step 529, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A432, failure step 480, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A432_M, failure step 478, overflow steps 0.
