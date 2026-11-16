# Experiment 007: BONES-SEED Benchmark

Community results are reused from Experiment 006; community weights were not rerun.
BONES-SEED is training-related data, not held-out generalization. Raw policy weights; same community evaluator.

| Set | Model | Tracking score | Failed / clips | Failure rate | Overflow clips / steps |
|---|---|---:|---:|---:|---:|
| rand2000 | community_exp006 | 0.956083 | 0 / 2000 | 0.0000% | 13 / 15 |
| rand2000 | trained_exp007 | 0.956671 | 0 / 2000 | 0.0000% | 21 / 21 |
| hard | community_exp006 | 0.744589 | 1 / 130 | 0.7692% | 3 / 3 |
| hard | trained_exp007 | 0.747267 | 4 / 130 | 3.0769% | 2 / 2 |
| full | community_exp006 | 0.954480 | 12 / 129491 | 0.0093% | 1186 / 1225 |
| full | trained_exp007 | 0.955040 | 6 / 129491 | 0.0046% | 1043 / 1077 |

Clip names and frame counts match the archived baselines. Solver capacity overflow and run-to-run variation limit causal interpretation. The rand2000/hard runs shared the GPU with another evaluation (recorded in run metadata).

For rand2000/hard, commands, hashes, timestamps and exit codes are in *_run.json. See bones_failures.csv for failed motions.

## Full benchmark provenance

The original full run failed with CUDA out-of-memory after 52,000 completed clips; no final JSON was saved. The successful retry is full_retry.json (129,491 motions). The retry command used --num-envs 512 --chunk-size 1000, compared with 1024 / 2000 for the archived community baseline. These changed parallel settings may affect numerical outcomes. A structured run metadata file was not captured for the retry; do not infer exact timing or GPU exclusivity.

Both use the community evaluator, raw policy weights, the G1 robot, freeze_clip_end=true, and data/bench/full.txt. The new checkpoint is full_seed0/ckpt_latest.pt at iteration 86,000. Raw JSON and full logs remain local; only aggregate results and the small failure list are published.

## Failed motions

- hard / community_exp006: crossed_legs_R_001__A430, failure step 545, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A429, failure step 186, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A430_M, failure step 529, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A432, failure step 480, overflow steps 0.
- hard / trained_exp007: crossed_legs_R_001__A432_M, failure step 478, overflow steps 0.
- full / community_exp006: baby_full_diaper_jog_ff_270_loop_R_004__A460_M, failure step 49, overflow steps 0.
- full / community_exp006: buckets_R_001__A531, failure step 263, overflow steps 0.
- full / community_exp006: crossed_legs_R_001__A428_M, failure step 477, overflow steps 0.
- full / community_exp006: crossed_legs_R_001__A429, failure step 537, overflow steps 0.
- full / community_exp006: dance_hiphop_shuffle_sequence_v_001_R_002__A309_M, failure step 879, overflow steps 0.
- full / community_exp006: dance_latino_kick_kick_padeburee_doubled_R_001__A322_M, failure step 168, overflow steps 0.
- full / community_exp006: dozing_cross_legged_R_001__A428_M, failure step 679, overflow steps 0.
- full / community_exp006: faint_stand_up_lying_puke_walk_ff_180_R_001__A473_M, failure step 382, overflow steps 0.
- full / community_exp006: hook_jump_off_wall_R_101__A305, failure step 93, overflow steps 0.
- full / community_exp006: jog_ff_loop_270_003__A049_M, failure step 25, overflow steps 0.
- full / community_exp006: kick_back_001__A495, failure step 407, overflow steps 0.
- full / community_exp006: rest_R_001__A428_M, failure step 837, overflow steps 0.
- full / trained_exp007: crossed_legs_R_001__A429, failure step 186, overflow steps 0.
- full / trained_exp007: crossed_legs_R_001__A429_M, failure step 508, overflow steps 0.
- full / trained_exp007: crossed_legs_R_001__A432, failure step 479, overflow steps 0.
- full / trained_exp007: crossed_legs_R_001__A432_M, failure step 467, overflow steps 0.
- full / trained_exp007: dozing_cross_legged_R_001__A429_M, failure step 730, overflow steps 0.
- full / trained_exp007: rest_R_001__A429_M, failure step 836, overflow steps 0.
