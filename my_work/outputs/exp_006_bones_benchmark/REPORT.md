# Experiment 006: BONES-SEED Benchmark Reproduction

Pretrained community policy only; no training. BONES-SEED is training-related data, not unseen-motion generalization.

| Stage | Clips | Mean tracking score | Failed | Failure rate | Overflow clips |
|---|---:|---:|---:|---:|---:|
| rand2000 | 2000 | 0.956083 | 0 | 0.0000% | 13 |
| hard | 130 | 0.744589 | 1 | 0.7692% | 3 |
| full | 129491 | 0.954480 | 12 | 0.0093% | 1186 |

## Community reference

Full benchmark: 129,491 clips, mean normalized return 0.9545, 10 failed. Random/hard subsets have different distributions; they are not direct reproductions of the full-set mean or failure count.

## Failure analysis

Local `failures.csv` contains exact failure names, timing, score and overflow flags. `category_summary.csv` uses the community Table A.8 name patterns; zero-sample categories have no rate. The 130-clip hard set covers only five HARD_SET_GROUPS, not every Table A.8 category.

## Reproducibility

`*_run.json` records the actual command, checkpoint hash, list hashes, requested/effective parallelism, chunk size, GPU resources, software versions and elapsed wall time. Raw JSON summaries are recomputed and exact motion coverage is checked.

Main tracking score is mean norm_return, penalizing early termination. Frame-weighted return is separate. The survived-step legacy score is not substituted. Overflow diagnostics are reported and may affect physical validity; numerical/contact behavior can depend on parallelism.

Raw per-clip JSON, failure names and logs remain local/ignored; aggregate tables and figures may be committed. Source data, lists and weights remain in their existing ignored directories.
