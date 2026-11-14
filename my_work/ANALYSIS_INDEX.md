# Results for analysis

Start with this index. Paths below are relative to `my_work/outputs/`.
This commit contains a snapshot of results, not a live training dashboard.

| Experiment | Read first | Supporting evidence |
|---|---|---|
| 001–003: LAFAN1 pretrained evaluation | `exp_003_failure_analysis/REPORT.md` | `category_summary.csv`, `motion_comparison.csv`, `segment_mapping.csv`; original results in `exp_001_lafan1_full/lafan1_full.json` and `exp_002_lafan1_seg10/lafan1_seg10.json` |
| 004: visual replay | `exp_004_motion_visualization/recordings/capture_001/<motion>/summary.json` | Viewer session reports, render manifests and representative screenshots; videos and trajectories stay local |
| 005: World Model prediction | `exp_005_world_model_prediction/summary.json` | `metrics.csv`, per-motion summaries and figures |
| 006: BONES-SEED pretrained evaluation | `exp_006_bones_benchmark/REPORT.md` | `summary.csv`, `category_summary.csv`, `failures.csv`, `*_run.json` and figures |
| 007: training from scratch, in progress | `exp_007_full_training/full_seed0/TRAINING_REPORT.md` | `training_summary.json`, `config.json`, `requested_config.json`, `run_metadata.json`, `monitor_status.json`, `metrics/analysis_scalars.csv`, `metrics/throughput.csv` and figures |

Experiment scripts, metric definitions and commands are in `my_work/experiments/`.

## Key results and limitations

- LAFAN1: 14/40 full sequences fail; 37/882 independently initialized segments fail. Fall-and-get-up accounts for 30/37 segment failures; all six full sequences in that category fail. The two failure rates have different denominators.
- BONES-SEED full: 129,491 clips, mean normalized return 0.9544803754, 12 failures versus the community reference of 10. The failure list includes a separate hard-stage result; filter `stage=full` to count the 12 full-stage failures.
- Community motion grouping puts some dance clips into `all_other`; category labels are filename pattern groups, not exhaustive semantic annotations.
- Experiments 001–006 use the released pretrained checkpoint. BONES-SEED is training-related data, not an unseen-motion generalization benchmark.
- Experiment 007 trains new networks with no pretrained initialization or resume. This snapshot's last logged iteration is 7,750 of 86,000; it is not a completed training result or a final evaluation. Read the snapshot timestamps; later local progress is not automatically uploaded.
- `analysis_scalars.csv` selects loss, prediction, objective, return, environment-step, overflow, unphysical and GPU tags from the local `metrics/scalars.csv`, preserving original values, steps and event-file provenance without smoothing. Bulk raw logs remain local. Figures use the documented trailing mean.
- Training losses and online prediction scores are not independent evaluation metrics. Simulator capacity overflows and unphysical diagnostics must be considered; statistics alone do not establish causal failure mechanisms or long-horizon drift.
- Smoke checks passed, but the smoke monitor also recorded 42 nonfinite scalar samples. Do not conflate selected finite forward metrics with all logged diagnostics being finite; see smoke report and checks together.
- In-progress report fields such as final checkpoint or verified final iterations can be unavailable. A running monitor status is a snapshot; it does not prove the process is still alive when this is read.

## Suggested analysis request

Please read this index and the linked experiment reports, then assess: (1) which motion types fail most often, separating full sequences from independently reset segments and BONES-SEED; (2) how closely the pretrained reproduction matches the community reference; (3) whether the in-progress World Model, Policy and Critic curves show issues; and (4) which further tests would distinguish plausible causes. Cite filenames and numbers, distinguish observations from hypotheses, account for overflows, and do not present unfinished training as a completed result.
