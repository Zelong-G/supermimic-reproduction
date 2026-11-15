# Personal SuperMimic reproduction workspace

This directory contains Zelong's independent evaluation organization and analysis of the
community **realmimic** implementation. Community source, pretrained weights, methods and
published results remain credited to their original authors. These experiments do not
claim authorship of the model or of the original implementation.

Upstream: https://github.com/kevinhenryzz/realmimic. Preserve the root `LICENSE`, `NOTICE`,
`LICENSES/` and `CITATION.cff`; see those files for component and asset attribution.

| ID | Code / documentation | Output directory (under `outputs/`) | Status |
|---|---|---|---|
| 001 | [LAFAN1 full sequences](experiments/exp_001_lafan1_full/README.md) | `exp_001_lafan1_full/` | Existing evaluation imported byte-for-byte; no training |
| 002 | [LAFAN1 10-second segments](experiments/exp_002_lafan1_seg10/README.md) | `exp_002_lafan1_seg10/` | Existing evaluation imported byte-for-byte; no training |
| 003 | [Failure analysis](experiments/exp_003_failure_analysis/README.md) | `exp_003_failure_analysis/` | Analysis executed; no new simulation or training |
| 004 | [MuJoCo motion visualization](experiments/exp_004_motion_visualization/README.md) | `exp_004_motion_visualization/` | Three cases recorded; MP4 videos and representative screenshots exported and verified |
| 005 | [World Model prediction analysis](experiments/exp_005_world_model_prediction/README.md) | `exp_005_world_model_prediction/` | Executed: one-step and 1/4/8/16/32-step rollout errors; PNG/PDF figures |
| 006 | [BONES-SEED benchmark](experiments/exp_006_bones_benchmark/README.md) | `exp_006_bones_benchmark/` | Full benchmark completed: 129,491 clips, score 0.954480, 12 failures |
| 007 | [Full training reproduction](experiments/exp_007_full_training/README.md) | `exp_007_full_training/` | Completed 86k training; held-out LAFAN1 and BONES rand2000/hard evaluations completed; see paired reports |
| 008 | [Friction adaptation diagnostics](experiments/exp_008_friction_adaptation/README.md) | `exp_008_friction_adaptation/` | Fixed friction, sudden switches, live contact verification and grouped history probes completed; no retraining |
| 009 | [History-based physics adaptation](experiments/exp_009_history_physics_adaptation/README.md) | `exp_009_history_physics_adaptation/` | Six-hour continuation completed at 80,862; checkpoint sweep, ten-friction original/final comparison and 320-episode random-switch study completed; no consistent overall improvement established |

For the cross-experiment result index, source-linked tables, English figures and
a PPT handoff prompt, start with [presentation/README.md](presentation/README.md).

## Layout and rules

Use immutable, sequential `exp_XXX_short_name` identifiers with lowercase English and
underscores. Code goes in `my_work/experiments/<id>/`; outputs go in
`my_work/outputs/<identical-id>/`. Use the next unused ID for new experiments.
Do not renumber earlier experiments. Invoke shared community modules/scripts instead of
copying them. Keep every executed experiment's purpose, commands and results in its README.

Each implemented experiment has its own README. Preserve historical identifiers when adding new work.

Small result JSON, CSV, reports, and analysis PNG/PDF figures may be committed. Put large
trajectories in each output directory's `raw/`, videos in `videos/`, rendered frame sequences
in `frames/`, and temporary caches in `.cache/`; these are ignored. Keep weights in root
`weights/`, datasets in root `data/`, and large logs in root `runs/` (all ignored). Do not put
model files or account credentials in reports or configuration templates.

Experiment 009 follows its explicit isolation requirement: its newly trained checkpoints,
logs, and compilation caches also stay under its matching output directory.

## Environment and running

Run commands from the repository root. Reuse `.venv/bin/python` to avoid changing the locked
community environment. Analysis requires matplotlib, already installed in the current environment;
its direct version is recorded in `requirements-analysis.txt`. Exact NumPy/matplotlib versions
used in the analysis are captured in `outputs/exp_003_failure_analysis/provenance.json`.
If missing in a newly created environment, install the analysis dependency with
`uv pip install --python .venv/bin/python -r my_work/requirements-analysis.txt`.

```bash
# CPU analysis of the archived results (no training, checkpoint loading or simulation)
.venv/bin/python my_work/experiments/exp_003_failure_analysis/analyze.py

# Optional GPU re-evaluations; writes *_rerun.json, preserving the archives
bash my_work/experiments/exp_001_lafan1_full/run.sh
bash my_work/experiments/exp_002_lafan1_seg10/run.sh
```

Original `runs/lafan1_full.json` and `runs/lafan1_seg10.json` were left untouched. Imported
copies have their SHA-256 recorded in each experiment README and analysis provenance. The
historical exact commands were not captured; runner commands are explicitly documented
reproduction recipes rather than reconstructed terminal history.

See the [generated analysis report](outputs/exp_003_failure_analysis/REPORT.md). Independent
segment resets, unequal weighting, omitted tails and simulator capacity overflows limit
interpretation; score differences alone do not establish long-horizon drift as their cause.
