#!/usr/bin/env bash
# Re-evaluate existing segments; preserve the archived historical result.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=my_work/outputs/exp_002_lafan1_seg10/lafan1_seg10_rerun.json
if [[ -e "$out" ]]; then echo "Refusing to overwrite $out" >&2; exit 1; fi
if [[ ! -d data/npz/lafan1_seg10 ]]; then
  .venv/bin/python tools/segment_clips.py data/npz/lafan1 data/npz/lafan1_seg10 --seconds 10
fi
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt weights/realmimic_g1_86k.pt --npz-dir data/npz/lafan1_seg10 \
  --num-envs 1024 --out "$out"
