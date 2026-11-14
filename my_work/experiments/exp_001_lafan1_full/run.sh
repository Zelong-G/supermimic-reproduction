#!/usr/bin/env bash
# Re-evaluate the community release; preserve the archived historical result.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=my_work/outputs/exp_001_lafan1_full/lafan1_full_rerun.json
if [[ -e "$out" ]]; then echo "Refusing to overwrite $out" >&2; exit 1; fi
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt weights/realmimic_g1_86k.pt --npz-dir data/npz/lafan1 \
  --num-envs 512 --out "$out"
