#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPYCACHEPREFIX="$PWD/my_work/outputs/exp_009_history_physics_adaptation/.cache/pycache"
exec .venv/bin/python -u my_work/experiments/exp_009_history_physics_adaptation/train.py "$@"
