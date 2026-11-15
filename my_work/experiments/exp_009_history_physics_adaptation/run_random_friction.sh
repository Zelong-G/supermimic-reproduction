#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -u my_work/experiments/exp_009_history_physics_adaptation/evaluate_random_friction.py "$@"
