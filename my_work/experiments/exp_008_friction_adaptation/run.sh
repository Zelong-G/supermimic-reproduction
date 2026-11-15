#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2
exec .venv/bin/python -u my_work/experiments/exp_008_friction_adaptation/run.py "$@"
