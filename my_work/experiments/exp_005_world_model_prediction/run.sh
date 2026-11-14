#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python my_work/experiments/exp_005_world_model_prediction/run.py "$@"
