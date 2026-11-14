#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python my_work/experiments/exp_007_full_training/training.py "$@"
