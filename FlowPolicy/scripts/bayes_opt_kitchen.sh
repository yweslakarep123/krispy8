#!/usr/bin/env bash
# Bayesian optimization (Optuna TPE) untuk FlowPolicy FrankaKitchen-v1.
#
#   bash scripts/bayes_opt_kitchen.sh 0 --dry-run
#   bash scripts/bayes_opt_kitchen.sh 0 --n-trials 30
#   bash scripts/bayes_opt_kitchen.sh 0 --preprocess --n-trials 20
set -euo pipefail

gpu_id=${1:-0}
shift || true

cd "$(dirname "$0")/.."
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

PY="python -u"
if command -v stdbuf >/dev/null 2>&1; then
  PY="stdbuf -oL -eL python -u"
fi

${PY} scripts/bayes_opt_kitchen.py \
  --seeds 0 42 101 \
  --episodes 50 \
  --out-root data/outputs/bayes_opt \
  --gpu "${gpu_id}" \
  "$@"
