#!/usr/bin/env bash
# Suite eksperimen: 3 seed × (tanpa preprocess | preprocess) = 6× BO,
# lalu eval SR+latensi semua arm, inferensi video untuk model terbaik global.
#
#   bash scripts/bo_franka_kitchen_suite.sh 0 --dry-run
#   bash scripts/bo_franka_kitchen_suite.sh 0 --suite-seeds 0 42 101 --n-trials 1
# Argumen pertama = id GPU (sama konvensi dengan bayes_opt_kitchen.sh).
set -euo pipefail
gpu_id="${1:-0}"
shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
PY="python3 -u"
if command -v stdbuf >/dev/null 2>&1; then
  PY="stdbuf -oL -eL python3 -u"
fi
exec ${PY} scripts/bo_franka_kitchen_suite.py --gpu "${gpu_id}" "$@"
