#!/usr/bin/env bash
# Wrapper eksperimen penuh (seed × profile × CV × random HP).
# Contoh:
#   bash scripts/run_experiment.sh 0 --dry-run
#   bash scripts/run_experiment.sh 0 --output-dir data/outputs/experiment --n-configs 10
set -euo pipefail

gpu_id=${1:-0}
shift || true

cd "$(dirname "$0")/.."
export HYDRA_FULL_ERROR=1

python scripts/run_experiment.py \
    --gpu "${gpu_id}" \
    --output-dir data/outputs/experiment \
    "$@"
