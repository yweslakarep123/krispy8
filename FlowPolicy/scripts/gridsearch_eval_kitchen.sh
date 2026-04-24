#!/usr/bin/env bash
# Evaluasi cepat (tanpa video) untuk semua run Grid Search.
# Hasil per run: inference_ep<N>_novideo/metrics.json
# Agregasi: eval_results.csv di out-root.
#
# Contoh:
#   bash scripts/gridsearch_eval_kitchen.sh 0
#   bash scripts/gridsearch_eval_kitchen.sh 0 --preprocess
#   bash scripts/gridsearch_eval_kitchen.sh 0 --only-cfg 0 1 2
set -euo pipefail

gpu_id=${1:-0}
shift || true

cd "$(dirname "$0")/.."
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

PY="python -u"
if command -v stdbuf >/dev/null 2>&1; then
    PY="stdbuf -oL -eL python -u"
fi

${PY} scripts/gridsearch_infer_kitchen.py \
    --no-video \
    --episodes 50 \
    --gpu "${gpu_id}" \
    --device "cuda:0" \
    "$@"
