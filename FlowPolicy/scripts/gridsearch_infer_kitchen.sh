#!/usr/bin/env bash
# Inferensi (rollout + video + metrics) untuk semua run Grid Search.
#
# Output per run (cfg_<ii>_seed<s>/):
#   - inference_ep<N>/videos/episode_*.mp4
#   - inference_ep<N>/metrics.json
# Agregasi: infer_results.csv di out-root.
#
# Contoh:
#   bash scripts/gridsearch_infer_kitchen.sh 0
#   bash scripts/gridsearch_infer_kitchen.sh 0 --preprocess
#   bash scripts/gridsearch_infer_kitchen.sh 0 --episodes 20
#   bash scripts/gridsearch_infer_kitchen.sh 0 --only-cfg 0 1 2
#   bash scripts/gridsearch_infer_kitchen.sh 0 --force
#   bash scripts/gridsearch_infer_kitchen.sh 0 --dry-run
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
    --episodes 50 \
    --gpu "${gpu_id}" \
    --device "cuda:0" \
    "$@"
