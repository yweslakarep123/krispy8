#!/usr/bin/env bash
# OFAT inferensi (rollout + video + metrics) untuk semua 96 run hasil
# `ofat_search_kitchen.sh`.
#
# Menghasilkan per run (cfg_<ii>_seed<s>/):
#   - inference_ep<N>/videos/episode_*.mp4
#   - inference_ep<N>/metrics.json   (test_mean_score, mean_time, dll)
# dan mengagregasi ke infer_results.csv di out-root.
#
# Contoh:
#   bash scripts/ofat_infer_kitchen.sh 0                        # 50 ep, dengan video
#   bash scripts/ofat_infer_kitchen.sh 0 --preprocess           # out-root = ofat_search_preprocess
#   bash scripts/ofat_infer_kitchen.sh 0 --episodes 20          # turunkan jumlah episode
#   bash scripts/ofat_infer_kitchen.sh 0 --only-hp optimizer.lr
#   bash scripts/ofat_infer_kitchen.sh 0 --force                # ulangi walau metrics.json sudah ada
#   bash scripts/ofat_infer_kitchen.sh 0 --dry-run              # cek cfg -> ckpt mapping
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

${PY} scripts/ofat_infer_kitchen.py \
    --episodes 50 \
    --gpu "${gpu_id}" \
    --device "cuda:0" \
    "$@"
