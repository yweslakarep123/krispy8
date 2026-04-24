#!/usr/bin/env bash
# OFAT evaluasi cepat (tanpa video) untuk semua 96 run hasil
# `ofat_search_kitchen.sh`. Sama seperti `ofat_infer_kitchen.sh` namun
# menambahkan `--no-video` sehingga lebih hemat waktu / disk.
#
# Hasil per run: inference_ep<N>_novideo/metrics.json
# Agregasi: eval_results.csv di out-root.
#
# Contoh:
#   bash scripts/ofat_eval_kitchen.sh 0
#   bash scripts/ofat_eval_kitchen.sh 0 --preprocess
#   bash scripts/ofat_eval_kitchen.sh 0 --only-hp optimizer.lr
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

${PY} scripts/ofat_infer_kitchen.py \
    --no-video \
    --episodes 50 \
    --gpu "${gpu_id}" \
    --device "cuda:0" \
    "$@"
