#!/usr/bin/env bash
# OFAT sweep: 8 HP x 4 nilai x 3 seed = 96 run.
#
# Contoh:
#   bash scripts/ofat_search_kitchen.sh 0                       # GPU 0
#   bash scripts/ofat_search_kitchen.sh 0 --dry-run             # cek konfigurasi
#   bash scripts/ofat_search_kitchen.sh 0 --only-hp optimizer.lr
#   bash scripts/ofat_search_kitchen.sh 0 --max-minutes 600     # stop setelah 10 jam (Colab T4)
#
set -euo pipefail

gpu_id=${1:-0}
shift || true

cd "$(dirname "$0")/.."
export HYDRA_FULL_ERROR=1

python scripts/ofat_search_kitchen.py \
    --episodes 50 \
    --seeds 0 42 101 \
    --out-root data/outputs/ofat_search \
    --gpu "${gpu_id}" \
    "$@"
