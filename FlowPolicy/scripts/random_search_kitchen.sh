#!/usr/bin/env bash
# Wrapper untuk random search 30 konfigurasi x 3 seed x 50 episode inferensi.
#
# Contoh:
#   bash scripts/random_search_kitchen.sh 0            # GPU 0, default 30 cfg
#   bash scripts/random_search_kitchen.sh 0 --dry-run  # cek konfigurasi saja
#
set -euo pipefail

gpu_id=${1:-0}
shift || true

cd "$(dirname "$0")/.."

export HYDRA_FULL_ERROR=1

python scripts/random_search_kitchen.py \
    --n-configs 30 \
    --episodes 50 \
    --seeds 0 42 101 \
    --out-root data/outputs/random_search \
    --sampling-seed 42 \
    --gpu "${gpu_id}" \
    "$@"
