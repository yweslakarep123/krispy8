#!/usr/bin/env bash
# Grid search FlowPolicy di FrankaKitchen-v1 (full cartesian).
#
# Berbeda dengan OFAT (`ofat_search_kitchen.sh`), semua subset HP di-cross
# menjadi satu konfigurasi utuh. Setiap (cfg_idx, seed) training dari
# nol (training.resume=False dipaksa) di folder run_dir tersendiri.
#
# Contoh:
#   bash scripts/gridsearch_kitchen.sh 0 --dry-run                 # cek kombinasi subset default
#   bash scripts/gridsearch_kitchen.sh 0                           # sweep default (~1296 run)
#   bash scripts/gridsearch_kitchen.sh 0 --subset smoke            # preset kecil untuk smoke test
#   bash scripts/gridsearch_kitchen.sh 0 --preprocess              # data pipeline sliding-window
#   bash scripts/gridsearch_kitchen.sh 0 --max-minutes 660         # batas waktu 11 jam (Colab T4)
#
# Override subset HP via CLI (gunakan ':' sebagai separator; list pakai '[]'):
#   bash scripts/gridsearch_kitchen.sh 0 \
#     --override-hp optimizer.lr=1e-3:5e-4:1e-4 \
#     --override-hp policy.down_dims=[128,256,512]:[256,512,1024]
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

${PY} scripts/gridsearch_kitchen.py \
    --episodes 50 \
    --seeds 0 42 101 \
    --out-root data/outputs/gridsearch \
    --gpu "${gpu_id}" \
    "$@"
