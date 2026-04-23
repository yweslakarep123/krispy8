#!/usr/bin/env bash
# OFAT sweep: 8 HP x 4 nilai x 3 seed = 96 run.
#
# Contoh:
#   bash scripts/ofat_search_kitchen.sh 0                       # GPU 0, TANPA preprocessing (default)
#   bash scripts/ofat_search_kitchen.sh 0 --preprocess          # GPU 0, DENGAN sliding-window preprocessing
#   bash scripts/ofat_search_kitchen.sh 0 --dry-run             # cek konfigurasi
#   bash scripts/ofat_search_kitchen.sh 0 --only-hp optimizer.lr
#   bash scripts/ofat_search_kitchen.sh 0 --max-minutes 600     # stop setelah 10 jam (Colab T4)
#
# Catatan preprocessing:
#   - `--preprocess` mengaktifkan sliding-window + split 70/20/10 +
#     augmentasi noise hanya di split training (lihat README.md bagian
#     "Preprocessing sliding-window").
#   - Saat `--preprocess` aktif dan `--out-root` tidak di-override,
#     out-root otomatis jadi `data/outputs/ofat_search_preprocess`
#     supaya hasilnya tidak menimpa sweep non-preprocessing.
#   - Parameter preprocessing bisa di-override via CLI:
#         --preprocess --preprocess-window-ratio 0.25 \
#         --preprocess-stride 1 --preprocess-train-ratio 0.70 \
#         --preprocess-val-ratio 0.20 --preprocess-test-ratio 0.10
#
set -euo pipefail

gpu_id=${1:-0}
shift || true

cd "$(dirname "$0")/.."
export HYDRA_FULL_ERROR=1
# Force real-time stdout/stderr di Colab (%%bash buffering).
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# stdbuf -oL -eL memaksa line-buffered di python; aman di-skip kalau stdbuf
# tidak ada (jarang terjadi di Colab).
PY="python -u"
if command -v stdbuf >/dev/null 2>&1; then
    PY="stdbuf -oL -eL python -u"
fi

${PY} scripts/ofat_search_kitchen.py \
    --episodes 50 \
    --seeds 0 42 101 \
    --out-root data/outputs/ofat_search \
    --gpu "${gpu_id}" \
    "$@"
