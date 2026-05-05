#!/usr/bin/env bash
# Wrapper eksperimen penuh (seed × profile × CV × random HP).
#
# Profil ``standard`` / ``minimal`` di sini: split mengikuti lipatan CV
# (train/val episode lists), BUKAN shuffle window 70/20/10 — windowing hanya
# untuk train tanpa daftar indeks (mis. train_kitchen.sh). Noise & stride
# sampler tetap mengikuti profil.
#
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
