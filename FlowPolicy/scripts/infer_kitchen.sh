#!/usr/bin/env bash
# Inferensi + simpan video + metrics.json
# Contoh:
#   bash scripts/infer_kitchen.sh data/outputs/kitchen_complete-flowpolicy_seed0/checkpoints/latest.ckpt 0
set -euo pipefail

CKPT=${1:?usage: $0 <path/to/latest.ckpt> [gpu_id] [episodes]}
GPU=${2:-0}
N_EP=${3:-10}

cd "$(dirname "$0")/../FlowPolicy"

export CUDA_VISIBLE_DEVICES=${GPU}
python infer_kitchen.py --checkpoint "${CKPT}" --episodes "${N_EP}" --device cuda:0
