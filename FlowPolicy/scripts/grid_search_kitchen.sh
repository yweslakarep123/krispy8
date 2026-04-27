#!/usr/bin/env bash
# Grid search Franka Kitchen. Contoh (3 GPU, satu seed per GPU):
#   bash scripts/grid_search_kitchen.sh 0,1,2
#   bash scripts/grid_search_kitchen.sh 0     # satu GPU, urut
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
gpu_pool="${1:-0}"
re='^[0-9,]+$'
if [[ "$gpu_pool" =~ $re ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu_pool"
  exec python3 -u "$here/grid_search_kitchen.py" --gpu-pool "$gpu_pool" "${@:2}"
else
  # arg pertama bukan pool (mis. --dry-run)
  exec python3 -u "$here/grid_search_kitchen.py" "$@"
fi
