#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-0}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

exec python3 -u scripts/experiment_3seed_4arms.py --gpu "${gpu_id}" "$@"
