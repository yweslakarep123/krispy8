#!/usr/bin/env bash
# Wrapper HalvingRandomSearchCV untuk tuning HP FlowPolicy kitchen.
# Usage:
#   bash scripts/halving_search_kitchen.sh 0 --dry-run
#   bash scripts/halving_search_kitchen.sh 0 --n-candidates 16 --min-episodes 10 --max-episodes 50
set -euo pipefail

gpu_id="${1:-0}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

exec python3 -u scripts/halving_search_kitchen.py --gpu "${gpu_id}" "$@"
