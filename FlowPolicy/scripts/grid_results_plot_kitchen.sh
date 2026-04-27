#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -u grid_results_plot_kitchen.py "$@"
