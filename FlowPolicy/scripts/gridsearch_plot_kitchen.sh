#!/usr/bin/env bash
# Plot perbandingan Grid Search: SR & latency per-HP, heatmap pair, scatter Pareto.
#
# Input default: <out-root>/infer_results.csv (dari gridsearch_infer_kitchen.sh)
# Fallback: eval_results.csv atau results.csv.
#
# Output di <out-root>/plots/:
#   - grid_sr_marginal.png
#   - grid_latency_marginal.png
#   - grid_pair_heatmap.png
#   - grid_sr_vs_latency.png
#   - grid_summary.csv
#
# Contoh:
#   bash scripts/gridsearch_plot_kitchen.sh
#   bash scripts/gridsearch_plot_kitchen.sh --preprocess
#   bash scripts/gridsearch_plot_kitchen.sh --metric-source eval
#   bash scripts/gridsearch_plot_kitchen.sh --metric-source search
#   bash scripts/gridsearch_plot_kitchen.sh --csv /path/custom.csv
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/gridsearch_plot_kitchen.py "$@"
