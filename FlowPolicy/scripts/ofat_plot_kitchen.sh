#!/usr/bin/env bash
# Plot perbandingan OFAT: success rate & inference latency per hyperparameter.
#
# Input default: `<out-root>/infer_results.csv` (hasil `ofat_infer_kitchen.sh`).
# Fallback: `eval_results.csv` (ofat_eval_kitchen.sh) atau `results.csv` (training).
#
# Output (di <out-root>/plots/):
#   - ofat_sr_vs_hp.png
#   - ofat_latency_vs_hp.png
#   - ofat_sr_and_latency.png
#   - ofat_summary.csv
#
# Contoh:
#   bash scripts/ofat_plot_kitchen.sh                           # pakai infer_results.csv
#   bash scripts/ofat_plot_kitchen.sh --preprocess              # out-root = ofat_search_preprocess
#   bash scripts/ofat_plot_kitchen.sh --metric-source eval
#   bash scripts/ofat_plot_kitchen.sh --metric-source search
#   bash scripts/ofat_plot_kitchen.sh --csv /path/ke/custom.csv
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/ofat_plot_kitchen.py "$@"
