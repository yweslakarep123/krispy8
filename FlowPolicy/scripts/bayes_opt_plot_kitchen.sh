#!/usr/bin/env bash
# Plot riwayat BO dari trials.csv
#   bash scripts/bayes_opt_plot_kitchen.sh
#   bash scripts/bayes_opt_plot_kitchen.sh --preprocess
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/bayes_opt_plot_kitchen.py "$@"
