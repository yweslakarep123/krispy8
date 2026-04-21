#!/usr/bin/env bash
# Evaluate a trained FlowPolicy checkpoint on FrankaKitchen-v1.
#
# Example:
#   bash scripts/eval_kitchen.sh 0 0  # seed=0, gpu 0
#
set -euo pipefail

seed=${1:-0}
gpu_id=${2:-0}

task_name=kitchen_complete
alg_name=flowpolicy
config_name=${alg_name}
exp_name=${task_name}-${alg_name}
run_dir="data/outputs/${exp_name}_seed${seed}"

cd "$(dirname "$0")/../FlowPolicy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python eval.py --config-name=${config_name} \
               task=${task_name} \
               hydra.run.dir=${run_dir} \
               training.seed=${seed} \
               training.device="cuda" \
               exp_name=${exp_name} \
               logging.mode=offline
