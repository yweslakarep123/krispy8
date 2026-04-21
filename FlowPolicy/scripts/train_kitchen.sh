#!/usr/bin/env bash
# Train FlowPolicy (low-dim, consistency flow-matching) on FrankaKitchen-v1
# with the `kitchen-complete-v1` Minari dataset.
#
# Examples:
#   bash scripts/train_kitchen.sh 0 0         # seed=0, gpu_id=0
#   bash scripts/train_kitchen.sh 42 0 FALSE  # seed=42, gpu 0, not debug
#
set -euo pipefail

seed=${1:-0}
gpu_id=${2:-0}
DEBUG=${3:-FALSE}
save_ckpt=${4:-True}

task_name=kitchen_complete
alg_name=flowpolicy
config_name=${alg_name}
exp_name=${task_name}-${alg_name}
run_dir="data/outputs/${exp_name}_seed${seed}"

if [ "$DEBUG" = "True" ] || [ "$DEBUG" = "TRUE" ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd "$(dirname "$0")/../FlowPolicy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name} \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=${DEBUG} \
                training.seed=${seed} \
                training.device="cuda" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                checkpoint.save_ckpt=${save_ckpt}
