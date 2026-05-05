#!/usr/bin/env bash
# Train FlowPolicy (low-dim, consistency flow-matching) on FrankaKitchen-v1
# with the `D4RL/kitchen/complete-v2` Minari dataset.
#
# Setiap invokasi memakai folder output UNIK (timestamp + random) sehingga
# tidak menimpa run sebelumnya. Hydra/WandB resume dimatikan untuk run mandiri.
#
# Examples:
#   bash scripts/train_kitchen.sh 0 0         # seed=0, gpu_id=0
#   bash scripts/train_kitchen.sh 42 0 FALSE  # seed=42, gpu 0, not debug
#   bash scripts/train_kitchen.sh 0 0 FALSE True minimal  # profil preprocessing ke-5
#   bash scripts/train_kitchen.sh 0 0 FALSE True raw      # profil raw
#
# Argumen: seed gpu [DEBUG] [save_ckpt] [preprocessing_profile]
#
# preprocessing_profile:
#   standard — sliding-window buffer + split 70/20/10 + augmentasi noise (yaml).
#   minimal  — sama, tanpa Gaussian noise.
#   raw      — tanpa windowing (set task.dataset.preprocess.sliding_window=true untuk opt-in), stride horizon, tanpa noise.
#
set -euo pipefail

seed=${1:-0}
gpu_id=${2:-0}
DEBUG=${3:-FALSE}
save_ckpt=${4:-True}
# standard | minimal | legacy_minimal | raw — lihat KitchenDataset / README
preprocessing_profile=${5:-standard}

task_name=kitchen_complete
alg_name=flowpolicy
config_name=${alg_name}
exp_name=${task_name}-${alg_name}

RUN_TS=$(date +%Y%m%d_%H%M%S)
if command -v openssl >/dev/null 2>&1; then
  RUN_RAND=$(openssl rand -hex 4)
else
  RUN_RAND=$(printf '%04x%04x' "${RANDOM}" "${RANDOM}")
fi
# Pola: <exp>_seed<N>_<profil>__<YYYYMMDD_HHMMSS>_<hex> — tidak overwrite run lama
run_slug="${exp_name}_seed${seed}_${preprocessing_profile}__${RUN_TS}_${RUN_RAND}"
run_dir="data/outputs/${run_slug}"

if [ "$DEBUG" = "True" ] || [ "$DEBUG" = "TRUE" ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

START_ISO=$(date -Iseconds 2>/dev/null || date)
WANDB_ID=$(openssl rand -hex 16 2>/dev/null || printf '%s_%s' "${RUN_TS}" "${RUN_RAND}")
wandb_name="${run_slug}"

echo ""
echo -e "\033[35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo -e "\033[32m  Mulai training\033[0m  ${START_ISO}"
echo -e "  host ...................... $(hostname 2>/dev/null || echo '(n/a)')"
echo -e "  GPU (CUDA_VISIBLE_DEVICES)  ${gpu_id}"
echo -e "  seed / preprocessing ...... ${seed} / ${preprocessing_profile}"
echo -e "  run_id (WandB) ............ ${WANDB_ID}"
echo -e "  run_slug .................. ${run_slug}"
echo -e "\033[36m  output_dir (relatif) .... ${run_dir}\033[0m"
echo -e "\033[35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo ""

cd "$(dirname "$0")/../FlowPolicy"
mkdir -p "${run_dir}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name} \
                task=${task_name} \
                task.dataset.preprocessing_profile=${preprocessing_profile} \
                hydra.run.dir=${run_dir} \
                training.resume=false \
                training.debug=${DEBUG} \
                training.seed=${seed} \
                training.device="cuda" \
                exp_name=${exp_name} \
                logging.name=${wandb_name} \
                logging.mode=${wandb_mode} \
                logging.resume=false \
                logging.id=${WANDB_ID} \
                checkpoint.save_ckpt=${save_ckpt}

END_ISO=$(date -Iseconds 2>/dev/null || date)
echo ""
echo -e "\033[35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo -e "\033[32m  Selesai training\033[0m  ${END_ISO}"
echo -e "\033[36m  output_dir: ${run_dir}\033[0m"
echo -e "\033[35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo ""
