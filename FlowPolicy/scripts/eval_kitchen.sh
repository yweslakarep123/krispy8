#!/usr/bin/env bash
# Evaluasi checkpoint FlowPolicy di FrankaKitchen-v1 menggunakan `eval.py`.
#
# Usage:
#   bash scripts/eval_kitchen.sh <ckpt_or_run_dir_or_empty> [gpu_id] [seed] [-- <extra hydra overrides>]
#
# Contoh:
#   # 1) Tunjuk file checkpoint langsung
#   bash scripts/eval_kitchen.sh data/outputs/2026.04.23/08.33.13_.../checkpoints/latest.ckpt 0
#
#   # 2) Tunjuk run_dir training (skrip akan menambahkan /checkpoints/latest.ckpt)
#   bash scripts/eval_kitchen.sh data/outputs/2026.04.23/08.33.13_... 0
#
#   # 3) Auto-detect: cari run_dir terbaru di data/outputs/ yang punya
#   #    checkpoints/latest.ckpt (biarkan argumen pertama kosong / "-")
#   bash scripts/eval_kitchen.sh - 0
#   bash scripts/eval_kitchen.sh "" 0
#
#   # 4) Override Hydra tambahan setelah "--"
#   bash scripts/eval_kitchen.sh - 0 0 -- task.env_runner.eval_episodes=50
#
# Catatan: checkpoint hanya tersimpan bila training dijalankan dengan
#   checkpoint.save_ckpt=True (sekarang default di flowpolicy.yaml).

set -euo pipefail

ckpt_arg=${1:-}
gpu_id=${2:-0}
seed=${3:-0}
shift $(( $# >= 3 ? 3 : $# )) || true

# Hapus separator "--" bila ada
if [[ "${1:-}" == "--" ]]; then
    shift
fi
extra_overrides=("$@")

cd "$(dirname "$0")/../FlowPolicy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

resolve_ckpt() {
    local arg="$1"
    if [[ -z "${arg}" || "${arg}" == "-" ]]; then
        # Auto-detect: run_dir terbaru yang punya checkpoints/latest.ckpt.
        # Hindari ketergantungan ke bash globstar (`**`) karena sering nonaktif.
        # Cari di beberapa root output yang umum dipakai:
        #   1) ./data/outputs        (jika cwd sudah repo root)
        #   2) ../data/outputs       (jika cwd di FlowPolicy/FlowPolicy)
        #   3) ../../data/outputs    (jika train.py mengubah cwd ke repo root)
        local latest
        latest=$(python - <<'PY'
import pathlib

roots = [
    pathlib.Path("data/outputs"),
    pathlib.Path("../data/outputs"),
    pathlib.Path("../../data/outputs"),
]
cands = []
for root in roots:
    if not root.exists():
        continue
    for p in root.glob("**/checkpoints/latest.ckpt"):
        try:
            cands.append((p.stat().st_mtime, p.resolve()))
        except OSError:
            pass

if cands:
    cands.sort(key=lambda x: x[0], reverse=True)
    print(cands[0][1])
PY
)
        if [[ -z "${latest}" ]]; then
            echo "ERROR: auto-detect gagal, tidak menemukan file" \
                 "**/checkpoints/latest.ckpt di data/outputs." >&2
            echo "Jalankan training dulu (mis. bash scripts/train_kitchen.sh 0 0)" \
                 "atau sebutkan path ckpt/run_dir sebagai argumen pertama." >&2
            return 1
        fi
        echo "${latest}"
        return 0
    fi
    if [[ -f "${arg}" ]]; then
        echo "${arg}"
        return 0
    fi
    if [[ -d "${arg}" ]]; then
        local cand="${arg%/}/checkpoints/latest.ckpt"
        if [[ -f "${cand}" ]]; then
            echo "${cand}"
            return 0
        fi
    fi
    echo "ERROR: argumen '${arg}' bukan file ckpt dan bukan run_dir yang" \
         "mengandung checkpoints/latest.ckpt." >&2
    return 1
}

ckpt_path=$(resolve_ckpt "${ckpt_arg}")
ckpt_path=$(readlink -f "${ckpt_path}")
run_dir=$(dirname "$(dirname "${ckpt_path}")")

echo "[eval] ckpt    : ${ckpt_path}"
echo "[eval] run_dir : ${run_dir}"
echo "[eval] gpu     : ${gpu_id}, seed: ${seed}"

python eval.py --config-name=flowpolicy \
               task=kitchen_complete \
               hydra.run.dir="${run_dir}" \
               training.seed="${seed}" \
               training.device="cuda" \
               exp_name=kitchen_complete-flowpolicy \
               logging.mode=offline \
               +eval.checkpoint_path="${ckpt_path}" \
               "${extra_overrides[@]}"
