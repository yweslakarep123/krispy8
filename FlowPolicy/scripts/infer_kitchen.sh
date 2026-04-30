#!/usr/bin/env bash
# Inferensi + simpan video + metrics.json.
# Usage:
#   bash scripts/infer_kitchen.sh <path/to/latest.ckpt|run_dir> [gpu_id] [episodes]
#
# Contoh:
#   bash scripts/infer_kitchen.sh data/outputs/kitchen_complete-flowpolicy_seed0/checkpoints/latest.ckpt 0
#   bash scripts/infer_kitchen.sh data/outputs/kitchen_complete-flowpolicy_seed0 0
set -euo pipefail

CKPT_ARG=${1:?usage: $0 <path/to/latest.ckpt|run_dir> [gpu_id] [episodes]}
GPU=${2:-0}
N_EP=${3:-10}

CALLER_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PKG_DIR="${REPO_ROOT}/FlowPolicy"

resolve_ckpt() {
    local arg="$1"

    python - "$arg" "$CALLER_CWD" "$REPO_ROOT" "$PKG_DIR" <<'PY'
import pathlib, sys

arg = sys.argv[1]
caller = pathlib.Path(sys.argv[2]).resolve()
repo = pathlib.Path(sys.argv[3]).resolve()
pkg = pathlib.Path(sys.argv[4]).resolve()
inp = pathlib.Path(arg).expanduser()

def emit(p: pathlib.Path):
    print(str(p.resolve()))
    raise SystemExit(0)

def ckpt_if_valid(path: pathlib.Path):
    if path.is_file():
        emit(path)
    if path.is_dir():
        cand = path / "checkpoints" / "latest.ckpt"
        if cand.is_file():
            emit(cand)

cands = []
if inp.is_absolute():
    cands.append(inp)
else:
    # Prioritas:
    # 1) relatif terhadap folder tempat user menjalankan command
    # 2) relatif terhadap parent folder caller (umum saat output ada di ../data/outputs)
    # 3) relatif terhadap root repo (FlowPolicy/) dan parent-nya
    # 4) relatif terhadap package dir (FlowPolicy/FlowPolicy/) dan parent-nya
    cands.extend([
        caller / inp,
        caller.parent / inp,
        repo / inp,
        repo.parent / inp,
        pkg / inp,
        pkg.parent / inp,
    ])

for c in cands:
    ckpt_if_valid(c)

raise SystemExit(1)
PY
}

if ! CKPT="$(resolve_ckpt "${CKPT_ARG}")"; then
    echo "ERROR: tidak menemukan checkpoint dari argumen '${CKPT_ARG}'." >&2
    echo "Coba berikan path absolut ke latest.ckpt, misalnya:" >&2
    echo "  /home/user/krispy8/data/outputs/<run>/checkpoints/latest.ckpt" >&2
    exit 1
fi

cd "${PKG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
python infer_kitchen.py --checkpoint "${CKPT}" --episodes "${N_EP}" --device cuda:0
