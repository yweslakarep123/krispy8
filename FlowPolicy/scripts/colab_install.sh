#!/usr/bin/env bash
# Install semua dependensi untuk menjalankan FlowPolicy (low-dim) di
# Google Colab (Free T4 atau Pro).
#
# Asumsi:
#   - Colab sudah menyediakan python 3.10-3.12 + torch dengan CUDA.
#   - Anda sudah `!git clone` repo ini dan cwd-nya ada di root repo
#     (folder yang berisi subfolder FlowPolicy/ dan scripts/).
#
# Pemakaian (Colab cell):
#   %%bash
#   cd /content/<nama_repo> && bash scripts/colab_install.sh
#
set -euo pipefail

echo "[colab-install] python: $(python --version 2>&1)"
echo "[colab-install] torch  : $(python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo 'not installed')"

# ----- 1. Core deps ---------------------------------------------------------
# Versi-versi ini disamakan dengan env lokal 'flowp' (linux + py3.12 + cu124).
# Colab sudah punya torch (biasanya >= 2.3 + cu12x) — tidak di-override agar
# tidak bentrok dengan driver T4.
pip install -q \
    "optuna>=3.0" \
    "gymnasium==1.2.3" \
    "gymnasium-robotics==1.4.2" \
    "minari==0.5.3" \
    "mujoco==3.7.0" \
    "hydra-core>=1.3.2" \
    "omegaconf" \
    "dill" "einops" "termcolor" \
    "wandb" \
    "zarr" "numcodecs" "numba" \
    "tqdm" "scipy" \
    "h5py" "pyarrow" \
    "moviepy" "imageio" "imageio-ffmpeg"

# ----- 2. Install package lokal (flow_policy_3d) ---------------------------
pushd FlowPolicy > /dev/null
pip install -q -e .
popd > /dev/null

# ----- 3. (Sekali saja) download dataset Minari -----------------------------
# Download offline ke ~/.minari; aman dipanggil berulang (idempotent).
python -c "
import minari, sys
try:
    ds = minari.load_dataset('D4RL/kitchen/complete-v2', download=True)
    print('[colab-install] minari dataset OK, episodes=', len(list(ds.iterate_episodes())))
except Exception as e:
    print('[colab-install] minari load failed:', e, file=sys.stderr)
    sys.exit(1)
"

# ----- 4. (Opsional) OpenGL render headless --------------------------------
# Colab sudah punya libEGL; gymnasium-robotics FrankaKitchen-v1 render
# dengan mujoco off-screen (MUJOCO_GL=egl). Kita set sekali di sini.
if ! grep -q "MUJOCO_GL=egl" ~/.bashrc 2>/dev/null; then
    echo 'export MUJOCO_GL=egl' >> ~/.bashrc
fi
export MUJOCO_GL=egl

echo "[colab-install] selesai. env siap untuk training & inferensi."
