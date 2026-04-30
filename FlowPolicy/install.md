# Installation (Franka Kitchen / low-dim variant)

This repo was ported from the original 3D point-cloud FlowPolicy (Adroit /
Metaworld) to a low-dim state-based policy for **FrankaKitchen-v1**
(`gymnasium-robotics`) trained on Minari `kitchen-complete-v1`.
It no longer needs `mujoco-py`, `pytorch3d`, `open3d`, Metaworld, VRL3, or
Gym 0.21.

## 1. Conda env

```bash
conda create -n flowp python=3.12 -y
conda activate flowp
```

## 2. PyTorch (pick the CUDA build matching your driver)

```bash
# example for CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

## 3. Core Python deps

```bash
pip install \
    gymnasium==1.2.3 \
    gymnasium-robotics==1.4.2 \
    minari==0.5.3 \
    mujoco==3.7.0 \
    'hydra-core>=1.3.2' \
    omegaconf \
    dill einops termcolor \
    wandb \
    zarr numcodecs numba \
    tqdm scipy \
    h5py pyarrow \
    moviepy imageio imageio-ffmpeg
```

### Paket `pip` yang perlu diinstall (ringkas)

Supaya lebih jelas, ini daftar paket `pip` yang dipakai:

- **Deep learning (GPU):**
  - `torch==2.6.0`
  - `torchvision==0.21.0`
- **RL env + dataset kitchen:**
  - `gymnasium==1.2.3`
  - `gymnasium-robotics==1.4.2`
  - `minari==0.5.3`
  - `mujoco==3.7.0`
- **Config & serialization:**
  - `hydra-core>=1.3.2`
  - `omegaconf`
  - `dill`
- **Model/training utilities:**
  - `einops`
  - `termcolor`
  - `tqdm`
  - `scipy`
  - `wandb`
- **Data utilities:**
  - `zarr`
  - `numcodecs`
  - `numba`
  - `h5py`
  - `pyarrow`
- **Video/export:**
  - `moviepy`
  - `imageio`
  - `imageio-ffmpeg`

### Opsi cepat: sekali install semua `pip` package

Jika Anda ingin 1 command langsung:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install gymnasium==1.2.3 gymnasium-robotics==1.4.2 minari==0.5.3 mujoco==3.7.0 "hydra-core>=1.3.2" omegaconf dill einops termcolor wandb zarr numcodecs numba tqdm scipy h5py pyarrow moviepy imageio imageio-ffmpeg
```

Jika CUDA Anda bukan 12.4, ganti index URL PyTorch sesuai versi CUDA Anda.

`minari[gcs]` is already pulled in by `minari` 0.5.3; the Franka Kitchen
dataset will be fetched automatically on first use.

## 4. Install this package

```bash
cd FlowPolicy && pip install -e . && cd ..
```

## 5. Download the Minari dataset (first run only)

```bash
python -c "import minari; minari.load_dataset('D4RL/kitchen/complete-v2', download=True)"
```

## 6. Weights & Biases

```bash
wandb login
```

Now you are ready to train/evaluate; see [README.md](README.md).
