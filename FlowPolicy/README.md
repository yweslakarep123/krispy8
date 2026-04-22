# FlowPolicy (low-dim) for Franka Kitchen

Low-dim state-based adaptation of [FlowPolicy (AAAI 2025)](https://arxiv.org/abs/2412.04987)
for the **FrankaKitchen-v1** environment (`gymnasium-robotics`), trained with
behaviour cloning on the Minari `D4RL/kitchen/complete-v2` dataset.

The consistency flow-matching formulation and `ConditionalUnet1D` backbone of
the original repo are kept unchanged. All 3D point-cloud / PointNet /
Metaworld / Adroit code paths, along with the `mujoco-py`, `pytorch3d`,
`open3d`, and old-`gym` dependencies, have been removed.

## Observation / action

The policy input is a single low-dim vector per timestep:

```
agent_pos_t = concat(obs_dict['observation'],
                     flatten(obs_dict['desired_goal']))
```

with the `desired_goal` dict flattened in **alphabetical key order** so the
same layout is used during training (Minari) and rollout (`gymnasium`).

For `tasks_to_complete = [microwave, kettle, light switch, slide cabinet]`:

| piece              | dim |
|--------------------|-----|
| `observation`      |  59 |
| `desired_goal`     |  11 (kettle 7 + light switch 2 + microwave 1 + slide cabinet 1) |
| **agent_pos**      |  **70** |
| action             |   9 (Box[-1, 1]) |

## Installation

See [install.md](install.md).

## Training

```bash
bash scripts/train_kitchen.sh 0 0        # seed 0, GPU 0
```

Under the hood this runs:

```bash
python FlowPolicy/train.py --config-name=flowpolicy task=kitchen_complete ...
```

Logs are pushed to Weights & Biases (project `flowpolicy_kitchen`).

## Evaluation

```bash
bash scripts/eval_kitchen.sh 0 0
```

The evaluation runner instantiates FrankaKitchen-v1 via `gymnasium.make`,
computes the fraction of `tasks_to_complete` completed per episode (averaged
over `eval_episodes`), and (optionally) logs a few rollout videos.

## Urutan sub-task & filter demonstrasi

Task yaml (`FlowPolicy/flow_policy_3d/config/task/kitchen_complete.yaml`)
mendefinisikan `tasks_to_complete` dengan urutan:

```
[microwave, kettle, light switch, slide cabinet]
```

Urutan ini dipakai di dua tempat:

1. **Layout vektor goal** pada `agent_pos`. `KitchenDataset` dan
   `KitchenRunner` sama-sama memakai urutan ini (bukan alfabetis seperti
   versi sebelumnya) agar komposisi `agent_pos_t` di training dan rollout
   konsisten.
2. **Filter demonstrasi**. Hanya episode yang menyelesaikan keempat task
   persis dalam urutan di atas yang dipakai untuk training. Set
   `dataset.enforce_task_order=false` untuk menonaktifkan.

## Augmentasi data (generalisasi)

Untuk mencegah model menghafal trajektori, tersedia dua augmentasi (hanya
diterapkan pada split training):

| yaml field                   | default | keterangan                                   |
|------------------------------|---------|----------------------------------------------|
| `dataset.obs_noise_std`      | `0.01`  | Gaussian noise di 59 dim observation state   |
| `dataset.action_noise_std`   | `0.0`   | Gaussian noise pada target action            |
| `dataset.normalizer_mode`    | `limits`| `limits` atau `gaussian`                     |

Override via Hydra, mis. `dataset.obs_noise_std=0.02`.

## OFAT hyperparameter sweep (8 HP x 4 nilai x 3 seed = 96 run)

Skrip `scripts/ofat_search_kitchen.py` meng-eksplorasi **setiap** hyperparameter
satu per satu (One-Factor-At-a-Time), menahan yang lain di nilai baseline.

Baseline:

| hyperparameter                                  | baseline |
|-------------------------------------------------|----------|
| `training.num_epochs`                           | 3000     |
| `optimizer.lr`                                  | 1e-4     |
| `dataloader.batch_size`                         | 128      |
| `policy.Conditional_ConsistencyFM.num_segments` | 2        |
| `policy.Conditional_ConsistencyFM.eps`          | 1e-2     |
| `policy.Conditional_ConsistencyFM.delta`        | 1e-2     |
| `n_action_steps`                                | 4        |
| `n_obs_steps`                                   | 4        |

Untuk tiap hyperparameter, 4 nilai dari ruang sampling dijalankan (total 32
konfigurasi), tiap konfigurasi di 3 seed `[0, 42, 101]`, evaluasi 50 episode.

Menjalankan:

```bash
cd FlowPolicy
bash scripts/ofat_search_kitchen.sh 0 --dry-run          # cek 32 konfigurasi
bash scripts/ofat_search_kitchen.sh 0                    # full sweep di GPU 0
bash scripts/ofat_search_kitchen.sh 0 --only-hp optimizer.lr   # hanya sweep lr
bash scripts/ofat_search_kitchen.sh 0 --max-minutes 600   # stop setelah 10 jam (Colab T4)
```

Keluaran di `data/outputs/ofat_search/`:

- `configs.json` — 32 konfigurasi unik
- `cfg_<ii>_seed<s>/` — run_dir per (konfigurasi, seed)
- `results.csv` — satu baris per run (96 bila lengkap)
- `summary.csv` — mean +/- std per konfigurasi di 3 seed, diurutkan dari
  `test_mean_score` tertinggi ke terendah

Resume otomatis via `metrics.json` di tiap run_dir.

## Menjalankan di Google Colab (Free T4)

Semua cell di Colab hanya menjalankan perintah bash; tidak perlu menulis
ulang kode Python.

**Cell 1 - clone repo (ganti dengan URL repo GitHub Anda):**

```bash
%%bash
cd /content
rm -rf flowpolicy_kitchen
git clone https://github.com/<USER>/<REPO>.git flowpolicy_kitchen
```

**Cell 2 - install dependencies + download dataset Minari:**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
bash scripts/colab_install.sh
```

**Cell 3 - (opsional, sangat disarankan) mount Google Drive agar hasil
persisten antar restart Colab:**

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!mkdir -p /content/drive/MyDrive/flowpolicy_kitchen_outputs
!rm -rf /content/flowpolicy_kitchen/FlowPolicy/data/outputs/ofat_search
!ln -s /content/drive/MyDrive/flowpolicy_kitchen_outputs /content/flowpolicy_kitchen/FlowPolicy/data/outputs/ofat_search
```

**Cell 4 - jalankan OFAT sweep DI BACKGROUND supaya bisa dipantau dari cell
lain (Colab `%%bash` membuffer output saat running).** `--max-minutes 660`
memberi budget 11 jam (sisakan buffer untuk timeout 12h Colab Free).

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
export MUJOCO_GL=egl
# kill sisa proses kalau ada, lalu jalankan di background
pkill -f ofat_search_kitchen.py 2>/dev/null || true
nohup bash scripts/ofat_search_kitchen.sh 0 --max-minutes 660 \
    > /tmp/ofat_master.log 2>&1 &
sleep 2
echo "Sweep PID: $(pgrep -f ofat_search_kitchen.py || echo '(tidak jalan)')"
echo "Master log: /tmp/ofat_master.log"
echo "Progress log : data/outputs/ofat_search/progress.log"
```

**Cell 5 - pantau progress satu baris per run (paling ringkas):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
# Tampilkan hanya baris [PROGRESS n/96] dari progress.log, live
grep -n "^\[PROGRESS" data/outputs/ofat_search/progress.log 2>/dev/null | tail -n 30
echo "---"
echo "Jumlah selesai: $(grep -c '^\[PROGRESS' data/outputs/ofat_search/progress.log 2>/dev/null || echo 0) / 96"
echo "Sweep masih jalan? $(pgrep -f ofat_search_kitchen.py >/dev/null && echo YES || echo NO)"
```

Jalankan cell 5 berulang kali (Ctrl+Enter) untuk refresh. Contoh output:
```
[PROGRESS   1/96] cfg_00 seed=0   [training.num_epochs=500] SR=0.2150 status=ok t_train=  420s t_infer=  55s elapsed=00:07:55 eta=12:25:00
[PROGRESS   2/96] cfg_00 seed=42  [training.num_epochs=500] SR=0.1875 status=ok ...
```

**Cell 6 - live tail master log (output penuh dari semua training/inferensi):**

```bash
%%bash
tail -n 60 /tmp/ofat_master.log
```

**Cell 7 - lihat log run aktif paling baru (train/infer):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy/data/outputs/ofat_search
latest=$(ls -td cfg_* 2>/dev/null | head -n 1)
echo "Run aktif terbaru: $latest"
echo "===== train_stdout.log (tail) ====="
tail -n 40 "$latest/train_stdout.log" 2>/dev/null || echo "(belum ada)"
echo "===== infer_stdout.log (tail) ====="
tail -n 40 "$latest/infer_stdout.log" 2>/dev/null || echo "(belum ada)"
```

**Cell 8 - setelah selesai: cek summary top-10:**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
ls data/outputs/ofat_search/
echo "----- summary.csv (top 10) -----"
head -n 11 data/outputs/ofat_search/summary.csv
```

**Cell 9 (opsional) - hentikan sweep di tengah jalan:**

```bash
%%bash
pkill -f ofat_search_kitchen.py && echo "dihentikan" || echo "tidak ada proses"
```

Bila runtime Colab restart sebelum 96 run selesai, jalankan ulang cell 4
— skrip otomatis skip `(cfg, seed)` yang sudah punya `metrics.json` valid.

**Cell 10 (opsional) - inferensi 50 episode manual dari 1 checkpoint:**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
python infer_kitchen.py \
  --checkpoint data/outputs/ofat_search/cfg_00_seed0/checkpoints/latest.ckpt \
  --episodes 50 --device cuda:0
```

## Random search hyperparameter (alternatif)

Skrip `scripts/random_search_kitchen.py` melakukan random search pada 8
hyperparameter, 30 konfigurasi acak × 3 seed `[0, 42, 101]`, dengan
evaluasi akhir 50 episode per run.

Ruang sampling:

| hyperparameter                                   | nilai                                |
|--------------------------------------------------|--------------------------------------|
| `training.num_epochs`                            | `[500, 1000, 3000, 5000]`            |
| `optimizer.lr`                                   | `[1e-3, 5e-4, 1e-4, 1e-5]`           |
| `dataloader.batch_size`                          | `[64, 128, 256, 512]`                |
| `policy.Conditional_ConsistencyFM.num_segments`  | `[1, 2, 3, 4]`                       |
| `policy.Conditional_ConsistencyFM.eps`           | `[1e-2, 1e-3, 1e-4, 1.0]`            |
| `policy.Conditional_ConsistencyFM.delta`         | `[1e-2, 1e-3, 1e-4, 1.0]`            |
| `n_action_steps`                                 | `[2, 4, 6, 8]`                       |
| `n_obs_steps`                                    | `[4, 6, 8, 16]`                      |

`horizon` dihitung otomatis sebagai `max(n_obs_steps + n_action_steps - 1, 4)`.

Menjalankan:

```bash
cd FlowPolicy
# dry-run dulu untuk inspeksi 30 konfigurasi
bash scripts/random_search_kitchen.sh 0 --dry-run

# eksekusi sebenarnya di GPU 0
bash scripts/random_search_kitchen.sh 0
```

Keluaran di `data/outputs/random_search/`:

- `configs.json` — daftar 30 konfigurasi yang di-sample (stabil selama
  `--sampling-seed` tidak berubah).
- `cfg_<ii>_seed<s>/` — run_dir per (konfigurasi, seed). Berisi
  `checkpoints/latest.ckpt`, `train_stdout.log`, `infer_stdout.log`, dan
  `inference_ep50/metrics.json`.
- `results.csv` — satu baris per run (90 baris bila lengkap).
- `summary.csv` — ringkasan per konfigurasi: mean ± std `test_mean_score`,
  `mean_n_completed_tasks`, `mean_time` di 3 seed, **diurutkan** dari
  terbaik ke terburuk.

Resume otomatis: jika `metrics.json` untuk `(cfg_idx, seed)` tertentu
sudah valid, skrip melewatinya.

## Inferensi (checkpoint → video + metrik)

Setelah training, jalankan inferensi mandiri (tanpa WandB kecuali `--wandb`):

```bash
cd FlowPolicy
python infer_kitchen.py \
  --checkpoint data/outputs/kitchen_complete-flowpolicy_seed0/checkpoints/latest.ckpt \
  --episodes 10 \
  --device cuda:0
```

Atau lewat skrip:

```bash
bash scripts/infer_kitchen.sh data/outputs/kitchen_complete-flowpolicy_seed0/checkpoints/latest.ckpt 0 10
```

Keluaran default: folder `data/outputs/<run>/inference_latest/` berisi:

- `videos/episode_000.mp4`, `episode_001.mp4`, … (satu file per episode)
- `metrics.json` — antara lain `test_mean_score` (success rate), `mean_time` (latensi rata-rata **per panggilan** `predict_action` dalam detik)

Catatan: saya tidak bisa melampirkan file video di chat; jalankan perintah di atas di mesin Anda lalu buka file MP4 di folder `videos/`.

## Layout

```
FlowPolicy/
  FlowPolicy/                             # python package
    flow_policy_3d/
      config/
        flowpolicy.yaml                   # main config (low-dim)
        task/kitchen_complete.yaml        # task config
      dataset/kitchen_dataset.py          # Minari -> ReplayBuffer
      env_runner/kitchen_runner.py        # gymnasium FrankaKitchen-v1 rollout
      gym_util/multistep_wrapper.py       # gymnasium framestack wrapper
      model/
        flow/                             # UNet1D + ConsistencyFM (kept)
        vision/lowdim_encoder.py          # MLP state encoder
      policy/flowpolicy_lowdim.py         # low-dim FlowPolicy
      sde_lib.py                          # ConsistencyFM (kept)
    train.py
    eval.py
    infer_kitchen.py                      # inferensi + simpan video lokal
  scripts/
    train_kitchen.sh
    eval_kitchen.sh
    infer_kitchen.sh
```

## Acknowledgements

Based on the original
[FlowPolicy](https://github.com/zql-kk/FlowPolicy) by Zhang et al. (AAAI 2025)
and `Consistency_FM`. If you use this code, please cite the original paper:

```
@article{zhang2024flowpolicy,
  title={FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via
         Consistency Flow Matching for Robot Manipulation},
  author={Qinglun Zhang and Zhen Liu and Haoqiang Fan and Guanghui Liu and
          Bing Zeng and Shuaicheng Liu},
  year={2024},
  eprint={2412.04987},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2412.04987}
}
```

## License

MIT (see [LICENSE](LICENSE)).
