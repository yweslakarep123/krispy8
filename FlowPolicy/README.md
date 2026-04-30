# FlowPolicy Kitchen - Cara Menjalankan Eksperimen

Dokumen ini menjelaskan cara menjalankan eksperimen `3 seed x 2 scenario` dengan alur:
- random search hyperparameter (`n_iter`, `cv`)
- train final model terbaik per scenario
- inferensi evaluasi
- inferensi final untuk global winner
- generate plot performa

Scenario yang dijalankan per seed:
- `tuned_no_preprocess`
- `tuned_preprocess`

## 1) Jalankan dari root repository

Pastikan perintah dijalankan dari:
- `C:/Users/Thinkpad/Downloads/krispy8`

## 2) Command utama (sesuai skenario Anda)

Menjalankan eksperimen penuh dengan:
- seed: `0 42 101`
- random search: `n_iter=100`
- cross-validation: `cv=5`
- simpan video final global winner
- strict GPU aktif default (akan gagal jika CUDA tidak tersedia)

```bash
py -3 FlowPolicy/scripts/experiment_3seed_4arms.py --seeds 0 42 101 --random-n-iter 100 --random-cv 5 --hero-video
```

Jika ingin mengizinkan fallback CPU (tidak disarankan):

```bash
py -3 FlowPolicy/scripts/experiment_3seed_4arms.py --seeds 0 42 101 --random-n-iter 100 --random-cv 5 --hero-video --no-strict-gpu
```

## 3) Opsi cepat untuk pengecekan alur (dry-run)

Tidak menjalankan training/inferensi sungguhan, hanya validasi urutan command:

```bash
py -3 FlowPolicy/scripts/experiment_3seed_4arms.py --seeds 0 --random-n-iter 2 --random-cv 2 --eval-episodes 2 --hero-episodes 2 --dry-run
```

## 4) Arti output yang dihasilkan

Default output ada di:
- `FlowPolicy/FlowPolicy/data/outputs/exp_3seed_4arms`

File penting:
- `all_models_eval.csv`: hasil evaluasi semua scenario/model lintas seed
- `seed_winners.json`: pemenang per seed
- `global_winner.json`: model terbaik global
- `final_summary.json`: ringkasan inferensi final global winner
- `plot_summary.json`: ringkasan path file plot

Plot yang dibuat:
- `plots/success_rate_all_models.png`
- `plots/global_winner_sr_latency.png`

## 5) Script yang dipakai di balik orkestrasi

- `FlowPolicy/scripts/random_search_kitchen.py`
  - random search per seed/mode preprocess
  - output utama: `results.csv` dan `best_trial.json`
- `FlowPolicy/scripts/train_best_from_random.py`
  - training model final dari `best_trial.json`
- `FlowPolicy/scripts/experiment_3seed_4arms.py`
  - orkestrasi end-to-end + plotting
  - menampilkan progress `[progress] ...` saat run

## 6) Monitoring progress saat run

Saat eksperimen berjalan, terminal menampilkan progres seperti:
- `[progress] 3/22 (13.6%) - seed0 scenario=tuned_no_preprocess train_best`
- `[random_search][progress] 12/500 (2.4%) cfg=3/100 cv=2/5`

Artinya:
- progress orkestrasi global (seed/scenario/eval/final)
- progress detail random search (kandidat dan fold CV yang sedang diproses)

## 7) GPU mode dan verifikasi

Script sekarang memaksa mode GPU dan optimasi performa:
- set `training.device=cuda`
- `strict GPU`: stop jika CUDA tidak tersedia
- `FLOWP_GPU_PERF_MODE=1` mengaktifkan:
  - `torch.backends.cudnn.benchmark=True`
  - `torch.backends.cuda.matmul.allow_tf32=True`
  - `torch.backends.cudnn.allow_tf32=True`

Verifikasi saat run:
- akan muncul log `[gpu-check] CUDA ready ...`
- monitor utilitas GPU dengan:

```bash
nvidia-smi -l 1
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

**Cell 4 - jalankan OFAT sweep dengan budget waktu (Colab free timeout 12h,
sisakan buffer untuk summary):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
export MUJOCO_GL=egl
bash scripts/ofat_search_kitchen.sh 0 --max-minutes 660
```

Bila runtime Colab restart sebelum 96 run selesai, cell 4 bisa dijalankan
ulang — skrip otomatis skip `(cfg, seed)` yang sudah punya `metrics.json`
valid (berkat symlink ke Drive, progres tidak hilang).

**Cell 5 - inspect hasil:**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
ls data/outputs/ofat_search/
echo "----- summary.csv (top 10) -----"
head -n 11 data/outputs/ofat_search/summary.csv
```

**Cell 6 (opsional) - inferensi 50 episode manual dari 1 checkpoint:**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
python infer_kitchen.py \
  --checkpoint data/outputs/ofat_search/cfg_00_seed0/checkpoints/latest.ckpt \
  --episodes 50 --device cuda:0
```

## Random search hyperparameter (alternatif)

Skrip `scripts/random_search_kitchen.py` melakukan random search yang
**sekaligus men-train model** untuk setiap kombinasi `(cfg_idx, seed)`,
kemudian langsung inferensi/evaluasi.

Default: 30 konfigurasi acak × 3 seed `[0, 42, 101]`, evaluasi 50 episode
per run.

Ruang sampling:

| hyperparameter                                   | nilai                                |
|--------------------------------------------------|--------------------------------------|
| `training.num_epochs`                            | `[500, 1000, 3000, 5000]`            |
| `optimizer.lr`                                   | `[1e-5, 1e-4, 5e-4, 1e-3]`           |
| `dataloader.batch_size`                          | `[64, 128, 256, 512]`                |
| `training.lr_warmup_steps`                       | `[200, 500, 1000, 2000]`             |
| `training.ema_decay`                             | `[0.90, 0.95, 0.99, 0.999]`          |
| `policy.encoder_output_dim`                      | `[32, 64, 128, 256]`                 |
| `policy.Conditional_ConsistencyFM.num_segments`  | `[1, 2, 3, 4]`                       |
| `policy.Conditional_ConsistencyFM.boundary`      | `[0.5, 1.0, 2.0, 4.0]`               |
| `policy.Conditional_ConsistencyFM.delta`         | `[1e-3, 1e-2, 1e-1, 1.0]`            |
| `policy.Conditional_ConsistencyFM.alpha`         | `[1e-6, 1e-5, 1e-4, 1e-3]`           |
| `policy.Conditional_ConsistencyFM.eps`           | `[1e-4, 1e-3, 1e-2, 1e-1]`           |
| `policy.Conditional_ConsistencyFM.num_inference_step` | `[1, 2, 3, 5]`                 |
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

Contoh 100 iterasi:

```bash
cd FlowPolicy
python scripts/random_search_kitchen.py \
  --n-configs 100 \
  --seeds 0 42 101 \
  --episodes 50 \
  --sampling-seed 42 \
  --gpu 0
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

### Alur setelah tuning

Tidak perlu retrain terpisah jika checkpoint dari random search sudah ada dan
valid. Langkah yang disarankan:

1. Pilih `cfg_idx` terbaik dari `summary.csv` (baris teratas, karena sudah
   diurutkan berdasarkan `test_mean_score_mean`).
2. Langsung inferensi checkpoint terbaik untuk seed yang diinginkan.

Contoh inferensi dari hasil random search (mis. `cfg_17_seed0`):

```bash
cd FlowPolicy/FlowPolicy
python infer_kitchen.py \
  --checkpoint data/outputs/random_search/cfg_17_seed0/checkpoints/latest.ckpt \
  --episodes 50 \
  --device cuda:0
```

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

Catatan: penggunaan GPU tidak selalu 100% setiap detik karena bisa dibatasi pipeline environment/data loading, tetapi konfigurasi ini memastikan eksekusi utama berjalan di CUDA dan di-tune untuk throughput.
