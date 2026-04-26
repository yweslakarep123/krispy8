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

with the `desired_goal` dict flattened mengikuti urutan `tasks_to_complete`
(default: `microwave -> kettle -> light switch -> slide cabinet`) agar layout
state konsisten antara training (Minari) dan rollout (`gymnasium`).

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

Default `checkpoint.save_ckpt: True` sekarang diaktifkan, sehingga
menjalankan `python FlowPolicy/train.py ...` langsung pun akan menyimpan
`latest.ckpt` di `<run_dir>/checkpoints/` (Hydra default run_dir-nya
`data/outputs/<tanggal>/<jam>_train_flowpolicy_lowdim_kitchen_complete/`).
Checkpoint ini yang dipakai oleh `eval_kitchen.sh` dan
`infer_kitchen.sh`.

## Evaluation

`eval_kitchen.sh` menerima path **ckpt** atau **run_dir** training sebagai
argumen pertama (boleh juga kosong / `-` untuk auto-detect run_dir terbaru
yang punya `checkpoints/latest.ckpt`):

```bash
# 1) tunjuk file checkpoint langsung
bash scripts/eval_kitchen.sh data/outputs/2026.04.23/08.33.13_.../checkpoints/latest.ckpt 0

# 2) tunjuk run_dir training (skrip auto menambahkan /checkpoints/latest.ckpt)
bash scripts/eval_kitchen.sh data/outputs/2026.04.23/08.33.13_... 0

# 3) auto-detect: cari run_dir terbaru di data/outputs/
bash scripts/eval_kitchen.sh - 0

# 4) override Hydra tambahan setelah "--"
bash scripts/eval_kitchen.sh - 0 0 -- task.env_runner.eval_episodes=50
```

Prasyarat: training harus menyimpan checkpoint. Sejak pembaruan ini,
default `checkpoint.save_ckpt=True` sudah dinyalakan di
`flow_policy_3d/config/flowpolicy.yaml`, jadi menjalankan
`python FlowPolicy/train.py ...` secara manual pun akan menghasilkan
`<run_dir>/checkpoints/latest.ckpt`. Bila checkpoint tidak ditemukan,
`eval.py` akan gagal cepat dengan pesan yang jelas (bukan lagi silent
skip yang berakhir `AttributeError 'ParameterDict' ... agent_pos`).

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

Override via Hydra, mis. `task.dataset.obs_noise_std=0.02`.

## Preprocessing sliding-window (opsional)

Secara default pipeline lama yang dipakai: setiap demonstrasi Minari jadi
satu trajectory, `SequenceSampler` menarik potongan panjang `horizon`
dengan stride 1, dan split hanya `val_ratio` episode untuk validasi
(tidak ada test split).

Bila `task.dataset.preprocess.enabled=true`, alur pra-pelatihan berubah jadi:

1. **Sliding window per demonstrasi.** Untuk setiap episode panjang `T`,
   dipotong jadi window berukuran `round(window_ratio * T)` dengan stride
   `stride` (default 1 -> setiap timestep jadi awal window baru). Setiap
   window jadi "episode" independen di `ReplayBuffer`.
2. **Split 70/20/10.** Semua window di-shuffle (`split_seed`) lalu dibagi
   menjadi `train_ratio` / `val_ratio` / `test_ratio` (default
   `0.70 / 0.20 / 0.10`).
3. **Augmentasi noise hanya di train.** `obs_noise_std` dan
   `action_noise_std` diterapkan sesudah split, dan hanya untuk sample
   milik training split (validasi/test tetap bersih).

Konfigurasi di `FlowPolicy/flow_policy_3d/config/task/kitchen_complete.yaml`:

```yaml
dataset:
  preprocess:
    enabled: false            # true = aktifkan sliding window + 70/20/10
    window_ratio: 0.25        # ukuran window = 25% panjang episode
    stride: 1
    train_ratio: 0.70
    val_ratio: 0.20
    test_ratio: 0.10
    split_seed: 42
```

Override cepat via Hydra. Blok `dataset` tinggal di dalam `task`, jadi
path override-nya `task.dataset.preprocess.*` (bukan
`dataset.preprocess.*`):

```bash
# aktifkan preprocessing
python FlowPolicy/train.py --config-name=flowpolicy \
  task=kitchen_complete \
  task.dataset.preprocess.enabled=true

# matikan preprocessing (default)
python FlowPolicy/train.py --config-name=flowpolicy \
  task=kitchen_complete \
  task.dataset.preprocess.enabled=false

# contoh override rasio/window
python FlowPolicy/train.py --config-name=flowpolicy \
  task=kitchen_complete \
  task.dataset.preprocess.enabled=true \
  task.dataset.preprocess.window_ratio=0.25 \
  task.dataset.preprocess.stride=1 \
  task.dataset.preprocess.train_ratio=0.70 \
  task.dataset.preprocess.val_ratio=0.20 \
  task.dataset.preprocess.test_ratio=0.10
```

Catatan:

- Evaluasi utama (`test_mean_score`) tetap dihitung dari rollout di
  `FrankaKitchen-v1`, bukan dari split test window. Split test window
  tersedia lewat `dataset.get_test_dataset()` untuk evaluasi offline
  (loss BC murni) bila dibutuhkan.
- Window ratio yang terlalu kecil (misal 0.05) bisa menghasilkan window
  lebih pendek dari `horizon` dan menyebabkan padding konstan agresif di
  `SequenceSampler`. Pastikan `round(window_ratio * T) >= horizon`.

## HalvingRandomSearchCV (hyperparameter tuning)

Pipeline tuning hyperparameter sekarang memakai `HalvingRandomSearchCV`
(`sklearn.model_selection`) melalui:
`scripts/halving_search_kitchen.py` (wrapper `halving_search_kitchen.sh`).

Strategi ini memulai banyak kandidat dengan resource kecil, lalu hanya
kandidat terbaik yang dilanjutkan ke resource lebih besar.

### Mengapa Halving Random Search?

- Evaluasi kandidat mahal (train + infer), jadi perlu eliminasi dini.
- Kombinasi HP diuji acak di awal, lalu diseleksi bertahap berdasarkan skor.
- Resource yang ditingkatkan di sini adalah `episodes` inferensi.

### Cara kerja di repo ini

1. Sampling kandidat HP dari ruang diskrit.
2. Jalankan train + infer per seed.
3. Skor kandidat = mean `test_mean_score` lintas seed.
4. Hanya kandidat terbaik yang lanjut ke iterasi berikut dengan `episodes` lebih besar.

### Menjalankan tuning

Prasyarat: `pip install scikit-learn`.

```bash
cd FlowPolicy
bash scripts/halving_search_kitchen.sh 0 --dry-run
bash scripts/halving_search_kitchen.sh 0 --n-candidates 16 --min-episodes 10 --max-episodes 50
bash scripts/halving_search_kitchen.sh 0 --n-candidates 24 --factor 3 --min-episodes 8 --max-episodes 60
bash scripts/halving_search_kitchen.sh 0 --preprocess --n-candidates 16 --min-episodes 10 --max-episodes 50
```

Argumen penting:

- `--n-candidates`: jumlah kandidat awal.
- `--factor`: rasio eliminasi tiap iterasi halving.
- `--min-episodes`: resource awal (episodes infer).
- `--max-episodes`: resource maksimum di iterasi akhir.
- `--seeds`: seed evaluasi (default `0 42 101`).
- `--preprocess`: aktifkan preprocessing sliding-window.

Keluaran di `<out-root>/` (default `data/outputs/halving_search/`):

- `trials.csv` — log nilai tiap trial yang dieksekusi.
- `best_trial.json` — ringkasan parameter terbaik dari Halving search.
- `progress.log` — log progres ringkas.
- `trial_<nnnnn>_seed<s>/` — folder run train/infer per trial×seed.

### Plot riwayat tuning

```bash
cd FlowPolicy
bash scripts/halving_plot_kitchen.sh
# atau:
python scripts/halving_plot_kitchen.py --out-root data/outputs/halving_search
```

Output default:

- `plots/halving_search_history.png`

### Training final dari hasil tuning terbaik

Setelah `best_trial.json` tersedia, jalankan training final:

```bash
cd FlowPolicy
bash scripts/train_best_from_halving.sh 0 --seed 0
```

Contoh dengan path `best_trial.json` custom:

```bash
cd FlowPolicy
bash scripts/train_best_from_halving.sh 0 \
  --best-json data/outputs/halving_search/best_trial.json \
  --seed 42 \
  --run-dir data/outputs/kitchen_complete-flowpolicy_halving_best_seed42
```

## Suite eksperimen 12 model (3 seed x 4 skenario)

Skrip orkestrasi: `scripts/experiment_3seed_4arms.py`
(wrapper `scripts/experiment_3seed_4arms.sh`).

Untuk setiap seed (`0, 42, 101`) dijalankan 4 skenario:

1. `baseline`: tanpa tuning, tanpa preprocessing
2. `baseline_pre`: tanpa tuning, dengan preprocessing
3. `tuned`: tuning Halving tanpa preprocessing, lalu train final best params
4. `tuned_pre`: tuning Halving dengan preprocessing, lalu train final best params

Setelah 12 model selesai:

- pilih pemenang per-seed berdasarkan `test_mean_score`
- bandingkan 3 pemenang seed untuk juara global
- jalankan inferensi final juara global (SR + latensi; video opsional)

### Jalankan

```bash
cd FlowPolicy
bash scripts/experiment_3seed_4arms.sh 0 --dry-run
bash scripts/experiment_3seed_4arms.sh 0
```

Contoh dengan parameter tuning custom:

```bash
cd FlowPolicy
bash scripts/experiment_3seed_4arms.sh 0 \
  --seeds 0 42 101 \
  --halving-n-candidates 16 \
  --halving-factor 2 \
  --halving-min-episodes 10 \
  --halving-max-episodes 50 \
  --eval-episodes 50 \
  --hero-episodes 20
```

Aktifkan video pada inferensi final juara global:

```bash
bash scripts/experiment_3seed_4arms.sh 0 --hero-video
```

### Output utama

Default root output: `data/outputs/exp_3seed_4arms/`

- `seed0/`, `seed42/`, `seed101/` berisi 4 arm per-seed
- `all_models_eval.csv` ringkasan 12 model (SR + latensi)
- `seed_winners.json` pemenang per-seed
- `global_winner.json` juara global
- `global_winner_inference/metrics.json` metrik inferensi final juara global
- `final_summary.json` ringkasan akhir

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
