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

## Bayesian optimization (hyperparameter tuning)

Sweep OFAT, grid search, dan random search diganti **satu** pipeline:
`scripts/bayes_opt_kitchen.py` (wrapper `bayes_opt_kitchen.sh`). Tujuannya sama
dengan ide Bayesian optimization pada umumnya: evaluasi satu konfigurasi
(training + inferensi) **mahal**, jadi kita ingin **sedikit iterasi** namun
**mengarah** ke kombinasi HP yang bagus, dengan **belajar dari hasil trial
sebelumnya** (bukan mencoba semua kombinasi seperti grid, juga bukan murni
acak seperti random search skala besar).

### Mengapa BO (bukan grid / random penuh)?

- **Efisien saat evaluasi mahal** — deep learning + simulasi kitchen memakan
  waktu; BO menghemat trial dibanding brute-force.
- **Pembelajaran berkelanjutan** — setiap trial memperbarui model internal
  sampler tentang wilayah HP yang menjanjikan.
- **Eksplorasi vs eksploitasi** — fase awal (`--n-startup-trials`) mengeksplorasi;
  setelahnya sampler mengarahkan ke area yang terbukti baik sambil tetap
  menjelajah secukupnya.

### Komponen “surrogate” dan “acquisition” di repo ini

Di literatur klasik, BO sering digambarkan sebagai **Gaussian Process**
(surrogate) + **Expected Improvement** (acquisition). Itu ideal untuk ruang
**kontinu** berdimensi rendah. Di FlowPolicy, ruang HP kita **dominan diskrit /
kategorikal** (19 dimensi, banyak pilihan per kartu). Implementasi yang umum
dan stabil di HPO deep learning adalah **TPE (Tree-structured Parzen
Estimator)** lewat Optuna:

| Konsep teksbuku | Setara praktis di skrip ini |
|-----------------|-----------------------------|
| Surrogate model | Dua kepadatan Parzen: ``p(x \| y baik)`` vs ``p(x \| y buruk)`` dari riwayat trial (TPE). Opsi default **multivariate** memodelkan ketergantungan antar HP bersamaan. |
| Acquisition | Pemilihan kandidat berikutnya memaksimalkan rasio / skor dari kedua kepadatan di atas (bukan EI eksplisit, tetapi peran serupa: “di mana mencoba berikutnya?”). |
| Inisialisasi acak | `--n-startup-trials` trial pertama sebelum TPE “mengambil alih”. |

Ini **tetap** Bayesian optimization dalam arti *sequential model-based
optimization*; hanya **bukan** GP+EI literal. Jika Anda butuh GP+EI khusus,
biasanya memerlukan ruang kontinu + dependensi seperti BoTorch (di luar cakupan
skrip default ini).

### Alur satu trial

1. Sampler Optuna mengusulkan vektor HP (dari model internal setelah trial
   cukup banyak).
2. Konfigurasi tidak valid → *prune* (tanpa train).
3. Train + infer mini-eval per seed; objektif = **mean** `test_mean_score`.
4. Hasil masuk ke `study.db` dan memperbarui TPE untuk trial berikutnya.

Detail teknis:

- Kombinasi tidak valid (`n_groups` vs `policy.down_dims`) → *prune*.
- `--no-mem-cap` mematikan penurunan otomatis batch untuk UNet lebar.
- Resume: `metrics.json` yang valid untuk `trial_<nnnnn>_seed<s>/` dilewati.
- Storage: `sqlite:///<out-root>/study.db` (lanjutkan dengan menjalankan ulang
  perintah yang sama).
- `--no-tpe-multivariate` memaksa TPE univariat (lebih cepat, kurang memodelkan
  interaksi antar HP).

Prasyarat: `pip install optuna` (ada di `scripts/colab_install.sh`).

```bash
cd FlowPolicy
bash scripts/bayes_opt_kitchen.sh 0 --dry-run
bash scripts/bayes_opt_kitchen.sh 0 --n-trials 30
bash scripts/bayes_opt_kitchen.sh 0 --n-trials 50 --max-minutes 660
bash scripts/bayes_opt_kitchen.sh 0 --preprocess --n-trials 20
```

Keluaran di `<out-root>/` (default `data/outputs/bayes_opt/`; dengan
`--preprocess` → `data/outputs/bayes_opt_preprocess/`):

- `study.db` — basis data Optuna (riwayat trial). Bila Anda mengubah daftar
  nilai HP di skrip dan studi lama error *dynamic value space*, hapus
  `study.db` (dan opsional `trials.csv`) atau gunakan `--study-name` baru.
- `trials.csv` — satu baris per trial sukses: `trial_number`, `value` (mean SR),
  `config_json` (HP lengkap).
- `best_trial.json` — ringkasan trial terbaik setelah run selesai.
- `trial_<nnnnn>_seed<s>/` — `hydra.run.dir` per trial×seed (`train_stdout.log`,
  `infer_stdout.log`, checkpoint, `metrics.json`).

### Plot riwayat optimisasi

```bash
cd FlowPolicy
bash scripts/bayes_opt_plot_kitchen.sh
# atau: python scripts/bayes_opt_plot_kitchen.py --out-root data/outputs/bayes_opt_preprocess
```

Menghasilkan `plots/bayes_optimization_history.png` (nilai trial + *best so far*).
Prasyarat: `pip install matplotlib`.

### Suite eksperimen (3 seed × preprocess on/off = 6× BO)

Untuk desain: **tiga seed** (`0, 42, 101`) dan tiap seed punya **dua mode**
(dataset tanpa preprocess dan dengan preprocess) = **enam** jalur BO terpisah.

Skrip: `scripts/bo_franka_kitchen_suite.py` (wrapper `bo_franka_kitchen_suite.sh`).

```bash
cd FlowPolicy
bash scripts/bo_franka_kitchen_suite.sh 0 --dry-run
bash scripts/bo_franka_kitchen_suite.sh 0 --suite-seeds 0 42 101 --n-trials 1 --bo-episodes 50
bash scripts/bo_franka_kitchen_suite.sh 0 --suite-seeds 0 42 101 --n-trials 1 --parallel-bo --gpu-pool 0,1,2,3,4,5
```

Argumen berguna:

- `--suite-root` — induk keluaran (default `data/outputs/bo_franka_suite/`).
- `--suite-seeds` — daftar seed eksperimen (default `0 42 101`).
- `--n-trials` — jumlah trial BO per arm (default `1`).
- `--bo-episodes` — episode infer mini di dalam BO (diteruskan ke `bayes_opt_kitchen.py`).
- `--eval-episodes` — episode infer **tanpa video** setelah BO untuk SR + latensi tiap arm.
- `--hero-episodes` — episode infer **dengan video** hanya untuk model global terbaik.
- `--parallel-bo` — jalankan 6 arm BO secara paralel.
- `--gpu-pool` — daftar GPU untuk BO paralel (contoh `0,1,2,3,4,5`), alokasi round-robin per arm.
- `--skip-bo` — hanya agregasi + eval + hero (enam subfolder BO sudah selesai).

Keluaran utama di bawah `--suite-root`:

| Berkas / folder | Isi |
|-----------------|-----|
| `seed<S>_nopre/`, `seed<S>_pre/` | Enam arm BO total (3 seed × pre/non-pre) |
| `suite_arms.json` | Ringkasan best trial + path checkpoint tiap arm |
| `suite_eval.csv` | SR overall + metrik SR lain (mis. `SR_*`) + latensi (ms) |
| `suite_winner.json` | Pemenang overall (berdasarkan `test_mean_score`) |
| `suite_winner_by_metric.json` | Pemenang per metrik SR (mis. `SR_*` per-task) |
| `hero_best/` | Inferensi pemenang: `videos/*.mp4`, `metrics.json` |

#### Quick run di Vast.ai

Gunakan ini bila Anda menjalankan di instance Vast.ai (Ubuntu + GPU). Jalankan
dari root repo `FlowPolicy`.

```bash
cd /workspace/FlowPolicy
bash scripts/colab_install.sh
```

Lalu cek dulu command yang akan dieksekusi:

```bash
cd /workspace/FlowPolicy
bash scripts/bo_franka_kitchen_suite.sh 0 --dry-run
```

Run penuh (mode default sekuensial, 1 GPU), simpan log ke file:

```bash
cd /workspace/FlowPolicy
export MUJOCO_GL=egl
nohup bash scripts/bo_franka_kitchen_suite.sh 0 \
  --suite-seeds 0 42 101 \
  --n-trials 1 \
  --bo-episodes 50 \
  --eval-episodes 50 \
  --hero-episodes 20 \
  > /tmp/bo_suite.log 2>&1 &
echo "PID: $(pgrep -f bo_franka_kitchen_suite.py || echo '(tidak jalan)')"
echo "Log: /tmp/bo_suite.log"
```

Untuk Vast.ai multi-GPU (disarankan), jalankan 6 arm BO paralel:

```bash
cd /workspace/FlowPolicy
export MUJOCO_GL=egl
nohup bash scripts/bo_franka_kitchen_suite.sh 0 \
  --parallel-bo \
  --gpu-pool 0,1,2,3,4,5 \
  --suite-seeds 0 42 101 \
  --n-trials 1 \
  --bo-episodes 50 \
  --eval-episodes 50 \
  --hero-episodes 20 \
  > /tmp/bo_suite_parallel.log 2>&1 &
echo "Log: /tmp/bo_suite_parallel.log"
```

Monitoring:

```bash
tail -n 60 /tmp/bo_suite.log
```

Setelah run selesai, hasil utama:

- `data/outputs/bo_franka_suite/suite_eval.csv`
- `data/outputs/bo_franka_suite/suite_winner.json`
- `data/outputs/bo_franka_suite/hero_best/videos/`

Jika BO 6 arm sudah selesai tapi proses terhenti sebelum evaluasi/video:

```bash
cd /workspace/FlowPolicy
bash scripts/bo_franka_kitchen_suite.sh 0 --skip-bo --eval-episodes 50 --hero-episodes 20
```

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
!rm -rf /content/flowpolicy_kitchen/FlowPolicy/data/outputs/bayes_opt
!ln -s /content/drive/MyDrive/flowpolicy_kitchen_outputs /content/flowpolicy_kitchen/FlowPolicy/data/outputs/bayes_opt
```

**Cell 4a — dry-run BO (contoh kombinasi HP; menulis header `trials.csv`,
tanpa `study.db` sampai run penuh):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
bash scripts/bayes_opt_kitchen.sh 0 --dry-run
```

**Cell 4 — jalankan Bayesian optimization di background** (`--max-minutes`
opsional, sama seperti sebelumnya untuk OFAT).

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
export MUJOCO_GL=egl
pkill -f bayes_opt_kitchen.py 2>/dev/null || true
nohup bash scripts/bayes_opt_kitchen.sh 0 --n-trials 30 --max-minutes 660 \
    > /tmp/bo_master.log 2>&1 &
sleep 2
echo "BO PID: $(pgrep -f bayes_opt_kitchen.py || echo '(tidak jalan)')"
echo "Master log: /tmp/bo_master.log"
echo "Progress: data/outputs/bayes_opt/progress.log"
```

Mode preprocessing: ganti perintah menjadi
`bash scripts/bayes_opt_kitchen.sh 0 --preprocess --n-trials 20 ...` dan
folder progres ke `data/outputs/bayes_opt_preprocess/`.

**Cell 5 — pantau progress:**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
tail -n 40 data/outputs/bayes_opt/progress.log 2>/dev/null || true
echo "Masih jalan? $(pgrep -f bayes_opt_kitchen.py >/dev/null && echo YES || echo NO)"
```

**Cell 6 — tail master log:**

```bash
%%bash
tail -n 60 /tmp/bo_master.log
```

**Cell 7 — trial terbaru (train/infer):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy/data/outputs/bayes_opt
latest=$(ls -td trial_* 2>/dev/null | head -n 1)
echo "Terbaru: $latest"
tail -n 30 "$latest/train_stdout.log" 2>/dev/null || true
tail -n 20 "$latest/infer_stdout.log" 2>/dev/null || true
```

**Cell 8 — ringkasan (`trials.csv` + `best_trial.json`):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
tail -n 15 data/outputs/bayes_opt/trials.csv 2>/dev/null || true
cat data/outputs/bayes_opt/best_trial.json 2>/dev/null || echo "(belum selesai)"
```

**Cell 9 — hentikan BO:**

```bash
%%bash
pkill -f bayes_opt_kitchen.py && echo dihentikan || echo tidak ada proses
```

Setelah restart Colab, jalankan ulang cell 4: trial/seed yang sudah punya
`metrics.json` valid akan dilewati.

**Cell 10 (opsional) — inferensi manual dari checkpoint trial terbaik**
(sesuaikan path dari `best_trial.json` atau folder `trial_*`):

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
python infer_kitchen.py \
  --checkpoint data/outputs/bayes_opt/trial_00000_seed0/checkpoints/latest.ckpt \
  --episodes 50 --device cuda:0
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
