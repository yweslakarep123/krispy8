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
bash scripts/ofat_search_kitchen.sh 0                    # full sweep di GPU 0 (tanpa preprocessing)
bash scripts/ofat_search_kitchen.sh 0 --preprocess       # full sweep di GPU 0 (semua run pakai preprocessing)
bash scripts/ofat_search_kitchen.sh 0 --only-hp optimizer.lr   # hanya sweep lr
bash scripts/ofat_search_kitchen.sh 0 --max-minutes 600   # stop setelah 10 jam (Colab T4)
```

Jika `--preprocess` aktif, OFAT meneruskan override
`task.dataset.preprocess.*` ke **setiap** perintah training (`cfg_idx x seed`),
jadi semua run yang dieksekusi memakai preprocessing (bukan hanya run tertentu).

Keluaran:

- mode default (tanpa `--preprocess`) -> `data/outputs/ofat_search/`
- mode preprocessing (`--preprocess`) -> `data/outputs/ofat_search_preprocess/`
  (auto-suffix agar hasil dua mode tidak saling menimpa saat resume)

Isi folder output:

- `configs.json` — 32 konfigurasi unik
- `cfg_<ii>_seed<s>/` — run_dir per (konfigurasi, seed)
- `results.csv` — satu baris per run (96 bila lengkap)
- `summary.csv` — mean +/- std per konfigurasi di 3 seed, diurutkan dari
  `test_mean_score` tertinggi ke terendah

Resume otomatis via `metrics.json` di tiap run_dir.

### Isolasi model antar hyperparameter

Setiap kombinasi `(cfg_idx, seed)` mendapat folder kerja sendiri
`<out-root>/cfg_<ii>_seed<s>/` dengan `hydra.run.dir` yang unik per run.
Checkpoint hanya dibaca/ditulis dari folder itu, sehingga tidak ada
konfigurasi yang melanjutkan dari bobot konfigurasi lain. `training.resume`
hanya melanjutkan ckpt **dalam run_dir yang sama** (berguna bila proses
crash/timeout di tengah). Bila Anda ingin memaksa training ulang dari
awal untuk kombinasi tertentu, cukup hapus `<run_dir>/checkpoints/`
sebelum menjalankan ulang OFAT.

### Evaluasi & inferensi untuk semua 96 run OFAT

Setelah sweep training selesai, pakai dua skrip berikut untuk
re-evaluasi/rollout semua kombinasi tanpa training ulang:

```bash
# Evaluasi cepat (tanpa video, hanya SR + latency): hasil di eval_results.csv
bash scripts/ofat_eval_kitchen.sh 0

# Inferensi lengkap (dengan video mp4 per episode): hasil di infer_results.csv
bash scripts/ofat_infer_kitchen.sh 0

# Mode preprocessing (out-root otomatis ke ofat_search_preprocess)
bash scripts/ofat_infer_kitchen.sh 0 --preprocess

# Dry-run: cek mapping cfg -> ckpt yang akan dipakai
bash scripts/ofat_infer_kitchen.sh 0 --dry-run

# Batasi scope (sweep HP tertentu / seed tertentu / cfg_idx tertentu)
bash scripts/ofat_infer_kitchen.sh 0 --only-hp optimizer.lr
bash scripts/ofat_infer_kitchen.sh 0 --only-cfg 0 4 8 --seeds 0 42
```

Output per run (di `cfg_<ii>_seed<s>/`):

- `inference_ep50/videos/episode_000.mp4 ...` (hanya untuk `ofat_infer_kitchen.sh`)
- `inference_ep50[_novideo]/metrics.json` berisi antara lain:
  - `test_mean_score`: success rate rata-rata per episode
  - `mean_time`: latensi rata-rata per panggilan `predict_action` (detik)
  - `mean_n_completed_tasks`: jumlah sub-task yang terselesaikan
- `infer_stdout.log` atau `eval_stdout.log`

Agregasi ke CSV di `<out-root>/`:

- `infer_results.csv` (dari `ofat_infer_kitchen.sh`)
- `eval_results.csv`  (dari `ofat_eval_kitchen.sh`)

Resume otomatis: run yang `metrics.json`-nya sudah ada tidak akan
dijalankan ulang kecuali Anda pakai `--force`.

### Plot perbandingan (SR & latency vs HP)

```bash
# Pakai hasil ofat_infer_kitchen.sh (default)
bash scripts/ofat_plot_kitchen.sh

# Mode preprocessing
bash scripts/ofat_plot_kitchen.sh --preprocess

# Pakai eval_results.csv (hasil ofat_eval_kitchen.sh)
bash scripts/ofat_plot_kitchen.sh --metric-source eval

# Pakai results.csv dari sweep training (tidak butuh infer terpisah)
bash scripts/ofat_plot_kitchen.sh --metric-source search

# Atau CSV kustom
bash scripts/ofat_plot_kitchen.sh --csv /path/ke/custom.csv
```

Keluaran di `<out-root>/plots/`:

- `ofat_sr_vs_hp.png` — 8 subplot: success rate (mean±std) vs nilai HP.
- `ofat_latency_vs_hp.png` — 8 subplot: latency inferensi (ms) vs nilai HP.
- `ofat_sr_and_latency.png` — 8 subplot gabungan (dual y-axis).
- `ofat_summary.csv` — rekap `mean±std` SR & latency per `(swept_hp, value)`.

Prasyarat: `pip install matplotlib`. Plot dibuat dengan backend `Agg`
sehingga aman dijalankan headless (vast.ai / SSH tanpa display).

## Grid Search (subset HP, full cartesian)

OFAT hanya mengubah satu HP sekaligus — tidak menangkap interaksi antar-HP.
Untuk eksplorasi kombinasi, tersedia pipeline grid search:

- `FlowPolicy/scripts/gridsearch_kitchen.sh` + `gridsearch_kitchen.py`
- `FlowPolicy/scripts/gridsearch_infer_kitchen.sh` + `gridsearch_eval_kitchen.sh`
- `FlowPolicy/scripts/gridsearch_plot_kitchen.sh`

Grid full untuk 20 HP × 4 nilai tidak realistis (triliunan run). Jadi grid
search dijalankan pada **subset** HP yang dipilih; semua HP lain dikunci ke
baseline `flowpolicy.yaml`.

### 11 HP model baru (di luar 8 HP lama)

Semua HP model ini bisa di-subset lewat `--override-hp`. 4 nilai per HP
(2 untuk boolean) dipilih supaya reasonable dan saling kompatibel
(`n_groups` sengaja dibatasi agar selalu membagi semua channel di setiap
`down_dims`).

| HP | Kategori | 4 (atau 2) nilai default | Argumen |
|---|---|---|---|
| `policy.down_dims` | skala UNet | `[128,256,512]`, `[192,384,768]`, `[256,512,1024]`, `[384,768,1536]` | Kapasitas model — trade-off SR vs latency/memori. |
| `policy.diffusion_step_embed_dim` | cond waktu | `64, 128, 256, 512` | Ekspresivitas embedding timestep CFM. |
| `policy.kernel_size` | temporal | `3, 5, 7, 9` | Receptive field Conv1D di UNet. |
| `policy.n_groups` | GroupNorm | `1, 2, 4, 8` | Normalisasi; `8` aman universal (habis bagi semua `down_dims`). |
| `policy.encoder_output_dim` | encoder obs | `32, 64, 128, 256` | Dim fitur observasi yang masuk ke global cond. |
| `policy.state_mlp_size` | encoder obs | `[128,128]`, `[256,256]`, `[512,512]`, `[256,256,256]` | Kapasitas & kedalaman MLP encoder. |
| `policy.action_clip` | stabilitas | `0.5, 1.0, 2.0, 5.0` | Clamp output action — stabilitas inferensi. |
| `policy.encoder_use_layernorm` | stabilitas (bool) | `true, false` | LayerNorm di encoder obs. |
| `policy.use_down_condition` | conditioning (bool) | `true, false` | Pakai cond di down blocks. |
| `policy.use_mid_condition` | conditioning (bool) | `true, false` | Pakai cond di mid blocks. |
| `policy.use_up_condition` | conditioning (bool) | `true, false` | Pakai cond di up blocks. |

`policy.condition_type` **tidak** di-sweep (dikunci `film` sesuai
`flowpolicy.yaml`).

### Subset default (preset `default`)

7 HP, 1296 run (= 432 kombinasi × 3 seed):

| HP | nilai yang disweep |
|---|---|
| `optimizer.lr` | `1e-3, 1e-4, 1e-5` |
| `dataloader.batch_size` | `64, 128, 256` |
| `policy.Conditional_ConsistencyFM.num_segments` | `2, 3` |
| `policy.down_dims` | `[128,256,512]`, `[256,512,1024]`, `[384,768,1536]` |
| `policy.diffusion_step_embed_dim` | `128, 256` |
| `policy.kernel_size` | `3, 5` |
| `policy.use_mid_condition` | `true, false` |

Alternatif preset: `smoke` (4 HP, 16 kombinasi, 48 run) untuk verifikasi pipeline.

### Isolasi per run (penting)

Setiap `(cfg_idx, seed)` membuat `run_dir` baru `<out-root>/cfg_<4digit>_seed<s>/`
dengan `hydra.run.dir` unik dan **`training.resume=False` dipaksa** di
override. Ini memastikan setiap konfigurasi melatih model **dari nol**,
tidak melanjutkan bobot konfigurasi lain. Auto-resume by `metrics.json`
hanya berlaku dalam run_dir yang sama (recovery crash/timeout).

Kombinasi HP invalid (mis. `n_groups` tidak membagi `down_dims`) otomatis
dilewati dengan status `skipped_invalid`.

### Menjalankan training grid

```bash
cd FlowPolicy

# 1) dry-run: inspeksi kombinasi yang akan dijalankan
bash scripts/gridsearch_kitchen.sh 0 --dry-run

# 2) full sweep (preset default) — ~1296 run di GPU 0
bash scripts/gridsearch_kitchen.sh 0

# 3) mode preprocessing (out-root otomatis gridsearch_preprocess)
bash scripts/gridsearch_kitchen.sh 0 --preprocess

# 4) batas waktu (mis. 11 jam, Colab Free T4)
bash scripts/gridsearch_kitchen.sh 0 --max-minutes 660

# 5) preset kecil untuk verifikasi pipeline
bash scripts/gridsearch_kitchen.sh 0 --subset smoke

# 6) override subset HP (separator nilai ':' ; list pakai '[]')
bash scripts/gridsearch_kitchen.sh 0 \
  --override-hp optimizer.lr=1e-3:5e-4:1e-4 \
  --override-hp policy.down_dims=[128,256,512]:[256,512,1024] \
  --override-hp policy.use_mid_condition=true:false
```

Output di `<out-root>/` (default `data/outputs/gridsearch/`):

- `configs.json` — seluruh kombinasi + HP penuhnya
- `cfg_<4digit>_seed<s>/` — run_dir per (kombinasi, seed)
- `results.csv` — per run (hasil inferensi mini 50 ep yang dijalankan otomatis)
- `summary.csv` — mean±std per kombinasi (ter-urut `test_mean_score_mean`)
- `progress.log` — progres baris per run

### Evaluasi & inferensi semua run grid

Sama pola dengan OFAT, tapi dibaca dari `configs.json` grid (bukan mapping
OFAT hardcoded).

```bash
# Evaluasi cepat (tanpa video), hasil di eval_results.csv
bash scripts/gridsearch_eval_kitchen.sh 0

# Inferensi lengkap (video + metrik), hasil di infer_results.csv
bash scripts/gridsearch_infer_kitchen.sh 0

# Mode preprocessing
bash scripts/gridsearch_infer_kitchen.sh 0 --preprocess

# Subset run saja
bash scripts/gridsearch_infer_kitchen.sh 0 --only-cfg 0 1 2 --seeds 0 42
```

### Plot perbandingan (grid search)

```bash
bash scripts/gridsearch_plot_kitchen.sh                 # infer_results.csv (default)
bash scripts/gridsearch_plot_kitchen.sh --metric-source eval
bash scripts/gridsearch_plot_kitchen.sh --metric-source search
bash scripts/gridsearch_plot_kitchen.sh --preprocess
```

Keluaran di `<out-root>/plots/`:

- `grid_sr_marginal.png` — SR (mean±std) per HP, di-marginalize atas HP
  lain. Menjawab: "rata-rata SR kalau HP X = v apa?".
- `grid_latency_marginal.png` — idem untuk latency (ms).
- `grid_pair_heatmap.png` — heatmap SR untuk 2 HP paling berdampak
  (dipilih otomatis berdasar varians marginal SR).
- `grid_sr_vs_latency.png` — scatter Pareto: satu titik = satu kombinasi.
- `grid_summary.csv` — rekap mean±std per kombinasi, ter-urut dari SR
  tertinggi lalu latency terendah.

Prasyarat: `pip install matplotlib`.

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

**Cell 4a - (opsional) cek 32 konfigurasi x 3 seed = 96 run via dry-run.**
Jangan gunakan `| head`; daftarnya pendek, tampilkan langsung. `| head`
akan memicu SIGPIPE dan `set -o pipefail` menganggapnya gagal.

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
bash scripts/ofat_search_kitchen.sh 0 --dry-run
```

Untuk dry-run dengan preprocessing:

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
bash scripts/ofat_search_kitchen.sh 0 --preprocess --dry-run
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

Jika ingin mode preprocessing di Colab:

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
export MUJOCO_GL=egl
pkill -f ofat_search_kitchen.py 2>/dev/null || true
nohup bash scripts/ofat_search_kitchen.sh 0 --preprocess --max-minutes 660 \
    > /tmp/ofat_master.log 2>&1 &
sleep 2
echo "Sweep PID: $(pgrep -f ofat_search_kitchen.py || echo '(tidak jalan)')"
echo "Master log: /tmp/ofat_master.log"
echo "Progress log : data/outputs/ofat_search_preprocess/progress.log"
```

**Cell 5 - pantau progress satu baris per run (paling ringkas):**

```bash
%%bash
cd /content/flowpolicy_kitchen/FlowPolicy
# Tampilkan hanya baris [PROGRESS n/96] dari progress.log, live
# (ganti folder ke ofat_search_preprocess jika mode preprocessing aktif)
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
