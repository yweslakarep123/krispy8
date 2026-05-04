# Hyperparameter Fine-Tuning FlowPolicy
> Panduan konfigurasi dan eksplorasi hyperparameter untuk implementasi **State-Based FlowPolicy** pada lingkungan **Franka Kitchen** (Long Horizon Manipulation Task)

---

## Daftar Isi
1. [Konfigurasi Default (Baseline)](#1-konfigurasi-default-baseline)
2. [Rentang Fine-Tuning](#2-rentang-fine-tuning)
3. [Deskripsi Setiap Hyperparameter](#3-deskripsi-setiap-hyperparameter)
4. [Strategi Eksplorasi: Random Search](#4-strategi-eksplorasi-random-search)
5. [Metrik Evaluasi](#5-metrik-evaluasi)
6. [Prosedur Eksperimen](#6-prosedur-eksperimen)
7. [Catatan Teknis dari Kode (Hydra Config)](#7-catatan-teknis-dari-kode-hydra-config)
8. [Menjalankan eksperimen di repositori ini](#8-menjalankan-eksperimen-di-repositori-ini)

---

## 1. Konfigurasi Default (Baseline)

Konfigurasi awal mengacu pada penelitian orisinal FlowPolicy (Q. Zhang et al., 2025) yang sebelumnya dioptimasi untuk **short horizon task** berbasis point cloud.

| Hyperparameter          | Nilai Default | Keterangan                          |
|-------------------------|---------------|-------------------------------------|
| `epoch`                 | 3000          | Dari `training.num_epochs`          |
| `learning_rate`         | 1e-4          | AdamW optimizer                     |
| `batch_size`            | 128           | Train & validation                  |
| `seed`                  | 3             | Random seed utama                   |
| `hidden_dim`            | 512           | Jumlah hidden dimension MLP         |
| `time_embedding_dim`    | 256           | Dimensi time embedding              |
| `num_segments` (K)      | 2             | Jumlah segmen Consistency FM        |
| `epsilon` (eps)         | 1e-2          | Dipakai di training & inferensi     |
| `delta_t`               | 1e-2          | Granularitas langkah waktu          |
| `action_horizon`        | 4             | Panjang sekuens aksi per inferensi  |
| `observation_horizon`   | 2             | Jumlah frame observasi historis     |

> **Catatan:** Nilai `num_segments` dan parameter Consistency FM lain (boundary, alpha) berasal dari `config/flowpolicy.yaml`. Pastikan override lewat CLI jika diperlukan.

---

## 2. Rentang Fine-Tuning

Rentang eksplorasi ditetapkan secara simetris di sekitar nilai default untuk memastikan landasan empiris yang valid.

| Hyperparameter          | Rentang Eksplorasi              | Tipe     |
|-------------------------|---------------------------------|----------|
| `epoch`                 | [500, 1000, 3000, 5000]         | Diskrit  |
| `learning_rate`         | [1e-3, 5e-4, 1e-4, 1e-5]       | Kontinu  |
| `batch_size`            | [64, 128, 256, 512]             | Diskrit  |
| `seed`                  | 3 (dikontrol, tidak di-tune)    | Fixed    |
| `hidden_dim`            | [128, 256, 512, 1024]           | Diskrit  |
| `time_embedding_dim`    | [128, 256, 512, 1024]           | Diskrit  |
| `num_segments` (K)      | [1, 2, 3, 4]                    | Diskrit  |
| `epsilon`               | [1e-4, 1e-3, 1e-2, 1]          | Kontinu  |
| `delta_t`               | [1e-4, 1e-3, 1e-2, 1]          | Kontinu  |
| `action_horizon`        | [2, 4, 6, 8]                    | Diskrit  |
| `observation_horizon`   | [4, 6, 8, 16]                   | Diskrit  |

---

## 3. Deskripsi Setiap Hyperparameter

### 3.1 Kelompok Transformasi Distribusi (Consistency FM)

#### `num_segments` (K)
- Menentukan berapa kali medan vektor dievaluasi selama transformasi distribusi dari noise menuju distribusi aksi target.
- **K besar** → lintasan lebih halus dan akurat, tetapi **inference latency meningkat linear**.
- **K kecil** → latensi rendah, tetapi risiko lintasan transformasi kurang presisi.
- Pada inferensi: total waktu ∝ K (setiap segmen = 1 forward pass jaringan).

```
Lat = Σ (t_akhir_forward_pass_i - t_awal_forward_pass_i), i = 1..K
```

#### `delta_t` (Δt)
- Granularitas interval waktu antara dua titik evaluasi dalam satu segmen.
- **Δt terlalu besar** → lompatan transformasi kasar, konsistensi antarsegmen sulit dipertahankan, training tidak konvergen.
- **Δt terlalu kecil** → gradien tidak signifikan, model butuh waktu sangat lama untuk belajar.
- Nilai yang direkomendasikan untuk eksplorasi awal: `1e-2`.

#### `epsilon` (ε)
- Batas waktu minimum pada sampling timestep `t` saat training dan inferensi.
- Dipakai di `compute_loss` maupun `predict_action`.
- **Pada inferensi:** hardcoded menggunakan nilai `self.eps` (sama dengan training).

#### `num_inference_step`
- Jumlah langkah inferensi pada `ConsistencyFM` (`sample_N`).
- Default: `1` (single-step inference).
- Meningkatkan nilai ini meningkatkan kualitas aksi namun menambah latensi.

---

### 3.2 Kelompok Optimasi Model

#### `learning_rate`
- Laju pembelajaran AdamW optimizer.
- Terlalu besar → training tidak stabil.
- Terlalu kecil → konvergensi lambat.
- Scheduler: **cosine** dengan warmup 500 steps (`num_cycles=0.5`).

#### `batch_size`
- Jumlah sampel per iterasi training.
- Batch kecil → update lebih sering, gradient noise tinggi.
- Batch besar → estimasi gradient lebih stabil, kebutuhan VRAM meningkat.
- Perhatikan batas VRAM: GTX 1080 (8 GB).

#### `epoch`
- Jumlah total epoch pelatihan.
- Dataset kecil (19 episode) → risiko overfitting pada epoch tinggi.
- Disarankan pantau validation loss untuk early stopping.

#### `hidden_dim`
- Ukuran layer tersembunyi pada Multi Layer Perceptron (MLP).
- MLP menerima input: `[t, a_t, s]` dengan total dimensi `1 + 9 + d = 10 + d`.
- **Terlalu kecil** → model tidak mampu mempelajari dinamika kompleks.
- **Terlalu besar** → risiko overfitting, terutama pada dataset kecil.

#### `time_embedding_dim`
- Dimensi representasi embedding untuk variabel waktu `t`.
- Mempengaruhi kemampuan model dalam membedakan tahap transformasi.

---

### 3.3 Kelompok Horizon

#### `action_horizon`
- Jumlah langkah aksi yang diprediksi model dalam satu inferensi.
- **Besar** → robot bergerak lebih "terencana", tetapi aksi awal mungkin sudah stale saat dieksekusi.
- **Kecil** → respons lebih reaktif terhadap lingkungan.

#### `observation_horizon`
- Jumlah frame observasi historis yang digunakan sebagai konteks.
- **Besar** → konteks temporal lebih kaya, tetapi meningkatkan dimensi input.
- **Kecil** → input lebih ringkas, risiko kehilangan konteks temporal penting.

> **Hubungan dengan `horizon` (panjang window trajectory):**
> ```
> horizon = 4 * ((max(n_obs_steps + n_action_steps - 1, 4) + 3) // 4)
> ```
> Contoh default: `max(2+4-1, 4) = 5` → `(5+3)//4 = 2` → `4*2 = 8`

---

## 4. Strategi Eksplorasi: Random Search

Penelitian ini menggunakan **Random Search** (Bergstra & Bengio, 2012) karena:
- Lebih efisien dari Grid Search untuk ruang hyperparameter berdimensi tinggi.
- Cocok ketika hanya sebagian kecil hyperparameter yang dominan memengaruhi performa.
- Tidak membuat asumsi struktural tertentu tentang bentuk ruang performa.

### Alur Random Search

```
1. Sampel acak satu kombinasi hyperparameter dari rentang di Tabel 2
2. Latih model FlowPolicy dari awal dengan konfigurasi tersebut
   └── Gunakan dataset Kitchen-Complete-v2 (15 ep train, 3 ep val, 1 ep test)
3. Jalankan inferensi pada lingkungan Franka Kitchen
   └── 50 episode × 3 random seed berbeda
4. Hitung metrik:
   └── success_rate, inference_latency, trade_off = success_rate / latency
5. Catat konfigurasi dan nilai trade_off
6. Ulangi hingga konfigurasi optimal ditemukan
```

### Pengendalian Seed
Tiga seed berbeda digunakan pada setiap konfigurasi untuk:
- Inisialisasi bobot awal neural network
- Pengacakan data batch saat training
- Sampling noise pada inferensi
- Reset state awal lingkungan simulasi

Hasil dilaporkan sebagai **rata-rata ± simpangan baku** dari 3 seed.

---

## 5. Metrik Evaluasi

### 5.1 Success Rate

#### Success Rate Total
```
success_rate = (N_success / N_total) × 100%
```

#### Success Rate Per Sub-Tugas (k ∈ {1,2,3,4})
```
success_rate_k = (N_success_k / N_total) × 100%
```

| k | Sub-Tugas yang Diselesaikan Secara Berurutan                          |
|---|-----------------------------------------------------------------------|
| 1 | Membuka pintu microwave                                               |
| 2 | Membuka microwave + memutar knob lampu kompor                         |
| 3 | Membuka microwave + knob + menaruh teko di kompor                     |
| 4 | Keempat sub-tugas lengkap (+ membuka slide cabinet)                   |

- Batas maksimum: **280 langkah per episode**
- Keberhasilan bersifat **biner** berdasarkan threshold konfigurasi objek

### 5.2 Inference Latency

```
Lat = Σ (t_akhir_forward_pass_i - t_awal_forward_pass_i), i = 1..K
```

- Latency berskala **linear** terhadap `K` (jumlah segmen)
- Gunakan **dummy pass** (GPU warm-up) sebelum pengukuran untuk menghilangkan bias CUDA initialization

### 5.3 Trade-Off Score

```
trade_off = success_rate / Lat
```

Nilai trade-off **lebih tinggi** = keseimbangan performa dan efisiensi komputasi **lebih baik**.

---

## 6. Prosedur Eksperimen

### 6.1 Preprocessing Dataset
- Dataset: `Kitchen-Complete-v2` (19 episode, 4.209 timestep total)
- Split: **15 train / 3 validasi / 1 test** (pada level episode)
- Sliding window dengan stride = 1
- Augmentasi: Gaussian noise pada vektor observasi (hanya data train)
  - `obs_noise_std: 0.01`
  - `action_noise_std: 0.0` (aksi tidak diaugmentasi)

### 6.2 Modifikasi Arsitektur (State-Based)
- PointNet++ encoder **dihapus**
- Diganti dengan **state encoder** (linear projection ℝ⁵⁹ → ℝᵈ)
- Input MLP: `[t ∈ ℝ¹, a_t ∈ ℝ⁹, s ∈ ℝᵈ]` → total `10 + d` dimensi
- Output MLP: vektor aksi ℝ⁹

### 6.3 Konfigurasi Inferensi (Hardcoded)
| Parameter       | Nilai        |
|-----------------|--------------|
| `noise_scale`   | 1.0          |
| `sigma_var`     | 0.0          |
| `ode_tol`       | 1e-5         |
| `ode_sampler`   | `rk45`       |

### 6.4 Perangkat Keras
| Komponen     | Spesifikasi             |
|--------------|-------------------------|
| GPU          | NVIDIA GTX 1080 Ti      |
| VRAM         | 8 GB                    |
| CPU          | Intel Core i7           |
| RAM          | 16 GB                   |
| OS           | Ubuntu 20.04 LTS        |

---

## 7. Catatan Teknis dari Kode (Hydra Config)

Berikut nilai aktual dari `flowpolicy.yaml` dan `kitchen_complete.yaml` yang perlu diperhatikan:

```yaml
# Training
training:
  num_epochs: 3000
  lr_scheduler: cosine
  lr_warmup_steps: 500
  ema_decay: 0.95  # ⚠️ TIDAK dipakai di kode, hanya ada di YAML

# Optimizer (AdamW)
optimizer:
  lr: 1.0e-4
  betas: [0.95, 0.999]
  eps: 1.0e-8
  weight_decay: 1.0e-6

# EMA (yang benar-benar dipakai)
ema:
  inv_gamma: 1.0
  power: 0.75
  min_value: 0.0
  max_value: 0.9999
  update_after_step: 0

# Consistency FM
policy:
  num_segments: 2
  boundary: 1
  delta: 1.0e-2
  alpha: 1.0e-5
  eps: 1.0e-2
  num_inference_step: 1

# Shape
shape_meta:
  obs:
    agent_pos:
      shape: [70]   # dimensi observasi per timestep
  action:
    shape: [9]      # dimensi aksi per timestep
```

> **Perhatian:** `training.ema_decay` hanya ada di config dan **tidak terhubung** ke `EMAModel`. EMA yang benar-benar dipakai menggunakan `inv_gamma`, `power`, `min_value`, `max_value`.

---

## 8. Menjalankan eksperimen di repositori ini

Implementasi mengikuti §4 (random search): skrip
`FlowPolicy/scripts/random_search_kitchen.py` mengambil sampel **N**
kombinasi dari ruang §2 (11 hyperparameter), melatih dari awal per
**(cfg_idx, seed)**, lalu menjalankan **`FlowPolicy/infer_kitchen.py`** sebanyak
**50 episode** pada `checkpoints/latest.ckpt`. Hasil per run ditulis ke
`results.csv`; agregasi mean ± std per `cfg_idx` ke **`summary.csv`**.

**Prasyarat:** environment terpasang (`install.md`), dataset Minari
`D4RL/kitchen/complete-v2` tersedia, GPU CUDA.

### Perintah (dari root repo `…/FlowPolicy/`)

```bash
cd FlowPolicy

# 1) Cek 30 konfigurasi yang akan dijalankan (tanpa training)
bash scripts/random_search_kitchen.sh 0 --dry-run

# 2) Jalankan penuh di GPU 0 (30 cfg × 3 seed × train + infer 50 ep)
#    Perkiraan waktu sangat lama; bisa dihentikan dan dijalankan ulang
#    (run yang sudah punya metrics.json valid akan dilewati).
bash scripts/random_search_kitchen.sh 0

# 3) Tulis ulang summary.csv dari hasil lama (urut trade-off §5.3)
python scripts/random_search_kitchen.py --gpu 0 \
  --out-root data/outputs/random_search --summary-only --sort-by trade_off
```

**Catatan:** split dataset **15 / 3 / 1** episode (§6.1) mengacu pada
`kitchen-complete-v2`; di kode saat ini pemisahan train/val memakai
`dataset.val_ratio` (default **0.02**), bukan rasio tepat 15/3/1. Untuk
reproduksi ketat §6.1 perlu override atau perubahan `KitchenDataset`
(issue terpisah).

**Keluaran:**

| Path | Isi |
|------|-----|
| `FlowPolicy/FlowPolicy/data/outputs/random_search/configs.json` | Daftar konfigurasi ter-sampel |
| `FlowPolicy/FlowPolicy/data/outputs/random_search/cfg_XX_seedY/` | Log train/infer, checkpoint, `inference_ep50/metrics.json` |
| `FlowPolicy/FlowPolicy/data/outputs/random_search/results.csv` | Satu baris per (cfg, seed) |
| `FlowPolicy/FlowPolicy/data/outputs/random_search/summary.csv` | Mean ± std per cfg_idx |

---

## Referensi

- Zhang, Q., et al. (2025). *FlowPolicy: Enabling Fast and Robust 3D Flow-Based Policy via Consistency Flow Matching for Robot Manipulation.* AAAI 2025.
- Yang, L., et al. (2024). *Consistency Flow Matching: Defining Straight Flows with Velocity Consistency.* arXiv.
- Bergstra, J., & Bengio, Y. (2012). *Random search for hyper-parameter optimization.* JMLR.
- Chi, C., et al. (2025). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR.
- Gupta, A., et al. (2019). *Relay Policy Learning.* arXiv.
