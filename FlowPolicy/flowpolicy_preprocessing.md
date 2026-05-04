# Preprocessing Dataset Franka Kitchen
> Panduan lengkap tahap preprocessing dataset **Kitchen-Complete-v2** untuk pelatihan **State-Based FlowPolicy** pada lingkungan Franka Kitchen

---

## Daftar Isi
1. [Gambaran Umum](#1-gambaran-umum)
2. [Struktur Dataset Kitchen-Complete-v2](#2-struktur-dataset-kitchen-complete-v2)
3. [Alur Preprocessing](#3-alur-preprocessing)
4. [Tahap 1 — Ekstraksi Vektor Observasi dan Aksi](#4-tahap-1--ekstraksi-vektor-observasi-dan-aksi)
5. [Tahap 2 — Sliding Window](#5-tahap-2--sliding-window)
6. [Tahap 3 — Split Dataset](#6-tahap-3--split-dataset)
7. [Tahap 4 — Augmentasi Data](#7-tahap-4--augmentasi-data)
8. [Ringkasan Parameter Preprocessing](#8-ringkasan-parameter-preprocessing)
9. [Catatan Teknis Tambahan](#9-catatan-teknis-tambahan)

---

## 1. Gambaran Umum

Preprocessing merupakan serangkaian prosedur pengolahan data yang mengubah dataset demonstrasi mentah menjadi format yang sesuai dengan arsitektur model. Dalam konteks imitation learning, kualitas data demonstrasi secara langsung menentukan kualitas kebijakan yang dipelajari.

### Tujuan Preprocessing pada Penelitian Ini
- Mengubah data demonstrasi raw menjadi sekuens temporal yang dapat dipelajari model
- Memaksimalkan jumlah sampel dari data yang terbatas (hanya 19 episode)
- Memitigasi risiko **overfitting** akibat sedikitnya data demonstrasi
- Memastikan tidak ada **kebocoran informasi** antara data train, validasi, dan test

### Sumber Data
- **Library:** Minari
- **Dataset:** `Kitchen-Complete-v2`
- **Total:** 19 episode, 4.209 timestep
- **Frekuensi kontrol:** 10 Hz (1 detik = 10 timestep)
- **Panjang per episode:** 280 timestep (bervariasi per demonstrasi)

---

## 2. Struktur Dataset Kitchen-Complete-v2

### 2.1 Vektor Observasi (dimensi: 59)

| Komponen                        | Dimensi       | Satuan        | Deskripsi                                              |
|---------------------------------|---------------|---------------|--------------------------------------------------------|
| Posisi sendi lengan             | 7 (7 DoF)     | Radian        | Sudut posisi tujuh sendi lengan robot                  |
| Posisi sendi gripper            | 2 (2 DoF)     | Meter         | Jarak pembukaan dua jari gripper                       |
| Kecepatan sendi lengan          | 7 (7 DoF)     | Radian/detik  | Kecepatan angular tujuh sendi lengan                   |
| Kecepatan sendi gripper         | 2 (2 DoF)     | Meter/detik   | Kecepatan linier jari gripper                          |
| Konfigurasi microwave           | 1             | Radian        | Sudut buka pintu microwave                             |
| Konfigurasi knob burner         | 4             | Radian        | Sudut rotasi knob kompor                               |
| Kecepatan knob burner           | 4             | Radian/detik  | Kecepatan perputaran rotasi knob kompor                |
| Konfigurasi pintu burner kanan  | 4             | Meter         | Sudut buka pintu burner                                |
| Kecepatan pintu burner kanan    | 4             | Meter/detik   | Kecepatan sudut buka pintu burner                      |
| Konfigurasi knob light switch   | 1             | Radian        | Sudut rotasi knob lampu                                |
| Kecepatan knob light switch     | 1             | Radian/detik  | Kecepatan sudut rotasi knob lampu                      |
| Konfigurasi light joint         | 1             | Radian        | Penyalaan lampu kompor melalui putaran knob            |
| Kecepatan light joint           | 1             | Radian/detik  | Kecepatan penyalaan lampu kompor                       |
| Konfigurasi teko (kettle)       | 7             | Meter/radian  | Posisi 3D + orientasi 4D (quaternion) teko             |
| Konfigurasi slide cabinet       | 1             | Meter         | Posisi geser pintu cabinet                             |
| Kecepatan slide cabinet         | 1             | Meter/detik   | Kecepatan linear membuka pintu cabinet                 |
| Konfigurasi hinge cabinet       | 2             | Radian        | Sudut bukaan dua engsel cabinet                        |
| Kecepatan hinge cabinet         | 2             | Radian/detik  | Kecepatan sudut buka dua engsel cabinet                |
| Status gripper                  | 1             | Biner         | Kondisi terbuka (1) atau tertutup (0)                  |
| **Total**                       | **59**        | —             | —                                                      |

> **Catatan:** Pada konfigurasi model, dimensi observasi yang digunakan adalah **70** sesuai `shape_meta.obs.agent_pos.shape: [70]` di `kitchen_complete.yaml`. Pastikan sinkronisasi dimensi antara dataset dan konfigurasi model.

### 2.2 Vektor Aksi (dimensi: 9)

| Komponen                    | Dimensi    | Satuan        | Deskripsi                                   |
|-----------------------------|------------|---------------|---------------------------------------------|
| Kecepatan target sendi lengan | 7 (7 DoF) | Radian/detik  | Perintah kecepatan angular per sendi lengan |
| Kecepatan target sendi gripper | 2 (2 DoF) | Meter/detik  | Perintah kecepatan linear jari gripper      |
| **Total**                   | **9**      | —             | —                                           |

> Vektor aksi merepresentasikan **velocity control** — bukan posisi target, melainkan kecepatan target pada setiap sendi robot pada timestep tersebut.

---

## 3. Alur Preprocessing

```
┌─────────────────────────────────┐
│  Dataset Kitchen-Complete-v2    │
│  (19 episode, 4.209 timestep)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  TAHAP 1: Ekstraksi             │
│  Ambil vektor observasi (59)    │
│  dan aksi (9) per timestep      │
│  per episode                    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  TAHAP 2: Sliding Window        │
│  Bentuk sekuens temporal        │
│  dengan window length T         │
│  dan stride = 1                 │
│  → menghasilkan L-T+1 segmen    │
│    per episode                  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  TAHAP 3: Split Dataset         │
│  15 ep → Data Latih             │
│   3 ep → Data Validasi          │
│   1 ep → Data Pengujian         │
│  (split pada level episode)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  TAHAP 4: Augmentasi            │
│  Tambahkan Gaussian noise       │
│  pada vektor OBSERVASI          │
│  (hanya data latih)             │
│  obs_noise_std = 0.01           │
│  action_noise_std = 0.0         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Dataset Siap untuk Pelatihan   │
└─────────────────────────────────┘
```

---

## 4. Tahap 1 — Ekstraksi Vektor Observasi dan Aksi

### Prosedur
Dari setiap episode, ekstraksi dilakukan secara terpisah untuk mempertahankan kontinuitas trajektori per episode.

```python
# Pseudocode ekstraksi data
for episode in dataset:                     # iterasi 19 episode
    observations = []
    actions = []
    for timestep in episode:
        obs = timestep.observation          # vektor ℝ⁵⁹
        act = timestep.action               # vektor ℝ⁹
        observations.append(obs)
        actions.append(act)
    episode_data[episode_id] = {
        'obs': observations,                # shape: [L, 59]
        'act': actions                      # shape: [L, 9]
    }
```

### Komponen Observasi yang Diekstraksi
- **Kinematika robot:** posisi dan kecepatan sendi lengan + gripper
- **Konfigurasi objek:** sudut/posisi microwave, knob kompor, teko, slide cabinet, light switch
- **Status gripper:** kondisi terbuka atau tertutup

### Komponen Aksi yang Diekstraksi
- **7 kecepatan sendi lengan** (radian/detik) — velocity control 7-DoF
- **2 kecepatan sendi gripper** (meter/detik) — kecepatan jari gripper

---

## 5. Tahap 2 — Sliding Window

### Tujuan
Memaksimalkan jumlah sampel pelatihan dari dataset terbatas dengan cara membentuk sekuens temporal yang saling tumpang tindih.

### Parameter
| Parameter         | Deskripsi                                                  | Nilai            |
|-------------------|------------------------------------------------------------|------------------|
| `window_length` T | Jumlah timestep berurutan per satu segmen sliding window   | `horizon` dari config |
| `stride`          | Seberapa jauh jendela digeser per iterasi                  | 1 (maksimal)     |
| `pad_before`      | Padding di awal episode                                    | `n_obs_steps - 1` |
| `pad_after`       | Padding di akhir episode                                   | `n_action_steps - 1` |

### Formula Jumlah Segmen per Episode
```
jumlah_segmen = L - T + 1
```
- `L` = panjang episode (timestep)
- `T` = window length (horizon)

### Contoh Default
```
n_obs_steps    = 2
n_action_steps = 4
horizon        = 4 * ((max(2+4-1, 4) + 3) // 4)
               = 4 * ((5 + 3) // 4)
               = 4 * 2 = 8

pad_before = n_obs_steps - 1    = 1
pad_after  = n_action_steps - 1 = 3

Untuk episode dengan L = 280 timestep dan T = 8:
jumlah_segmen = 280 - 8 + 1 = 273 segmen per episode
```

### Proses Sliding Window
```
Episode (L = 280 timestep):
[obs_1, obs_2, ..., obs_280]
[act_1, act_2, ..., act_280]

Jendela 1: timestep 1  → 8   → sampel ke-1
Jendela 2: timestep 2  → 9   → sampel ke-2
Jendela 3: timestep 3  → 10  → sampel ke-3
...
Jendela 273: timestep 273 → 280 → sampel ke-273
```

> **Perbedaan penting:** Segmen sliding window beroperasi pada **dimensi waktu episode** (potongan urutan timestep robot), sedangkan segmen internal consistency flow matching beroperasi pada **dimensi proses generatif** (langkah transformasi distribusi noise → aksi). Keduanya adalah konsep yang berbeda.

### Penanganan Awal Episode (Padding)
Pada timestep awal segmen dimana riwayat observasi belum tersedia sepanjang `observation_horizon`, mekanisme **padding** diterapkan dengan menduplikasi observasi awal.

```python
# Pseudocode padding di awal episode
if len(obs_history) < observation_horizon:
    obs_padded = [obs_current] * (observation_horizon - len(obs_history))
    obs_sequence = obs_padded + obs_history
else:
    obs_sequence = obs_history[-observation_horizon:]
```

---

## 6. Tahap 3 — Split Dataset

### Skema Pembagian

| Subset        | Jumlah Episode | Proporsi | Fungsi                                              |
|---------------|----------------|----------|-----------------------------------------------------|
| Data Latih    | 15 episode     | ~79%     | Memperbarui bobot jaringan via backpropagation      |
| Data Validasi | 3 episode      | ~16%     | Memantau overfitting selama training                |
| Data Test     | 1 episode      | ~5%      | Evaluasi akhir objektif (digunakan sekali saja)     |

### Mengapa Split pada Level Episode?
Split dilakukan **pada tingkat episode**, bukan pada tingkat segmen, untuk mencegah **kebocoran informasi temporal**:

```
✅ Split level episode (BENAR):
   Train:  [ep_1, ep_2, ..., ep_15]
   Val:    [ep_16, ep_17, ep_18]
   Test:   [ep_19]
   → Tidak ada segmen dari episode yang sama masuk ke dua subset berbeda

❌ Split level segmen (SALAH):
   → Segmen ep_5 timestep 1-8   → Train
   → Segmen ep_5 timestep 2-9   → Val
   → Tumpang tindih menyebabkan kontaminasi informasi!
```

### Deteksi Overfitting
Pantau kondisi berikut selama training:
- **Overfitting:** Training loss turun terus, validation loss stagnan atau naik
- **Underfitting:** Keduanya masih tinggi setelah banyak epoch
- **Optimal:** Keduanya rendah dan konvergen bersama

---

## 7. Tahap 4 — Augmentasi Data

### Aturan Augmentasi

| Data           | Augmentasi         | Alasan                                                    |
|----------------|--------------------|-----------------------------------------------------------|
| Data Latih     | ✅ Diterapkan       | Meningkatkan variasi, memitigasi overfitting              |
| Data Validasi  | ❌ Tidak diterapkan | Agar evaluasi mencerminkan kondisi data sesungguhnya      |
| Data Test      | ❌ Tidak diterapkan | Agar evaluasi akhir objektif                              |

### Metode: Gaussian Noise pada Vektor Observasi

```python
# Pseudocode augmentasi
obs_noise_std    = 0.01   # standar deviasi noise observasi
action_noise_std = 0.0    # aksi TIDAK diaugmentasi

for sample in train_data:
    noise = np.random.normal(
        mean = 0,
        std  = obs_noise_std,
        size = sample['obs'].shape    # ℝ⁵⁹
    )
    sample['obs'] = sample['obs'] + noise
    # sample['act'] TIDAK diubah
```

### Mengapa Distribusi Gaussian?
- Menghasilkan gangguan yang **terkonsentrasi di sekitar nol** → tidak menggeser nilai asli secara sistematis
- Semakin jarang pada nilai besar → gangguan ekstrem sangat langka
- Standar deviasi dikalibrasi proporsional terhadap rentang nilai tiap komponen

### Mengapa Aksi Tidak Diaugmentasi?
Vektor aksi merupakan **target keluaran** yang harus dipelajari secara akurat oleh model. Menambahkan noise pada aksi akan mengajarkan model untuk mereproduksi gerakan yang salah.

### Tujuan Augmentasi Ini
Mensimulasikan variasi kondisi observasi yang dapat terjadi saat pengujian akibat:
- Perbedaan kondisi awal lingkungan
- Variasi konfigurasi objek pada reset episode
- Ketidakpastian sensor dalam kondisi nyata

---

## 8. Ringkasan Parameter Preprocessing

| Parameter             | Nilai                  | Keterangan                                  |
|-----------------------|------------------------|---------------------------------------------|
| `n_episodes`          | 19                     | Total episode dataset                       |
| `total_timesteps`     | 4.209                  | Total timestep seluruh dataset              |
| `control_frequency`   | 10 Hz                  | 1 detik = 10 timestep                       |
| `max_episode_length`  | 280 timestep           | Batas maksimum per episode                  |
| `obs_dim`             | 59 (dataset) / 70 (model) | Dimensi vektor observasi                 |
| `action_dim`          | 9                      | Dimensi vektor aksi                         |
| `train_episodes`      | 15                     | Episode untuk data latih                    |
| `val_episodes`        | 3                      | Episode untuk data validasi                 |
| `test_episodes`       | 1                      | Episode untuk data test                     |
| `stride`              | 1                      | Langkah geser sliding window                |
| `pad_before`          | `n_obs_steps - 1`      | Padding awal episode (default: 1)           |
| `pad_after`           | `n_action_steps - 1`   | Padding akhir episode (default: 3)          |
| `obs_noise_std`       | 0.01                   | Standar deviasi Gaussian noise observasi    |
| `action_noise_std`    | 0.0                    | Tidak ada noise pada aksi                   |
| `split_level`         | Episode                | Bukan segmen, untuk cegah data leakage      |

---

## 9. Catatan Teknis Tambahan

### 9.1 Sinkronisasi Dimensi
Pastikan dimensi observasi konsisten antara dataset dan konfigurasi model:

```yaml
# Di kitchen_complete.yaml
shape_meta:
  obs:
    agent_pos:
      shape: [70]    # ← dimensi yang digunakan model
  action:
    shape: [9]

# Jika dataset memberikan obs berdimensi 59,
# perlu mapping/padding ke dimensi 70
```

### 9.2 Normalisasi
Pertimbangkan normalisasi nilai observasi dan aksi sebelum training untuk menstabilkan gradien:

```python
# Contoh min-max normalization
obs_min = obs_train.min(axis=0)
obs_max = obs_train.max(axis=0)
obs_normalized = (obs - obs_min) / (obs_max - obs_min + 1e-8)

# Simpan statistik untuk digunakan saat inferensi
```

### 9.3 Reproducibility
Kontrol semua sumber keacakan:

```python
import numpy as np
import torch
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### 9.4 Konsistensi Segmen
Ingat perbedaan dua jenis segmen dalam sistem ini:

| Jenis Segmen              | Domain          | Jumlah      | Fungsi                                      |
|---------------------------|-----------------|-------------|---------------------------------------------|
| Sliding Window Segment    | Waktu episode   | `L - T + 1` | Potongan urutan timestep sebagai input model |
| Consistency FM Segment (K) | Proses generatif | `K` (1–4)  | Langkah transformasi distribusi noise → aksi |

---

## Referensi

- Gupta, A., et al. (2019). *Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning.* arXiv.
- Chi, C., et al. (2025). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR.
- Bilal, M., et al. (2022). *Auto-Prep: Efficient and Automated Data Preprocessing Pipeline.* IEEE Access.
- Zhang, Q., et al. (2025). *FlowPolicy: Enabling Fast and Robust 3D Flow-Based Policy via Consistency Flow Matching for Robot Manipulation.* AAAI 2025.
