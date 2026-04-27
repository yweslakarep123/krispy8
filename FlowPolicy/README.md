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

Catatan: penggunaan GPU tidak selalu 100% setiap detik karena bisa dibatasi pipeline environment/data loading, tetapi konfigurasi ini memastikan eksekusi utama berjalan di CUDA dan di-tune untuk throughput.
