#!/usr/bin/env python3
"""Random search untuk FlowPolicy di FrankaKitchen-v1 (§4 dokumen
`flowpolicy_hyperparameter_finetuning.md`).

Melakukan:
1. Sampling N konfigurasi acak dari ruang hyperparameter §2 (11 knob:
   epoch, lr, batch, FM num_segments/eps/delta, n_action_steps, n_obs_steps,
   diffusion_step_embed_dim, lebar MLP state encoder, lebar kanal UNet).
2. Untuk setiap konfigurasi × seed, jalankan `train.py` (Hydra) lalu
   `infer_kitchen.py` (50 episode) terhadap checkpoint final.
3. Agregasi hasil ke `results.csv` dan `summary.csv` (mean ± std per
   konfigurasi di 3 seed), diurutkan menurut `--sort-by`.

Metrik inferensi mencakup `trade_off` = test_mean_score / mean_time (§5.3).

Resume: jika `metrics.json` valid sudah ada untuk sebuah
`(cfg_idx, seed)`, skrip melewatinya.

Contoh:
  cd FlowPolicy
  python scripts/random_search_kitchen.py --gpu 0
  python scripts/random_search_kitchen.py --gpu 0 --dry-run
  python scripts/random_search_kitchen.py --gpu 0 --sort-by trade_off
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple


SEARCH_SPACE: Dict[str, List[Any]] = {
    # --- Tabel §2 flowpolicy_hyperparameter_finetuning.md ---
    "training.num_epochs": [500, 1000, 3000, 5000],
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 1.0],
    "policy.Conditional_ConsistencyFM.delta": [1e-4, 1e-3, 1e-2, 1.0],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [4, 6, 8, 16],
    # Pemetaan dokumen: time_embedding_dim → diffusion_step_embed_dim
    "policy.diffusion_step_embed_dim": [128, 256, 512, 1024],
    # hidden_dim (MLP state encoder) → policy.state_mlp_size=[h,h] (di-expand di build_train_cmd)
    "_state_mlp_hidden": [128, 256, 512, 1024],
    # hidden_dim (lebar kanal UNet) → policy.down_dims (di-expand)
    "_unet_base_width": [128, 256, 512, 1024],
}

DEFAULT_SEEDS: List[int] = [0, 42, 101]
DEFAULT_N_CONFIGS: int = 30
DEFAULT_EVAL_EPISODES: int = 50


def sample_configs(n: int, sampling_seed: int) -> List[Dict[str, Any]]:
    """Sample `n` hyperparameter dicts independently & uniformly per key."""
    rng = random.Random(sampling_seed)
    configs: List[Dict[str, Any]] = []
    for _ in range(n):
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        configs.append(cfg)
    return configs


def format_override(key: str, value: Any) -> str:
    """Format single Hydra override. Floats sebaiknya dikirim dalam notasi
    scientific agar Hydra tidak salah parse integer-like float (mis. 1.0)."""
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, float):
        return f"{key}={value:.6g}"
    return f"{key}={value}"


def _expand_policy_arch_overrides(cfg: Dict[str, Any]) -> List[str]:
    """Kunci internal `_state_mlp_hidden` / `_unet_base_width` → override Hydra."""
    extra: List[str] = []
    h = cfg.get("_state_mlp_hidden")
    if h is not None:
        hi = int(h)
        extra.append(f"policy.state_mlp_size=[{hi},{hi}]")
    b = cfg.get("_unet_base_width")
    if b is not None:
        bi = int(b)
        d2 = min(2 * bi, 1024)
        d3 = min(4 * bi, 1024)
        extra.append(f"policy.down_dims=[{bi},{d2},{d3}]")
    return extra


def build_train_cmd(cfg: Dict[str, Any], seed: int, run_dir: pathlib.Path,
                    save_ckpt: bool = True,
                    preprocessing_profile: str = "standard") -> List[str]:
    overrides = [
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={seed}",
        "training.device=cuda",
        "training.debug=False",
        "exp_name=random_search",
        f"logging.name=random_search_{run_dir.name}",
        "logging.mode=offline",
        f"checkpoint.save_ckpt={'True' if save_ckpt else 'False'}",
        f"task.dataset.preprocessing_profile={preprocessing_profile}",
    ]
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        overrides.append(format_override(k, v))
        if k == "dataloader.batch_size":
            overrides.append(format_override("val_dataloader.batch_size", v))
    overrides.extend(_expand_policy_arch_overrides(cfg))

    return [sys.executable, "train.py", *overrides]


def build_infer_cmd(ckpt_path: pathlib.Path, episodes: int,
                    output_subdir: str) -> List[str]:
    return [
        sys.executable, "infer_kitchen.py",
        "--checkpoint", str(ckpt_path),
        "--episodes", str(episodes),
        "--device", "cuda:0",
        "--output-subdir", output_subdir,
    ]


def metrics_path_for(run_dir: pathlib.Path, output_subdir: str) -> pathlib.Path:
    return run_dir / output_subdir / "metrics.json"


def read_metrics(path: pathlib.Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def append_result_row(results_csv: pathlib.Path, row: Dict[str, Any]) -> None:
    write_header = not results_csv.exists()
    # Use explicit fieldnames so resumed runs keep the original column order.
    fieldnames = list(row.keys())
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_existing_results(results_csv: pathlib.Path) -> List[Dict[str, str]]:
    if not results_csv.exists():
        return []
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(results_csv: pathlib.Path, summary_csv: pathlib.Path,
                  configs: List[Dict[str, Any]], sort_by: str) -> None:
    rows = load_existing_results(results_csv)
    if not rows:
        print("[summary] results.csv kosong, summary dilewati.")
        return

    # group by cfg_idx
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        try:
            cid = int(r["cfg_idx"])
        except Exception:
            continue
        grouped.setdefault(cid, []).append(r)

    param_keys = list(SEARCH_SPACE.keys())
    metric_keys = [
        "test_mean_score",
        "mean_n_completed_tasks",
        "mean_time",
        "trade_off",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for cid, runs in grouped.items():
        if not runs:
            continue
        summary_row: Dict[str, Any] = {"cfg_idx": cid, "n_seeds": len(runs)}
        # hyperparameters (ambil dari konfigurasi yang di-sample, bukan csv,
        # agar tipe tetap numeric)
        if cid < len(configs):
            for k in param_keys:
                summary_row[k] = configs[cid][k]
        # metrics: mean + std
        for mk in metric_keys:
            vals: List[float] = []
            for r in runs:
                try:
                    vals.append(float(r.get(mk, "nan")))
                except Exception:
                    pass
            if vals:
                arr_mean = sum(vals) / len(vals)
                arr_std = (sum((x - arr_mean) ** 2 for x in vals) / len(vals)) ** 0.5
                summary_row[f"{mk}_mean"] = arr_mean
                summary_row[f"{mk}_std"] = arr_std
            else:
                summary_row[f"{mk}_mean"] = float("nan")
                summary_row[f"{mk}_std"] = float("nan")
        summary_rows.append(summary_row)

    def _finite_sort_key(row: Dict[str, Any], col: str) -> float:
        v = row.get(col, float("-inf"))
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("-inf")
        if x != x:  # NaN
            return float("-inf")
        return x

    sort_key = f"{sort_by}_mean"
    summary_rows.sort(
        key=lambda r: _finite_sort_key(r, sort_key),
        reverse=True,
    )

    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    for r in summary_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"[summary] {summary_csv}  ({len(summary_rows)} konfigurasi)")
    print(f"[summary] Top-3 (mean {sort_by}):")
    for r in summary_rows[:3]:
        m = r.get(f"{sort_by}_mean", float("nan"))
        s = r.get(f"{sort_by}_std", float("nan"))
        print(f"  cfg {r['cfg_idx']}: {sort_by}={m:.6g} ± {s:.6g}  "
              f"({r.get('n_seeds')} seeds)")


def run_subprocess(cmd: Sequence[str], log_path: pathlib.Path,
                   env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Random search FlowPolicy kitchen")
    parser.add_argument("--n-configs", type=int, default=DEFAULT_N_CONFIGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="Jumlah episode inferensi per run")
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/random_search")
    parser.add_argument("--sampling-seed", type=int, default=42,
                        help="Seed RNG untuk memilih kombinasi hyperparameter")
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry-run", action="store_true",
                        help="Cetak daftar konfigurasi tanpa menjalankan training")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Jangan tulis summary.csv di akhir")
    parser.add_argument(
        "--sort-by",
        choices=("test_mean_score", "trade_off"),
        default="test_mean_score",
        help="Kolom untuk mengurutkan summary.csv (mean ± std)",
    )
    parser.add_argument("--only-cfg", type=int, nargs="+", default=None,
                        help="Hanya jalankan cfg_idx ini (untuk debugging)")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Hanya baca results.csv + configs.json lalu tulis summary.csv "
             "(tidak training/infer). Pakai dengan --out-root yang sama.",
    )
    parser.add_argument(
        "--preprocessing-profile",
        choices=("standard", "minimal", "legacy_minimal"),
        default="standard",
        help="dataset.preprocessing_profile: standard | minimal (sliding, no aug) "
             "| legacy_minimal (stride=horizon, no split, no aug).",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    assert flowpolicy_dir.is_dir(), \
        f"Tidak menemukan folder FlowPolicy/FlowPolicy di {flowpolicy_dir}"

    out_root = (flowpolicy_dir / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "results.csv"
    summary_csv = out_root / "summary.csv"
    configs_json = out_root / "configs.json"

    if args.summary_only:
        if not results_csv.is_file() or not configs_json.is_file():
            print("[error] --summary-only memerlukan results.csv dan configs.json "
                  f"di {out_root}")
            return 1
        with open(configs_json, "r", encoding="utf-8") as f:
            configs_reload = json.load(f)
        write_summary(results_csv, summary_csv, configs_reload,
                      sort_by=args.sort_by)
        print("[done] summary-only selesai.")
        return 0

    configs = sample_configs(args.n_configs, args.sampling_seed)

    # Simpan daftar konfigurasi (stabil antar run bila --sampling-seed tetap).
    with open(configs_json, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    print(f"[search] {len(configs)} konfigurasi ditulis ke {configs_json}")

    if args.dry_run:
        print("[dry-run] daftar konfigurasi:")
        for i, cfg in enumerate(configs):
            if args.only_cfg and i not in args.only_cfg:
                continue
            print(f"  cfg {i:02d}: {cfg}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"

    total_runs = 0
    for i, cfg in enumerate(configs):
        if args.only_cfg and i not in args.only_cfg:
            continue
        for seed in args.seeds:
            total_runs += 1
            run_dir = (out_root / f"cfg_{i:02d}_seed{seed}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            infer_subdir = f"inference_ep{args.episodes}"
            mpath = metrics_path_for(run_dir, infer_subdir)

            print(f"\n========== [cfg {i:02d} / seed {seed}] "
                  f"({total_runs}) ==========")
            print(f"  cfg      : {cfg}")
            print(f"  run_dir  : {run_dir}")

            # Resume: metrics.json sudah ada & valid -> skip
            if mpath.is_file():
                try:
                    metrics = read_metrics(mpath)
                    if "test_mean_score" in metrics:
                        print(f"  [skip] metrics sudah ada: "
                              f"SR={metrics['test_mean_score']:.4f}")
                        _append_run_row(results_csv, i, seed, cfg, metrics,
                                        status="skip_resume",
                                        preprocessing_profile=args.preprocessing_profile)
                        continue
                except Exception:
                    pass

            # --- training ---
            t0 = time.time()
            ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
            if ckpt_path.is_file():
                print(f"  [skip-train] checkpoint sudah ada: {ckpt_path}")
            else:
                train_cmd = build_train_cmd(
                    cfg, seed, run_dir, save_ckpt=True,
                    preprocessing_profile=args.preprocessing_profile,
                )
                print(f"  [train] {' '.join(train_cmd)}")
                rc = run_subprocess(
                    train_cmd, run_dir / "train_stdout.log", env=env)
                if rc != 0 or not ckpt_path.is_file():
                    print(f"  [ERROR] training gagal (rc={rc}), skip inferensi.")
                    _append_run_row(results_csv, i, seed, cfg, {},
                                    status=f"train_failed_rc{rc}",
                                    t_train=time.time() - t0,
                                    preprocessing_profile=args.preprocessing_profile)
                    continue
            t_train = time.time() - t0

            # --- inference (50 ep) ---
            t1 = time.time()
            infer_cmd = build_infer_cmd(
                ckpt_path, episodes=args.episodes, output_subdir=infer_subdir)
            print(f"  [infer] {' '.join(infer_cmd)}")
            rc = run_subprocess(
                infer_cmd, run_dir / "infer_stdout.log", env=env)
            t_infer = time.time() - t1

            metrics: Dict[str, float] = {}
            if rc == 0 and mpath.is_file():
                try:
                    metrics = read_metrics(mpath)
                except Exception as exc:
                    print(f"  [WARN] gagal parse metrics: {exc}")
            else:
                print(f"  [ERROR] inferensi gagal (rc={rc})")

            _append_run_row(results_csv, i, seed, cfg, metrics,
                            status="ok" if metrics else f"infer_failed_rc{rc}",
                            t_train=t_train, t_infer=t_infer,
                            preprocessing_profile=args.preprocessing_profile)

    # Summary
    if not args.skip_summary:
        write_summary(results_csv, summary_csv, configs, sort_by=args.sort_by)
    print("[done] Selesai.")
    return 0


def _append_run_row(results_csv: pathlib.Path, cfg_idx: int, seed: int,
                    cfg: Dict[str, Any], metrics: Dict[str, float],
                    status: str,
                    t_train: float = 0.0, t_infer: float = 0.0,
                    preprocessing_profile: str = "standard") -> None:
    row: Dict[str, Any] = {"cfg_idx": cfg_idx, "seed": seed}
    row["dataset.preprocessing_profile"] = preprocessing_profile
    for k in SEARCH_SPACE.keys():
        row[k] = cfg.get(k)
    for mk in ("test_mean_score", "mean_n_completed_tasks", "mean_time",
               "mean_success_rates", "SR_test_L3", "SR_test_L5", "trade_off"):
        v = metrics.get(mk)
        row[mk] = float(v) if v is not None else ""
    if row.get("trade_off") == "":
        try:
            sr = float(row.get("test_mean_score", ""))
            mt = float(row.get("mean_time", ""))
            if mt > 1e-12:
                row["trade_off"] = sr / mt
        except (TypeError, ValueError):
            pass
    row["status"] = status
    row["t_train_s"] = round(t_train, 2)
    row["t_infer_s"] = round(t_infer, 2)
    row["timestamp"] = int(time.time())
    append_result_row(results_csv, row)


if __name__ == "__main__":
    # Pindah ke folder FlowPolicy/FlowPolicy agar train.py & infer_kitchen.py
    # menemukan path config & cwd yang konsisten.
    _here = pathlib.Path(__file__).resolve().parent
    _flowpolicy_pkg = _here.parent / "FlowPolicy"
    os.chdir(_flowpolicy_pkg)
    sys.exit(main())
