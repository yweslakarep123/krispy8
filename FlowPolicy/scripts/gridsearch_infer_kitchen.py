#!/usr/bin/env python3
"""Loop inferensi/evaluasi untuk seluruh run Grid Search.

Untuk setiap folder `cfg_<ii>_seed<s>/` di `out-root`, skrip ini:
  1. cari `checkpoints/latest.ckpt` (skip bila belum ada),
  2. jalankan `infer_kitchen.py` → menghasilkan `inference_ep<N>/`
     berisi `metrics.json` dan (opsional) folder `videos/`,
  3. agregasi hasil ke `infer_results.csv` (dengan kolom subset yang
     dibaca otomatis dari `configs.json` di out-root).

Beda dengan `ofat_infer_kitchen.py`:
  - Subset HP tidak hardcode; diambil dari field `sweep_subset` di
    `configs.json` yang dihasilkan oleh `gridsearch_kitchen.py`.
  - Mendukung pola `cfg_<4digit>_seed<seed>` (bukan 2 digit).

Contoh:
  bash scripts/gridsearch_infer_kitchen.sh 0
  bash scripts/gridsearch_eval_kitchen.sh 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


_HERE = pathlib.Path(__file__).resolve().parent

# reuse helper _fmt_value untuk serialisasi nilai list saat masuk ke CSV
sys.path.insert(0, str(_HERE))
from gridsearch_kitchen import _fmt_value  # type: ignore  # noqa: E402


_CFG_SEED_RE = re.compile(r"^cfg_(\d+)_seed(-?\d+)$")


def discover_runs(out_root: pathlib.Path
                  ) -> List[Tuple[int, int, pathlib.Path]]:
    runs: List[Tuple[int, int, pathlib.Path]] = []
    if not out_root.is_dir():
        return runs
    for p in sorted(out_root.iterdir()):
        if not p.is_dir():
            continue
        m = _CFG_SEED_RE.match(p.name)
        if not m:
            continue
        runs.append((int(m.group(1)), int(m.group(2)), p))
    runs.sort(key=lambda r: (r[0], r[1]))
    return runs


def load_configs_json(configs_json: pathlib.Path
                      ) -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
    """Return (subset_keys, cfg_idx -> subset_values)."""
    with open(configs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return [], {}
    subset_keys = list(data[0].get("sweep_subset", {}).keys())
    cfg_map: Dict[int, Dict[str, Any]] = {}
    for entry in data:
        cfg_idx = int(entry["cfg_idx"])
        cfg_map[cfg_idx] = entry.get("subset_values", {})
    return subset_keys, cfg_map


def build_infer_cmd(ckpt_path: pathlib.Path, episodes: int,
                    output_subdir: str, no_video: bool,
                    device: str) -> List[str]:
    cmd = [
        sys.executable, "infer_kitchen.py",
        "--checkpoint", str(ckpt_path),
        "--episodes", str(episodes),
        "--device", device,
        "--output-subdir", output_subdir,
    ]
    if no_video:
        cmd.append("--no-video")
    return cmd


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


def read_metrics(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loop inferensi/evaluasi untuk semua run grid search")
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/gridsearch",
                        help="Folder root hasil grid search training.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Shortcut: bila diset dan --out-root default, "
                             "alihkan ke data/outputs/gridsearch_preprocess.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Paksa ulangi inferensi meski metrics.json sudah ada.")
    parser.add_argument("--only-cfg", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subdir-suffix", type=str, default="")
    args = parser.parse_args()

    default_root = "data/outputs/gridsearch"
    if args.preprocess and args.out_root == default_root:
        args.out_root = default_root + "_preprocess"

    flowpolicy_dir = (_HERE.parent / "FlowPolicy").resolve()
    out_root = (flowpolicy_dir / args.out_root).resolve()
    if not out_root.is_dir():
        print(f"ERROR: out-root tidak ditemukan: {out_root}", file=sys.stderr)
        return 2

    configs_json = out_root / "configs.json"
    if not configs_json.is_file():
        print(f"ERROR: configs.json tidak ada di {out_root}. "
              f"Jalankan gridsearch_kitchen.sh dulu.", file=sys.stderr)
        return 2

    subset_keys, cfg_map = load_configs_json(configs_json)

    runs = discover_runs(out_root)
    if args.only_cfg:
        only_set = set(args.only_cfg)
        runs = [r for r in runs if r[0] in only_set]
    if args.seeds:
        seed_set = set(args.seeds)
        runs = [r for r in runs if r[1] in seed_set]

    if not runs:
        print(f"ERROR: tidak ada run yang cocok di {out_root}", file=sys.stderr)
        return 3

    if args.subdir_suffix:
        suffix = args.subdir_suffix
    elif args.no_video:
        suffix = "_novideo"
    else:
        suffix = ""
    infer_subdir = f"inference_ep{args.episodes}{suffix}"

    csv_name = "eval_results.csv" if args.no_video else "infer_results.csv"
    results_csv = out_root / csv_name

    print(f"[grid-infer] out-root       : {out_root}")
    print(f"[grid-infer] runs ditemukan : {len(runs)}")
    print(f"[grid-infer] infer_subdir   : {infer_subdir}")
    print(f"[grid-infer] episodes       : {args.episodes}")
    print(f"[grid-infer] video          : {'OFF' if args.no_video else 'ON'}")
    print(f"[grid-infer] results_csv    : {results_csv}")

    if args.dry_run:
        for (cfg_idx, seed, run_dir) in runs:
            sv = cfg_map.get(cfg_idx, {})
            ckpt = run_dir / "checkpoints" / "latest.ckpt"
            print(f"  cfg {cfg_idx:04d} seed {seed:<3d}  "
                  f"ckpt_exists={ckpt.is_file()}  subset={sv}  "
                  f"run_dir={run_dir}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    fieldnames = [
        "cfg_idx", "seed",
        *subset_keys,
        "test_mean_score", "mean_n_completed_tasks",
        "mean_time_s", "mean_time_ms",
        "status", "t_infer_s", "video_count", "run_dir", "timestamp",
    ]
    write_header = not results_csv.exists()
    if write_header:
        with open(results_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    t_start = time.time()
    done = 0
    for (cfg_idx, seed, run_dir) in runs:
        done += 1
        sv = cfg_map.get(cfg_idx, {})
        ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
        out_subdir = run_dir / infer_subdir
        metrics_path = out_subdir / "metrics.json"

        print(f"\n========== [{done}/{len(runs)}] cfg {cfg_idx:04d} seed {seed}"
              f" ==========")
        print(f"  run_dir : {run_dir}")
        print(f"  subset  : {sv}")

        if not ckpt_path.is_file():
            print(f"  [skip] ckpt tidak ada: {ckpt_path}")
            _append(results_csv, fieldnames, _row(cfg_idx, seed, sv,
                                                  subset_keys,
                                                  metrics=None,
                                                  status="missing_ckpt",
                                                  t_infer=0.0,
                                                  video_count=0,
                                                  run_dir=run_dir,
                                                  fieldnames=fieldnames))
            continue

        if not args.force and metrics_path.is_file():
            try:
                metrics = read_metrics(metrics_path)
                if "test_mean_score" in metrics:
                    print(f"  [skip-resume] metrics sudah ada: "
                          f"SR={float(metrics['test_mean_score']):.4f}")
                    _append(results_csv, fieldnames,
                            _row(cfg_idx, seed, sv, subset_keys,
                                 metrics=metrics, status="skip_resume",
                                 t_infer=0.0,
                                 video_count=len(metrics.get(
                                     "saved_video_paths", []) or []),
                                 run_dir=run_dir, fieldnames=fieldnames))
                    continue
            except Exception:
                pass

        cmd = build_infer_cmd(ckpt_path, episodes=args.episodes,
                              output_subdir=infer_subdir,
                              no_video=args.no_video, device=args.device)
        log_file = run_dir / ("eval_stdout.log" if args.no_video
                              else "infer_stdout.log")
        print(f"  [run] {' '.join(cmd)}")
        t0 = time.time()
        rc = run_subprocess(cmd, log_file, env=env)
        t_infer = time.time() - t0

        metrics: Optional[Dict[str, Any]] = None
        if rc == 0 and metrics_path.is_file():
            try:
                metrics = read_metrics(metrics_path)
            except Exception as exc:
                print(f"  [warn] gagal parse metrics: {exc}")

        status = "ok" if metrics else f"infer_failed_rc{rc}"
        video_count = len(metrics.get("saved_video_paths", []) or []) \
            if metrics else 0
        _append(results_csv, fieldnames,
                _row(cfg_idx, seed, sv, subset_keys, metrics, status=status,
                     t_infer=t_infer, video_count=video_count,
                     run_dir=run_dir, fieldnames=fieldnames))
        elapsed = time.time() - t_start
        print(f"  [{status}] t_infer={t_infer:.1f}s  elapsed={fmt_hms(elapsed)}")

    print(f"\n[grid-infer] selesai. Total elapsed: "
          f"{fmt_hms(time.time() - t_start)}")
    print(f"[grid-infer] hasil: {results_csv}")
    return 0


def _row(cfg_idx: int, seed: int, subset_values: Dict[str, Any],
         subset_keys: List[str], metrics: Optional[Dict[str, Any]],
         status: str, t_infer: float, video_count: int,
         run_dir: pathlib.Path, fieldnames: List[str]) -> Dict[str, Any]:
    row = {k: "" for k in fieldnames}
    row["cfg_idx"] = cfg_idx
    row["seed"] = seed
    for k in subset_keys:
        v = subset_values.get(k)
        row[k] = _fmt_value(v) if isinstance(v, list) else v
    if metrics:
        tms = metrics.get("test_mean_score")
        row["test_mean_score"] = float(tms) if tms is not None else ""
        mnc = metrics.get("mean_n_completed_tasks")
        row["mean_n_completed_tasks"] = float(mnc) if mnc is not None else ""
        mt = metrics.get("mean_time")
        if mt is not None:
            row["mean_time_s"] = float(mt)
            row["mean_time_ms"] = float(mt) * 1000.0
    row["status"] = status
    row["t_infer_s"] = round(float(t_infer), 2)
    row["video_count"] = int(video_count)
    row["run_dir"] = str(run_dir)
    row["timestamp"] = int(time.time())
    return row


def _append(csv_path: pathlib.Path, fieldnames: List[str],
            row: Dict[str, Any]) -> None:
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


if __name__ == "__main__":
    pkg_dir = (_HERE.parent / "FlowPolicy").resolve()
    os.chdir(pkg_dir)
    sys.exit(main())
