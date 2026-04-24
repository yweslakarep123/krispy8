#!/usr/bin/env python3
"""Loop inferensi/evaluasi untuk seluruh 96 run OFAT.

Untuk setiap `cfg_<ii>_seed<s>/` di `out-root`, skrip ini:
  1. mencari `checkpoints/latest.ckpt` (skip bila belum ada),
  2. menjalankan `infer_kitchen.py` → menghasilkan `inference_ep<N>/`
     berisi `metrics.json` dan (opsional) folder `videos/`,
  3. mengumpulkan hasilnya ke `infer_results.csv` (baris per run).

Resume otomatis: bila `metrics.json` sudah ada dan valid, run di-skip
kecuali `--force` diset.

Contoh:
  # inferensi (dengan video) untuk semua 96 run
  bash scripts/ofat_infer_kitchen.sh 0

  # evaluasi cepat tanpa video
  bash scripts/ofat_eval_kitchen.sh 0

  # dry-run cek mapping cfg -> ckpt
  bash scripts/ofat_infer_kitchen.sh 0 --dry-run
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

# Share search space & baseline dari skrip OFAT train supaya cfg_idx -> HP
# tetap konsisten satu sumber kebenaran.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ofat_search_kitchen import (  # type: ignore  # noqa: E402
    SEARCH_SPACE,
    BASELINE,
    DEFAULT_SEEDS,
    enumerate_configs,
)


_CFG_SEED_RE = re.compile(r"^cfg_(\d+)_seed(-?\d+)$")


def discover_runs(out_root: pathlib.Path) -> List[Tuple[int, int, pathlib.Path]]:
    """Temukan semua folder `cfg_<ii>_seed<s>` di `out_root`. Kembalikan
    (cfg_idx, seed, run_dir) terurut."""
    runs: List[Tuple[int, int, pathlib.Path]] = []
    if not out_root.is_dir():
        return runs
    for p in sorted(out_root.iterdir()):
        if not p.is_dir():
            continue
        m = _CFG_SEED_RE.match(p.name)
        if not m:
            continue
        cfg_idx = int(m.group(1))
        seed = int(m.group(2))
        runs.append((cfg_idx, seed, p))
    runs.sort(key=lambda r: (r[0], r[1]))
    return runs


def read_metrics(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def lookup_cfg(enum_cfgs, cfg_idx: int
               ) -> Tuple[Optional[str], Optional[Any], Optional[Dict[str, Any]]]:
    for (gidx, hp_key, hp_value, cfg) in enum_cfgs:
        if gidx == cfg_idx:
            return hp_key, hp_value, cfg
    return None, None, None


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loop inferensi/evaluasi untuk semua run OFAT")
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/ofat_search",
                        help="Folder root hasil OFAT training.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Shortcut: bila diset dan --out-root default, "
                             "out-root otomatis jadi "
                             "`data/outputs/ofat_search_preprocess`.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip perekaman video (evaluasi cepat).")
    parser.add_argument("--force", action="store_true",
                        help="Paksa ulangi inferensi meski metrics.json sudah ada.")
    parser.add_argument("--only-cfg", type=int, nargs="+", default=None)
    parser.add_argument("--only-hp", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Filter seed. Default: semua seed yang ditemukan.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subdir-suffix", type=str, default="",
                        help="Suffix tambahan untuk nama subfolder hasil "
                             "(mis. `_novid`). Default: kosong, memakai "
                             "`inference_ep<N>` kalau dengan video atau "
                             "`inference_ep<N>_novideo` kalau --no-video.")
    args = parser.parse_args()

    default_root = "data/outputs/ofat_search"
    if args.preprocess and args.out_root == default_root:
        args.out_root = default_root + "_preprocess"

    flowpolicy_dir = (_HERE.parent / "FlowPolicy").resolve()
    assert flowpolicy_dir.is_dir(), \
        f"Tidak menemukan folder package di {flowpolicy_dir}"
    out_root = (flowpolicy_dir / args.out_root).resolve()
    if not out_root.is_dir():
        print(f"ERROR: out-root tidak ditemukan: {out_root}", file=sys.stderr)
        return 2

    enum_cfgs = enumerate_configs()
    runs = discover_runs(out_root)

    if args.only_cfg:
        only_set = set(args.only_cfg)
        runs = [r for r in runs if r[0] in only_set]
    if args.seeds:
        seed_set = set(args.seeds)
        runs = [r for r in runs if r[1] in seed_set]
    if args.only_hp:
        hp_map = {g: k for (g, k, _, _) in enum_cfgs}
        runs = [r for r in runs if hp_map.get(r[0]) == args.only_hp]

    if not runs:
        print(f"ERROR: tidak ada run yang cocok di {out_root}", file=sys.stderr)
        return 3

    # subdir hasil inferensi
    if args.subdir_suffix:
        suffix = args.subdir_suffix
    elif args.no_video:
        suffix = "_novideo"
    else:
        suffix = ""
    infer_subdir = f"inference_ep{args.episodes}{suffix}"

    csv_name = "eval_results.csv" if args.no_video else "infer_results.csv"
    results_csv = out_root / csv_name

    print(f"[ofat-infer] out-root       : {out_root}")
    print(f"[ofat-infer] runs ditemukan : {len(runs)}")
    print(f"[ofat-infer] infer_subdir   : {infer_subdir}")
    print(f"[ofat-infer] episodes       : {args.episodes}")
    print(f"[ofat-infer] video          : {'OFF' if args.no_video else 'ON'}")
    print(f"[ofat-infer] results_csv    : {results_csv}")

    if args.dry_run:
        for (cfg_idx, seed, run_dir) in runs:
            hp_key, hp_val, _ = lookup_cfg(enum_cfgs, cfg_idx)
            ckpt = run_dir / "checkpoints" / "latest.ckpt"
            print(f"  cfg {cfg_idx:02d} seed {seed:<3d} "
                  f"[{hp_key}={hp_val}] ckpt_exists={ckpt.is_file()} "
                  f"run_dir={run_dir}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # fieldnames tetap konsisten (append row aman bila header cocok)
    fieldnames = [
        "cfg_idx", "seed", "swept_hp", "swept_value",
        *SEARCH_SPACE.keys(),
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
        hp_key, hp_val, cfg_full = lookup_cfg(enum_cfgs, cfg_idx)
        ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
        out_subdir = run_dir / infer_subdir
        metrics_path = out_subdir / "metrics.json"

        print(f"\n========== [{done}/{len(runs)}] cfg {cfg_idx:02d} seed {seed} "
              f"[{hp_key}={hp_val}] ==========")
        print(f"  run_dir : {run_dir}")

        if not ckpt_path.is_file():
            print(f"  [skip] ckpt tidak ada: {ckpt_path}")
            row = _make_row(cfg_idx, seed, hp_key, hp_val, cfg_full,
                            metrics=None, status="missing_ckpt",
                            t_infer=0.0, video_count=0,
                            run_dir=run_dir, fieldnames=fieldnames)
            _append_row(results_csv, fieldnames, row)
            continue

        if not args.force and metrics_path.is_file():
            try:
                metrics = read_metrics(metrics_path)
                if "test_mean_score" in metrics:
                    print(f"  [skip-resume] metrics sudah ada: "
                          f"SR={float(metrics['test_mean_score']):.4f}")
                    row = _make_row(cfg_idx, seed, hp_key, hp_val, cfg_full,
                                    metrics=metrics, status="skip_resume",
                                    t_infer=0.0,
                                    video_count=len(metrics.get(
                                        "saved_video_paths", []) or []),
                                    run_dir=run_dir, fieldnames=fieldnames)
                    _append_row(results_csv, fieldnames, row)
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
        row = _make_row(cfg_idx, seed, hp_key, hp_val, cfg_full, metrics,
                        status=status, t_infer=t_infer,
                        video_count=video_count, run_dir=run_dir,
                        fieldnames=fieldnames)
        _append_row(results_csv, fieldnames, row)
        elapsed = time.time() - t_start
        print(f"  [{status}] t_infer={t_infer:.1f}s  elapsed={fmt_hms(elapsed)}")

    print(f"\n[ofat-infer] selesai. Total elapsed: {fmt_hms(time.time() - t_start)}")
    print(f"[ofat-infer] hasil: {results_csv}")
    return 0


def _make_row(cfg_idx: int, seed: int, hp_key: Optional[str], hp_val: Any,
              cfg_full: Optional[Dict[str, Any]],
              metrics: Optional[Dict[str, Any]], status: str,
              t_infer: float, video_count: int,
              run_dir: pathlib.Path, fieldnames: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: "" for k in fieldnames}
    row["cfg_idx"] = cfg_idx
    row["seed"] = seed
    row["swept_hp"] = hp_key
    row["swept_value"] = hp_val
    if cfg_full is not None:
        for k in SEARCH_SPACE.keys():
            row[k] = cfg_full.get(k)
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


def _append_row(csv_path: pathlib.Path, fieldnames: List[str],
                row: Dict[str, Any]) -> None:
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


if __name__ == "__main__":
    # Agen dijalankan dari folder `scripts/`, tapi `infer_kitchen.py` butuh
    # cwd = package dir.
    pkg_dir = (_HERE.parent / "FlowPolicy").resolve()
    os.chdir(pkg_dir)
    sys.exit(main())
