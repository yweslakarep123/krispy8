#!/usr/bin/env python3
"""Random search hyperparameter untuk FlowPolicy kitchen.

Alur:
1) Sample n_iter kandidat hyperparameter.
2) Untuk tiap kandidat, jalankan training + inferensi per fold (cv).
3) Skor kandidat = rerata test_mean_score lintas fold.
4) Simpan results.csv dan best_trial.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time
from typing import Any, Dict, List


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ",".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def format_override(key: str, value: Any) -> str:
    return f"{key}={_fmt_value(value)}"


def _run(cmd: List[str], cwd: pathlib.Path, env: Dict[str, str], log_path: pathlib.Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if dry_run:
        line = f"{stamp} | DRY-RUN | {' '.join(cmd)}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
        print(line.strip())
        return 0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{stamp} | CMD | {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def _read_metrics(metrics_path: pathlib.Path) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_gpu_or_fail(py: str, env: Dict[str, str], gpu_arg: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[gpu-check] skip (dry-run), target GPU={gpu_arg}")
        return
    cmd = [
        py,
        "-c",
        (
            "import torch; "
            "assert torch.cuda.is_available(), 'CUDA tidak tersedia'; "
            "print(torch.cuda.device_count())"
        ),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[gpu-check] gagal: {proc.stderr.strip() or proc.stdout.strip()}")
    out = (proc.stdout or "").strip()
    print(f"[gpu-check] CUDA ready. visible_device_count={out} (CUDA_VISIBLE_DEVICES={gpu_arg})")


def _sample_candidates(rng: random.Random, n_iter: int) -> List[Dict[str, Any]]:
    space = {
        "training.num_epochs": [300, 500, 800, 1000],
        "optimizer.lr": [3e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        "dataloader.batch_size": [64, 128, 256],
        "policy.Conditional_ConsistencyFM.num_segments": [2, 4, 8, 16],
        "policy.Conditional_ConsistencyFM.eps": [1e-4, 5e-4, 1e-3, 1e-2],
        "policy.Conditional_ConsistencyFM.delta": [1e-4, 5e-4, 1e-3, 1e-2],
        "n_action_steps": [2, 4, 6, 8],
        "n_obs_steps": [2, 4, 6, 8, 12, 16],
    }
    candidates: List[Dict[str, Any]] = []
    seen = set()
    max_trials = max(n_iter * 20, 1000)
    attempts = 0
    while len(candidates) < n_iter and attempts < max_trials:
        attempts += 1
        cand = {k: rng.choice(vs) for k, vs in space.items()}
        key = tuple(sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)
    return candidates


def _build_train_cmd(
    py: str,
    run_dir: pathlib.Path,
    seed: int,
    preprocess: bool,
    params: Dict[str, Any],
    logging_name: str,
    split_seed: int,
) -> List[str]:
    cmd = [
        py,
        "train.py",
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        format_override("hydra.run.dir", str(run_dir)),
        format_override("training.seed", seed),
        "training.device=cuda",
        "training.debug=False",
        "exp_name=random_search",
        format_override("logging.name", logging_name),
        "logging.mode=offline",
        "checkpoint.save_ckpt=True",
        "training.resume=False",
        format_override("task.dataset.preprocess.enabled", preprocess),
        format_override("task.dataset.preprocess.split_seed", split_seed),
    ]
    for k, v in params.items():
        cmd.append(format_override(k, v))
        if k == "dataloader.batch_size":
            cmd.append(format_override("val_dataloader.batch_size", v))
    return cmd


def _build_infer_cmd(py: str, ckpt: pathlib.Path, episodes: int, out_subdir: str) -> List[str]:
    return [
        py,
        "infer_kitchen.py",
        "--checkpoint",
        str(ckpt),
        "--episodes",
        str(episodes),
        "--device",
        "cuda:0",
        "--output-subdir",
        out_subdir,
        "--no-video",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random search FlowPolicy kitchen")
    p.add_argument("--seed", type=int, required=True, help="Seed utama random search.")
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--scenario-name", type=str, default="scenario")
    p.add_argument(
        "--strict-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail jika CUDA tidak tersedia.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    here = pathlib.Path(__file__).resolve().parent
    flowpolicy_pkg = here.parent / "FlowPolicy"
    out_root = pathlib.Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["FLOWP_GPU_PERF_MODE"] = "1"

    if args.strict_gpu:
        _check_gpu_or_fail(py, env, args.gpu, args.dry_run)

    rng = random.Random(args.random_state + args.seed * 10007 + (1 if args.preprocess else 0))
    candidates = _sample_candidates(rng, args.n_iter)
    if not candidates:
        raise SystemExit("Tidak ada kandidat random search yang berhasil dibangkitkan.")

    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    total_fold_jobs = len(candidates) * args.cv
    fold_job_idx = 0
    print(
        f"[random_search] start scenario={args.scenario_name} seed={args.seed} "
        f"n_iter={len(candidates)} cv={args.cv} preprocess={args.preprocess}"
    )

    for cfg_idx, params in enumerate(candidates):
        print(f"[random_search] kandidat {cfg_idx + 1}/{len(candidates)}")
        fold_scores: List[float] = []
        fold_latencies: List[float] = []
        status = "ok"
        t_cfg_start = time.time()
        cfg_dir = out_root / f"cfg_{cfg_idx:04d}_seed{args.seed}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        params_json = cfg_dir / "params.json"
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        for fold_idx in range(args.cv):
            fold_job_idx += 1
            pct = 100.0 * fold_job_idx / total_fold_jobs if total_fold_jobs else 0.0
            print(
                f"[random_search][progress] {fold_job_idx}/{total_fold_jobs} ({pct:.1f}%) "
                f"cfg={cfg_idx + 1}/{len(candidates)} cv={fold_idx + 1}/{args.cv}"
            )
            fold_dir = cfg_dir / f"cv_{fold_idx:02d}"
            run_dir = fold_dir / "train_run"
            split_seed = args.seed * 1000 + fold_idx
            train_seed = args.seed * 100 + fold_idx
            log_name = f"random_cfg{cfg_idx:04d}_seed{args.seed}_cv{fold_idx:02d}"
            rc_train = _run(
                _build_train_cmd(
                    py=py,
                    run_dir=run_dir,
                    seed=train_seed,
                    preprocess=args.preprocess,
                    params=params,
                    logging_name=log_name,
                    split_seed=split_seed,
                ),
                cwd=flowpolicy_pkg,
                env=env,
                log_path=fold_dir / "train.log",
                dry_run=args.dry_run,
            )
            if rc_train != 0:
                status = f"train_fail_cv{fold_idx}"
                break

            ckpt = run_dir / "checkpoints" / "latest.ckpt"
            if not args.dry_run and not ckpt.is_file():
                status = f"missing_ckpt_cv{fold_idx}"
                break

            rc_inf = _run(
                _build_infer_cmd(py=py, ckpt=ckpt, episodes=args.eval_episodes, out_subdir="inference_cv_eval"),
                cwd=flowpolicy_pkg,
                env=env,
                log_path=fold_dir / "infer.log",
                dry_run=args.dry_run,
            )
            if rc_inf != 0:
                status = f"infer_fail_cv{fold_idx}"
                break

            metrics_path = run_dir / "inference_cv_eval" / "metrics.json"
            if not args.dry_run and metrics_path.is_file():
                metrics = _read_metrics(metrics_path)
                score = float(metrics.get("test_mean_score", float("nan")))
                lat_ms = float(metrics.get("mean_time", float("nan"))) * 1000.0
                if math.isfinite(score):
                    fold_scores.append(score)
                if math.isfinite(lat_ms):
                    fold_latencies.append(lat_ms)

        mean_score = float("nan")
        mean_lat_ms = float("nan")
        if fold_scores:
            mean_score = sum(fold_scores) / len(fold_scores)
        if fold_latencies:
            mean_lat_ms = sum(fold_latencies) / len(fold_latencies)
        if len(fold_scores) < args.cv and not args.dry_run and status == "ok":
            status = "partial_cv"

        row: Dict[str, Any] = {
            "cfg_idx": cfg_idx,
            "seed": args.seed,
            "cv": args.cv,
            "folds_ok": len(fold_scores),
            "preprocess": args.preprocess,
            "mean_test_mean_score_cv": mean_score,
            "mean_latency_ms_cv": mean_lat_ms,
            "status": status,
            "t_total_s": time.time() - t_cfg_start,
            **params,
        }
        rows.append(row)
        score_txt = "nan" if not math.isfinite(mean_score) else f"{mean_score:.4f}"
        lat_txt = "nan" if not math.isfinite(mean_lat_ms) else f"{mean_lat_ms:.3f}"
        print(
            f"[random_search] kandidat selesai cfg={cfg_idx:04d} "
            f"score_cv={score_txt} latency_ms_cv={lat_txt} status={status}"
        )
        if best_row is None:
            best_row = row
        else:
            old = float(best_row.get("mean_test_mean_score_cv", float("nan")))
            new = float(row.get("mean_test_mean_score_cv", float("nan")))
            if (not math.isfinite(old) and math.isfinite(new)) or (math.isfinite(new) and new > old):
                best_row = row

    results_csv = out_root / "results.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if best_row is None:
        raise SystemExit("Random search tidak menghasilkan kandidat valid.")

    best_trial = {
        "cfg_idx": int(best_row["cfg_idx"]),
        "seed": int(args.seed),
        "preprocess": bool(args.preprocess),
        "cv": int(args.cv),
        "score_key": "mean_test_mean_score_cv",
        "best_score": float(best_row.get("mean_test_mean_score_cv", float("nan"))),
        "best_latency_ms_cv": float(best_row.get("mean_latency_ms_cv", float("nan"))),
        "params": {k: v for k, v in best_row.items() if k in candidates[0]},
        "results_csv": str(results_csv),
    }
    with open(out_root / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(best_trial, f, indent=2)

    print(f"[random_search] results -> {results_csv}")
    print(f"[random_search] best -> {out_root / 'best_trial.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
