#!/usr/bin/env python3
"""Orkestrasi 3 seed x 4 skenario untuk FlowPolicy kitchen.

Per seed:
1) baseline      : tanpa tuning, tanpa preprocess
2) baseline_pre  : tanpa tuning, dengan preprocess
3) tuned         : halving tuning tanpa preprocess -> train final best params
4) tuned_pre     : halving tuning dengan preprocess -> train final best params
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


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


def _build_train_cmd(
    py: str,
    run_dir: pathlib.Path,
    seed: int,
    preprocess: bool,
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
        "exp_name=exp_3seed_4arms",
        format_override("logging.name", f"seed{seed}_{run_dir.parent.name}"),
        "logging.mode=offline",
        "checkpoint.save_ckpt=True",
        "training.resume=False",
    ]
    cmd.append(format_override("task.dataset.preprocess.enabled", preprocess))
    return cmd


def _eval_model(
    py: str,
    flowpolicy_pkg: pathlib.Path,
    env: Dict[str, str],
    ckpt_path: pathlib.Path,
    episodes: int,
    log_path: pathlib.Path,
    out_subdir: str = "inference_suite_eval",
    no_video: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cmd = [
        py,
        "infer_kitchen.py",
        "--checkpoint",
        str(ckpt_path),
        "--episodes",
        str(episodes),
        "--device",
        "cuda:0",
        "--output-subdir",
        out_subdir,
    ]
    if no_video:
        cmd.append("--no-video")
    rc = _run(cmd, cwd=flowpolicy_pkg, env=env, log_path=log_path, dry_run=dry_run)
    out_dir = ckpt_path.parent.parent / out_subdir
    metrics_path = out_dir / "metrics.json"
    row: Dict[str, Any] = {
        "eval_rc": rc,
        "metrics_path": str(metrics_path),
        "eval_test_mean_score": float("nan"),
        "eval_mean_latency_ms": float("nan"),
    }
    if rc == 0 and metrics_path.is_file():
        metrics = _read_metrics(metrics_path)
        row["eval_test_mean_score"] = float(metrics.get("test_mean_score", float("nan")))
        row["eval_mean_latency_ms"] = float(metrics.get("mean_time", float("nan"))) * 1000.0
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[f"metric_{k}"] = float(v)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eksperimen 3 seed x 4 skenario")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 101])
    p.add_argument("--out-root", type=str, default="data/outputs/exp_3seed_4arms")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--hero-episodes", type=int, default=20)
    p.add_argument("--hero-video", action="store_true", help="Aktifkan video untuk inferensi final juara global.")
    p.add_argument("--halving-n-candidates", type=int, default=16)
    p.add_argument("--halving-factor", type=float, default=2.0)
    p.add_argument("--halving-min-episodes", type=int, default=10)
    p.add_argument("--halving-max-episodes", type=int, default=50)
    p.add_argument("--halving-random-state", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    here = pathlib.Path(__file__).resolve().parent
    flowpolicy_pkg = here.parent / "FlowPolicy"
    if not flowpolicy_pkg.is_dir():
        raise SystemExit(f"Tidak menemukan package dir: {flowpolicy_pkg}")

    out_root = (flowpolicy_pkg / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "experiment.log"

    py = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    all_rows: List[Dict[str, Any]] = []
    seed_winners: Dict[str, Any] = {}

    for seed in args.seeds:
        seed_dir = out_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[exp] ===== seed {seed} =====")

        # baseline
        baseline_run = seed_dir / "baseline" / "train_run"
        baseline_run.mkdir(parents=True, exist_ok=True)
        rc = _run(
            _build_train_cmd(py, baseline_run, seed, preprocess=False),
            cwd=flowpolicy_pkg,
            env=env,
            log_path=seed_dir / "baseline" / "train.log",
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc
        baseline_ckpt = baseline_run / "checkpoints" / "latest.ckpt"

        # baseline_pre
        baseline_pre_run = seed_dir / "baseline_pre" / "train_run"
        baseline_pre_run.mkdir(parents=True, exist_ok=True)
        rc = _run(
            _build_train_cmd(py, baseline_pre_run, seed, preprocess=True),
            cwd=flowpolicy_pkg,
            env=env,
            log_path=seed_dir / "baseline_pre" / "train.log",
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc
        baseline_pre_ckpt = baseline_pre_run / "checkpoints" / "latest.ckpt"

        # tuned (halving -> train best)
        tuned_dir = seed_dir / "tuned"
        tuned_halving_out = tuned_dir / "halving"
        tuned_train_run = tuned_dir / "train_run"
        tuned_dir.mkdir(parents=True, exist_ok=True)
        halving_cmd = [
            py,
            str(here / "halving_search_kitchen.py"),
            "--n-candidates",
            str(args.halving_n_candidates),
            "--factor",
            str(args.halving_factor),
            "--min-episodes",
            str(args.halving_min_episodes),
            "--max-episodes",
            str(args.halving_max_episodes),
            "--seeds",
            str(seed),
            "--out-root",
            str(tuned_halving_out),
            "--gpu",
            str(args.gpu),
            "--random-state",
            str(args.halving_random_state),
        ]
        rc = _run(halving_cmd, flowpolicy_pkg, env, tuned_dir / "halving.log", args.dry_run)
        if rc != 0:
            return rc
        train_best_cmd = [
            py,
            str(here / "train_best_from_halving.py"),
            "--best-json",
            str(tuned_halving_out / "best_trial.json"),
            "--seed",
            str(seed),
            "--gpu",
            str(args.gpu),
            "--run-dir",
            str(tuned_train_run),
        ]
        rc = _run(train_best_cmd, flowpolicy_pkg, env, tuned_dir / "train_best.log", args.dry_run)
        if rc != 0:
            return rc
        tuned_ckpt = tuned_train_run / "checkpoints" / "latest.ckpt"

        # tuned_pre (halving preprocess -> train best preprocess)
        tuned_pre_dir = seed_dir / "tuned_pre"
        tuned_pre_halving_out = tuned_pre_dir / "halving"
        tuned_pre_train_run = tuned_pre_dir / "train_run"
        tuned_pre_dir.mkdir(parents=True, exist_ok=True)
        halving_pre_cmd = [
            py,
            str(here / "halving_search_kitchen.py"),
            "--n-candidates",
            str(args.halving_n_candidates),
            "--factor",
            str(args.halving_factor),
            "--min-episodes",
            str(args.halving_min_episodes),
            "--max-episodes",
            str(args.halving_max_episodes),
            "--seeds",
            str(seed),
            "--out-root",
            str(tuned_pre_halving_out),
            "--gpu",
            str(args.gpu),
            "--random-state",
            str(args.halving_random_state),
            "--preprocess",
        ]
        rc = _run(halving_pre_cmd, flowpolicy_pkg, env, tuned_pre_dir / "halving.log", args.dry_run)
        if rc != 0:
            return rc
        train_best_pre_cmd = [
            py,
            str(here / "train_best_from_halving.py"),
            "--best-json",
            str(tuned_pre_halving_out / "best_trial.json"),
            "--seed",
            str(seed),
            "--gpu",
            str(args.gpu),
            "--run-dir",
            str(tuned_pre_train_run),
            "--preprocess",
        ]
        rc = _run(train_best_pre_cmd, flowpolicy_pkg, env, tuned_pre_dir / "train_best.log", args.dry_run)
        if rc != 0:
            return rc
        tuned_pre_ckpt = tuned_pre_train_run / "checkpoints" / "latest.ckpt"

        arm_rows: List[Dict[str, Any]] = []
        for arm, ckpt in [
            ("baseline", baseline_ckpt),
            ("baseline_pre", baseline_pre_ckpt),
            ("tuned", tuned_ckpt),
            ("tuned_pre", tuned_pre_ckpt),
        ]:
            if not args.dry_run and not ckpt.is_file():
                print(f"[exp] ERROR ckpt tidak ditemukan: {ckpt}", file=sys.stderr)
                return 1
            eval_row = _eval_model(
                py=py,
                flowpolicy_pkg=flowpolicy_pkg,
                env=env,
                ckpt_path=ckpt,
                episodes=args.eval_episodes,
                log_path=seed_dir / arm / "eval.log",
                dry_run=args.dry_run,
            )
            row = {
                "seed": seed,
                "arm": arm,
                "run_dir": str(ckpt.parent.parent),
                "ckpt": str(ckpt),
                **eval_row,
            }
            arm_rows.append(row)
            all_rows.append(row)

        if not args.dry_run:
            valid = [
                r
                for r in arm_rows
                if r.get("eval_rc") == 0 and math.isfinite(float(r.get("eval_test_mean_score", float("nan"))))
            ]
            if not valid:
                print(f"[exp] ERROR tidak ada eval sukses untuk seed {seed}", file=sys.stderr)
                return 1
            winner = max(valid, key=lambda r: float(r["eval_test_mean_score"]))
            seed_winners[str(seed)] = winner

    all_csv = out_root / "all_models_eval.csv"
    if all_rows:
        keys: List[str] = sorted({k for r in all_rows for k in r.keys()})
        with open(all_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"[exp] all models csv -> {all_csv}")

    if args.dry_run:
        print("[exp] dry-run selesai.")
        return 0

    with open(out_root / "seed_winners.json", "w", encoding="utf-8") as f:
        json.dump(seed_winners, f, indent=2, default=str)

    winners = list(seed_winners.values())
    global_winner = max(winners, key=lambda r: float(r["eval_test_mean_score"]))
    with open(out_root / "global_winner.json", "w", encoding="utf-8") as f:
        json.dump(global_winner, f, indent=2, default=str)

    # final inference for global winner
    win_ckpt = pathlib.Path(global_winner["ckpt"])
    final_out = out_root / "global_winner_inference"
    final_out.mkdir(parents=True, exist_ok=True)
    final_cmd = [
        py,
        "infer_kitchen.py",
        "--checkpoint",
        str(win_ckpt),
        "--episodes",
        str(args.hero_episodes),
        "--device",
        "cuda:0",
        "--output-dir",
        str(final_out),
    ]
    if not args.hero_video:
        final_cmd.append("--no-video")
    rc = _run(final_cmd, flowpolicy_pkg, env, master_log, dry_run=False)
    if rc != 0:
        return rc
    final_metrics = final_out / "metrics.json"
    summary: Dict[str, Any] = {"global_winner": global_winner, "final_metrics_path": str(final_metrics)}
    if final_metrics.is_file():
        m = _read_metrics(final_metrics)
        summary["final_test_mean_score"] = float(m.get("test_mean_score", float("nan")))
        summary["final_mean_latency_ms"] = float(m.get("mean_time", float("nan"))) * 1000.0
    with open(out_root / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[exp] seed winners -> {out_root / 'seed_winners.json'}")
    print(f"[exp] global winner -> {out_root / 'global_winner.json'}")
    print(f"[exp] final summary -> {out_root / 'final_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
