#!/usr/bin/env python3
"""Orkestrasi eksperimen Franka Kitchen: 3 seed × (tanpa preprocess | preprocess).

Alur:
  1) Jalankan Bayesian optimization (``bayes_opt_kitchen.py``) **6 kali** —
     untuk setiap seed suite ``S``, sekali tanpa ``--preprocess`` dan sekali
     dengan ``--preprocess``. Objektif BO memakai **hanya seed ``S``** pada
     tiap trial (satu pelatihan per trial, selaras dengan "3 seed berbeda
     × 2 mode").
  2) Baca ``best_trial.json`` tiap arm, pilih **checkpoint pemenang lokal**.
  3) Inferensi **tanpa video** pada setiap pemenang (episode sama) → SR +
     latensi ke ``suite_eval.csv``.
  4) Pilih **model global terbaik** menurut ``test_mean_score`` eval itu.
  5) Inferensi **dengan video** hanya untuk pemenang global → folder
     ``<suite-root>/hero_best/``.

Prasyarat: sama seperti ``bayes_opt_kitchen.py`` (``optuna``, dsb.).

Contoh:
  cd FlowPolicy
  python scripts/bo_franka_kitchen_suite.py --dry-run
  bash scripts/bo_franka_kitchen_suite.sh 0 --n-trials 20
"""
from __future__ import annotations

import argparse
import csv
import math
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def _flowpolicy_inner() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    inner = here.parent / "FlowPolicy"
    if not inner.is_dir():
        raise SystemExit(f"Tidak menemukan {inner}")
    return inner


def _suite_arm_tag(seed: int, preprocess: bool) -> str:
    return f"seed{seed}_{'pre' if preprocess else 'nopre'}"


def _study_name(tag: str) -> str:
    return f"frk_bo_{tag}"


def _run_cmd(
        cmd: List[str],
        cwd: pathlib.Path,
        env: Dict[str, str],
        log_path: pathlib.Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def _read_best_trial(out_root: pathlib.Path) -> Optional[Dict[str, Any]]:
    p = out_root / "best_trial.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _ckpt_from_best(out_root: pathlib.Path, train_seed: int,
                    best: Dict[str, Any]) -> pathlib.Path:
    n = int(best["number"])
    return (out_root / f"trial_{n:05d}_seed{train_seed}" / "checkpoints"
            / "latest.ckpt").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite BO Franka Kitchen: 3 seed × 2 preprocess")
    parser.add_argument(
        "--suite-root",
        type=str,
        default="data/outputs/bo_franka_suite",
        help="Relatif ke FlowPolicy/ (folder induk 6 sub-run BO).")
    parser.add_argument(
        "--suite-seeds", type=int, nargs="+", default=[0, 42, 101],
        help="Tiga seed eksperimen (masing-masing × preprocess on/off).")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--bo-episodes", type=int, default=50,
        help="Episode infer mini di dalam BO (per trial).")
    parser.add_argument(
        "--eval-episodes", type=int, default=50,
        help="Episode infer eval setelah BO (tanpa video) untuk semua pemenang arm.")
    parser.add_argument(
        "--hero-episodes", type=int, default=20,
        help="Episode infer **dengan video** untuk model global terbaik.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-bo",
        action="store_true",
        help="Lewati fase BO; jalankan agregasi + eval + hero (sub-run harus sudah ada).")
    parser.add_argument("--no-mem-cap", action="store_true")
    args = parser.parse_args()

    inner = _flowpolicy_inner()
    script_py = pathlib.Path(__file__).resolve()
    bo_py = script_py.parent / "bayes_opt_kitchen.py"
    suite_root = (inner / args.suite_root).resolve()
    hero_dir = suite_root / "hero_best"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    py = sys.executable

    arms: List[Tuple[str, pathlib.Path, int, bool]] = []
    for s in args.suite_seeds:
        for pre in (False, True):
            tag = _suite_arm_tag(s, pre)
            arms.append((tag, suite_root / tag, s, pre))

    if args.dry_run:
        print("[suite] dry-run — 6 perintah BO yang akan dijalankan:")
        for tag, out, s, pre in arms:
            study = _study_name(tag)
            cmd = [
                py, str(bo_py),
                "--n-trials", str(args.n_trials),
                "--seeds", str(s),
                "--episodes", str(args.bo_episodes),
                "--out-root", f"{args.suite_root}/{tag}",
                "--study-name", study,
                "--gpu", str(args.gpu),
            ]
            if pre:
                cmd.append("--preprocess")
            if args.no_mem_cap:
                cmd.append("--no-mem-cap")
            print(" ", " ".join(cmd))
        print(f"[suite] lalu eval (infer_kitchen --no-video) per arm, lalu hero (dengan video) di {hero_dir}")
        return 0

    if not args.skip_bo:
        suite_root.mkdir(parents=True, exist_ok=True)
        for tag, out, s, pre in arms:
            study = _study_name(tag)
            cmd = [
                py, str(bo_py),
                "--n-trials", str(args.n_trials),
                "--seeds", str(s),
                "--episodes", str(args.bo_episodes),
                "--out-root", f"{args.suite_root}/{tag}",
                "--study-name", study,
                "--gpu", str(args.gpu),
            ]
            if pre:
                cmd.append("--preprocess")
            if args.no_mem_cap:
                cmd.append("--no-mem-cap")
            logp = out / "suite_bo_subprocess.log"
            print(f"\n[suite] === BO arm {tag} ===\n[suite] log: {logp}")
            rc = _run_cmd(cmd, inner, env, logp)
            if rc != 0:
                print(f"[suite] ERROR arm {tag} rc={rc}", file=sys.stderr)
                return rc

    # Agregasi best per arm
    rows: List[Dict[str, Any]] = []
    for tag, out, s, pre in arms:
        best = _read_best_trial(out)
        if not best:
            print(f"[suite] WARN: tidak ada best_trial.json di {out}", file=sys.stderr)
            continue
        ckpt = _ckpt_from_best(out, s, best)
        rows.append({
            "tag": tag,
            "suite_seed": s,
            "preprocess": pre,
            "bo_best_value": best.get("value"),
            "trial_number": best.get("number"),
            "ckpt": str(ckpt),
            "ckpt_exists": ckpt.is_file(),
        })

    agg_path = suite_root / "suite_arms.json"
    suite_root.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[suite] ringkasan arm -> {agg_path}")
    if not rows:
        print("[suite] tidak ada arm dengan best_trial.json; henti.", file=sys.stderr)
        return 1

    # Eval infer (no video) semua pemenang arm
    eval_rows: List[Dict[str, Any]] = []
    for r in rows:
        ckpt = pathlib.Path(r["ckpt"])
        if not ckpt.is_file():
            print(f"[suite] skip eval — tidak ada ckpt: {ckpt}", file=sys.stderr)
            continue
        out_sub = "inference_suite_eval"
        out_eval = ckpt.parent.parent / out_sub
        cmd = [
            py, "infer_kitchen.py",
            "--checkpoint", str(ckpt),
            "--episodes", str(args.eval_episodes),
            "--device", "cuda:0",
            "--output-subdir", out_sub,
            "--no-video",
        ]
        logp = ckpt.parent.parent / "infer_suite_eval.log"
        print(f"[suite] eval {r['tag']} -> {out_eval}")
        rc = _run_cmd(cmd, inner, env, logp)
        metrics_p = out_eval / "metrics.json"
        sr, lat_ms = float("nan"), float("nan")
        if rc == 0 and metrics_p.is_file():
            with open(metrics_p, "r", encoding="utf-8") as f:
                m = json.load(f)
            sr = float(m.get("test_mean_score", float("nan")))
            lat_ms = float(m.get("mean_time", 0.0)) * 1000.0
        eval_rows.append({
            **r,
            "eval_rc": rc,
            "eval_test_mean_score": sr,
            "eval_mean_latency_ms": lat_ms,
        })

    eval_csv = suite_root / "suite_eval.csv"
    if eval_rows:
        keys = list(eval_rows[0].keys())
        with open(eval_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for er in eval_rows:
                w.writerow(er)
        print(f"[suite] eval semua arm -> {eval_csv}")

    # Pemenang global (by eval SR)
    valid = [
        e for e in eval_rows
        if e.get("eval_rc") == 0
        and math.isfinite(float(e.get("eval_test_mean_score", float("nan"))))
    ]
    if not valid:
        print("[suite] tidak ada eval sukses; hero dilewati.", file=sys.stderr)
        return 1
    winner = max(valid, key=lambda e: e.get("eval_test_mean_score", float("-inf")))
    win_path = pathlib.Path(winner["ckpt"])
    hero_dir.mkdir(parents=True, exist_ok=True)
    with open(suite_root / "suite_winner.json", "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2, default=str)
    print(f"\n[suite] PEMENANG GLOBAL: {winner['tag']}  SR={winner['eval_test_mean_score']:.4f}  "
          f"lat={winner['eval_mean_latency_ms']:.1f} ms")
    print(f"[suite] ckpt: {win_path}")

    cmd_hero = [
        py, "infer_kitchen.py",
        "--checkpoint", str(win_path),
        "--episodes", str(args.hero_episodes),
        "--device", "cuda:0",
        "--output-dir", str(hero_dir),
    ]
    log_hero = suite_root / "hero_infer.log"
    print(f"\n[suite] inferensi HERO (dengan video) -> {hero_dir}")
    rc = _run_cmd(cmd_hero, inner, env, log_hero)
    if rc != 0:
        print(f"[suite] hero infer rc={rc}", file=sys.stderr)
        return rc
    print(f"[suite] selesai. Video: {hero_dir / 'videos'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
