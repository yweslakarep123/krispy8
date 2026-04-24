#!/usr/bin/env python3
"""Orkestrasi eksperimen Franka Kitchen: 3 seed × (tanpa preprocess | preprocess).

Alur:
  1) Jalankan Bayesian optimization (``bayes_opt_kitchen.py``) **6 kali** —
     untuk setiap seed suite ``S``, sekali tanpa ``--preprocess`` dan sekali
     dengan ``--preprocess``.
  2) Baca ``best_trial.json`` tiap arm, pilih **checkpoint pemenang lokal**.
  3) Inferensi **tanpa video** pada setiap pemenang → SR + latensi
     ke ``suite_eval.csv``.
  4) Pilih **model global terbaik** menurut ``test_mean_score`` eval.
  5) Inferensi **dengan video** hanya untuk pemenang global → ``hero_best/``.

Catatan:
  - Mode default BO tetap sekuensial (kompatibel lama).
  - Aktifkan ``--parallel-bo`` + ``--gpu-pool 0,1,2,...`` agar arm BO
    berjalan paralel di beberapa GPU (ideal untuk Vast.ai multi-GPU).
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
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, env=env
        )
        return proc.wait()


def _spawn_cmd(
    cmd: List[str],
    cwd: pathlib.Path,
    env: Dict[str, str],
    log_path: pathlib.Path,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")
    logf.write(f"# cmd: {' '.join(cmd)}\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, env=env
    )
    setattr(proc, "_suite_logf", logf)
    return proc


def _close_spawn_log(proc: subprocess.Popen) -> None:
    try:
        logf = getattr(proc, "_suite_logf")
        logf.close()
    except Exception:
        pass


def _read_best_trial(out_root: pathlib.Path) -> Optional[Dict[str, Any]]:
    p = out_root / "best_trial.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _ckpt_from_best(out_root: pathlib.Path, train_seed: int, best: Dict[str, Any]) -> pathlib.Path:
    n = int(best["number"])
    return (out_root / f"trial_{n:05d}_seed{train_seed}" / "checkpoints" / "latest.ckpt").resolve()


def _parse_gpu_pool(args_gpu: str, gpu_pool_arg: str) -> List[str]:
    raw = gpu_pool_arg.strip() if gpu_pool_arg else args_gpu.strip()
    gpus = [x.strip() for x in raw.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("GPU pool kosong. Gunakan --gpu 0 atau --gpu-pool 0,1,2,3")
    return gpus


def _build_bo_cmd(
    py: str,
    bo_py: pathlib.Path,
    n_trials: int,
    seed: int,
    episodes: int,
    out_root: str,
    study_name: str,
    gpu_id: str,
    preprocess: bool,
    no_mem_cap: bool,
) -> List[str]:
    cmd = [
        py,
        str(bo_py),
        "--n-trials",
        str(n_trials),
        "--seeds",
        str(seed),
        "--episodes",
        str(episodes),
        "--out-root",
        out_root,
        "--study-name",
        study_name,
        "--gpu",
        gpu_id,
    ]
    if preprocess:
        cmd.append("--preprocess")
    if no_mem_cap:
        cmd.append("--no-mem-cap")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Suite BO Franka Kitchen: 3 seed × 2 preprocess")
    parser.add_argument(
        "--suite-root",
        type=str,
        default="data/outputs/bo_franka_suite",
        help="Relatif ke FlowPolicy/ (folder induk 6 sub-run BO).",
    )
    parser.add_argument(
        "--suite-seeds",
        type=int,
        nargs="+",
        default=[0, 42, 101],
        help="Tiga seed eksperimen (masing-masing × preprocess on/off).",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--bo-episodes", type=int, default=50, help="Episode infer mini di dalam BO (per trial)."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Episode infer eval setelah BO (tanpa video) untuk semua pemenang arm.",
    )
    parser.add_argument(
        "--hero-episodes", type=int, default=20, help="Episode infer dengan video untuk model global terbaik."
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU tunggal (mode sekuensial).")
    parser.add_argument(
        "--gpu-pool",
        type=str,
        default="",
        help="Daftar GPU dipisah koma untuk BO paralel, contoh: 0,1,2,3",
    )
    parser.add_argument(
        "--parallel-bo",
        action="store_true",
        help="Jalankan arm BO paralel; GPU dialokasikan round-robin dari --gpu-pool.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-bo",
        action="store_true",
        help="Lewati fase BO; jalankan agregasi + eval + hero (sub-run harus sudah ada).",
    )
    parser.add_argument("--no-mem-cap", action="store_true")
    args = parser.parse_args()

    inner = _flowpolicy_inner()
    script_py = pathlib.Path(__file__).resolve()
    bo_py = script_py.parent / "bayes_opt_kitchen.py"
    suite_root = (inner / args.suite_root).resolve()
    hero_dir = suite_root / "hero_best"
    gpu_pool = _parse_gpu_pool(args.gpu, args.gpu_pool)
    eval_gpu = gpu_pool[0]

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    py = sys.executable

    arms: List[Tuple[str, pathlib.Path, int, bool]] = []
    for s in args.suite_seeds:
        for pre in (False, True):
            tag = _suite_arm_tag(s, pre)
            arms.append((tag, suite_root / tag, s, pre))

    if args.dry_run:
        mode = "PARALEL" if args.parallel_bo else "SEKUENSIAL"
        print(f"[suite] dry-run — mode BO: {mode}; gpu-pool={gpu_pool}")
        print("[suite] perintah BO yang akan dijalankan:")
        for i, (tag, _out, s, pre) in enumerate(arms):
            gpu_for_arm = gpu_pool[i % len(gpu_pool)]
            study = _study_name(tag)
            cmd = _build_bo_cmd(
                py=py,
                bo_py=bo_py,
                n_trials=args.n_trials,
                seed=s,
                episodes=args.bo_episodes,
                out_root=f"{args.suite_root}/{tag}",
                study_name=study,
                gpu_id=gpu_for_arm,
                preprocess=pre,
                no_mem_cap=args.no_mem_cap,
            )
            print(f"  [gpu {gpu_for_arm}] " + " ".join(cmd))
        print(
            f"[suite] lalu eval (infer_kitchen --no-video) per arm, lalu hero (dengan video) di {hero_dir}"
        )
        return 0

    if not args.skip_bo:
        suite_root.mkdir(parents=True, exist_ok=True)
        if args.parallel_bo:
            print(f"[suite] BO paralel aktif. gpu-pool={gpu_pool}")
            running: List[Tuple[str, pathlib.Path, subprocess.Popen]] = []
            for i, (tag, out, s, pre) in enumerate(arms):
                gpu_for_arm = gpu_pool[i % len(gpu_pool)]
                study = _study_name(tag)
                cmd = _build_bo_cmd(
                    py=py,
                    bo_py=bo_py,
                    n_trials=args.n_trials,
                    seed=s,
                    episodes=args.bo_episodes,
                    out_root=f"{args.suite_root}/{tag}",
                    study_name=study,
                    gpu_id=gpu_for_arm,
                    preprocess=pre,
                    no_mem_cap=args.no_mem_cap,
                )
                logp = out / "suite_bo_subprocess.log"
                arm_env = env.copy()
                arm_env["CUDA_VISIBLE_DEVICES"] = gpu_for_arm
                proc = _spawn_cmd(cmd, inner, arm_env, logp)
                print(f"[suite] launch arm {tag} on gpu {gpu_for_arm} pid={proc.pid} log={logp}")
                running.append((tag, logp, proc))

            failed: List[Tuple[str, int, pathlib.Path]] = []
            while running:
                pending: List[Tuple[str, pathlib.Path, subprocess.Popen]] = []
                for tag, logp, proc in running:
                    rc = proc.poll()
                    if rc is None:
                        pending.append((tag, logp, proc))
                        continue
                    _close_spawn_log(proc)
                    print(f"[suite] selesai arm {tag} rc={rc} log={logp}")
                    if rc != 0:
                        failed.append((tag, rc, logp))
                if pending:
                    time.sleep(5.0)
                running = pending
            if failed:
                for tag, rc, logp in failed:
                    print(f"[suite] ERROR arm {tag} rc={rc} log={logp}", file=sys.stderr)
                return 1
        else:
            run_gpu = gpu_pool[0]
            print(f"[suite] BO sekuensial aktif. gpu={run_gpu}")
            for tag, out, s, pre in arms:
                study = _study_name(tag)
                cmd = _build_bo_cmd(
                    py=py,
                    bo_py=bo_py,
                    n_trials=args.n_trials,
                    seed=s,
                    episodes=args.bo_episodes,
                    out_root=f"{args.suite_root}/{tag}",
                    study_name=study,
                    gpu_id=run_gpu,
                    preprocess=pre,
                    no_mem_cap=args.no_mem_cap,
                )
                logp = out / "suite_bo_subprocess.log"
                run_env = env.copy()
                run_env["CUDA_VISIBLE_DEVICES"] = run_gpu
                print(f"\n[suite] === BO arm {tag} ===\n[suite] log: {logp}")
                rc = _run_cmd(cmd, inner, run_env, logp)
                if rc != 0:
                    print(f"[suite] ERROR arm {tag} rc={rc}", file=sys.stderr)
                    return rc

    rows: List[Dict[str, Any]] = []
    for tag, out, s, pre in arms:
        best = _read_best_trial(out)
        if not best:
            print(f"[suite] WARN: tidak ada best_trial.json di {out}", file=sys.stderr)
            continue
        ckpt = _ckpt_from_best(out, s, best)
        rows.append(
            {
                "tag": tag,
                "suite_seed": s,
                "preprocess": pre,
                "bo_best_value": best.get("value"),
                "trial_number": best.get("number"),
                "ckpt": str(ckpt),
                "ckpt_exists": ckpt.is_file(),
            }
        )

    agg_path = suite_root / "suite_arms.json"
    suite_root.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[suite] ringkasan arm -> {agg_path}")
    if not rows:
        print("[suite] tidak ada arm dengan best_trial.json; henti.", file=sys.stderr)
        return 1

    eval_rows: List[Dict[str, Any]] = []
    for r in rows:
        ckpt = pathlib.Path(r["ckpt"])
        if not ckpt.is_file():
            print(f"[suite] skip eval — tidak ada ckpt: {ckpt}", file=sys.stderr)
            continue
        out_sub = "inference_suite_eval"
        out_eval = ckpt.parent.parent / out_sub
        cmd = [
            py,
            "infer_kitchen.py",
            "--checkpoint",
            str(ckpt),
            "--episodes",
            str(args.eval_episodes),
            "--device",
            "cuda:0",
            "--output-subdir",
            out_sub,
            "--no-video",
        ]
        logp = ckpt.parent.parent / "infer_suite_eval.log"
        print(f"[suite] eval {r['tag']} -> {out_eval}")
        eval_env = env.copy()
        eval_env["CUDA_VISIBLE_DEVICES"] = eval_gpu
        rc = _run_cmd(cmd, inner, eval_env, logp)
        metrics_p = out_eval / "metrics.json"
        sr, lat_ms = float("nan"), float("nan")
        if rc == 0 and metrics_p.is_file():
            with open(metrics_p, "r", encoding="utf-8") as f:
                m = json.load(f)
            sr = float(m.get("test_mean_score", float("nan")))
            lat_ms = float(m.get("mean_time", 0.0)) * 1000.0
        eval_rows.append(
            {
                **r,
                "eval_rc": rc,
                "eval_test_mean_score": sr,
                "eval_mean_latency_ms": lat_ms,
            }
        )

    eval_csv = suite_root / "suite_eval.csv"
    if eval_rows:
        keys = list(eval_rows[0].keys())
        with open(eval_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for er in eval_rows:
                w.writerow(er)
        print(f"[suite] eval semua arm -> {eval_csv}")

    valid = [
        e
        for e in eval_rows
        if e.get("eval_rc") == 0 and math.isfinite(float(e.get("eval_test_mean_score", float("nan"))))
    ]
    if not valid:
        print("[suite] tidak ada eval sukses; hero dilewati.", file=sys.stderr)
        return 1
    winner = max(valid, key=lambda e: e.get("eval_test_mean_score", float("-inf")))
    win_path = pathlib.Path(winner["ckpt"])
    hero_dir.mkdir(parents=True, exist_ok=True)
    with open(suite_root / "suite_winner.json", "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2, default=str)
    print(
        f"\n[suite] PEMENANG GLOBAL: {winner['tag']}  SR={winner['eval_test_mean_score']:.4f}  "
        f"lat={winner['eval_mean_latency_ms']:.1f} ms"
    )
    print(f"[suite] ckpt: {win_path}")

    cmd_hero = [
        py,
        "infer_kitchen.py",
        "--checkpoint",
        str(win_path),
        "--episodes",
        str(args.hero_episodes),
        "--device",
        "cuda:0",
        "--output-dir",
        str(hero_dir),
    ]
    log_hero = suite_root / "hero_infer.log"
    print(f"\n[suite] inferensi HERO (dengan video) -> {hero_dir}")
    hero_env = env.copy()
    hero_env["CUDA_VISIBLE_DEVICES"] = eval_gpu
    rc = _run_cmd(cmd_hero, inner, hero_env, log_hero)
    if rc != 0:
        print(f"[suite] hero infer rc={rc}", file=sys.stderr)
        return rc
    print(f"[suite] selesai. Video: {hero_dir / 'videos'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
