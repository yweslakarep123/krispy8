#!/usr/bin/env python3
"""Orkestrasi 3 seed x 2 scenario untuk FlowPolicy kitchen.

Per seed:
1) tuned_no_preprocess : random search tuning tanpa preprocess -> train final best params
2) tuned_preprocess    : random search tuning dengan preprocess -> train final best params
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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
    cmd.append("--strict-gpu")
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
    p = argparse.ArgumentParser(description="Eksperimen 3 seed x 2 scenario")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 101])
    p.add_argument("--out-root", type=str, default="data/outputs/exp_3seed_4arms")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--hero-episodes", type=int, default=20)
    p.add_argument("--hero-video", action="store_true", help="Aktifkan video untuk inferensi final juara global.")
    p.add_argument("--random-n-iter", type=int, default=100)
    p.add_argument("--random-cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument(
        "--strict-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail jika CUDA tidak tersedia.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _progress(step: int, total: int, label: str) -> None:
    pct = (100.0 * step / total) if total > 0 else 0.0
    print(f"[progress] {step}/{total} ({pct:.1f}%) - {label}")


def _plot_all_models(all_rows: List[Dict[str, Any]], plots_dir: pathlib.Path) -> Optional[pathlib.Path]:
    if plt is None or not all_rows:
        return None
    by_scenario: Dict[str, List[float]] = {}
    for r in all_rows:
        scenario = str(r.get("scenario", "unknown"))
        sr = float(r.get("eval_test_mean_score", float("nan")))
        if math.isfinite(sr):
            by_scenario.setdefault(scenario, []).append(sr)
    if not by_scenario:
        return None

    scenarios = sorted(by_scenario.keys())
    means = [sum(by_scenario[s]) / len(by_scenario[s]) for s in scenarios]
    counts = [len(by_scenario[s]) for s in scenarios]

    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "success_rate_all_models.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(scenarios, means)
    ax.set_title("Success Rate Semua Scenario (rata-rata lintas seed)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Success Rate (test_mean_score)")
    ax.set_ylim(0.0, max(1.0, max(means) * 1.1))
    for b, m, c in zip(bars, means, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{m:.3f}\n(n={c})", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_global_winner(summary: Dict[str, Any], plots_dir: pathlib.Path) -> Optional[pathlib.Path]:
    if plt is None:
        return None
    sr = float(summary.get("final_test_mean_score", float("nan")))
    lat = float(summary.get("final_mean_latency_ms", float("nan")))
    if not (math.isfinite(sr) and math.isfinite(lat)):
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "global_winner_sr_latency.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(["global_winner"], [sr])
    ax1.set_title("Success Rate")
    ax1.set_ylim(0.0, max(1.0, sr * 1.1))
    ax1.set_ylabel("test_mean_score")
    ax1.text(0, sr, f"{sr:.3f}", ha="center", va="bottom")
    ax2.bar(["global_winner"], [lat])
    ax2.set_title("Latency Inferensi")
    ax2.set_ylabel("ms per predict_action")
    ax2.text(0, lat, f"{lat:.2f} ms", ha="center", va="bottom")
    fig.suptitle("Model Terbaik (dipakai inferensi/video)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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
    env["FLOWP_GPU_PERF_MODE"] = "1"

    if args.strict_gpu:
        _check_gpu_or_fail(py, env, args.gpu, args.dry_run)

    all_rows: List[Dict[str, Any]] = []
    seed_winners: Dict[str, Any] = {}
    total_steps = len(args.seeds) * 6 + 4
    step = 0

    for seed_idx, seed in enumerate(args.seeds, start=1):
        seed_dir = out_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[exp] ===== seed {seed_idx}/{len(args.seeds)}: {seed} =====")

        # scenario: tuned_no_preprocess
        no_pre_dir = seed_dir / "tuned_no_preprocess"
        no_pre_random_out = no_pre_dir / "random_search"
        no_pre_train_run = no_pre_dir / "train_run"
        no_pre_dir.mkdir(parents=True, exist_ok=True)
        step += 1
        _progress(step, total_steps, f"seed{seed} scenario=tuned_no_preprocess random_search")
        random_no_pre_cmd = [
            py,
            str(here / "random_search_kitchen.py"),
            "--seed",
            str(seed),
            "--n-iter",
            str(args.random_n_iter),
            "--cv",
            str(args.random_cv),
            "--out-root",
            str(no_pre_random_out),
            "--gpu",
            str(args.gpu),
            "--random-state",
            str(args.random_state),
            "--scenario-name",
            "tuned_no_preprocess",
        ]
        if args.dry_run:
            random_no_pre_cmd.append("--dry-run")
        rc = _run(random_no_pre_cmd, flowpolicy_pkg, env, no_pre_dir / "random_search.log", args.dry_run)
        if rc != 0:
            return rc

        step += 1
        _progress(step, total_steps, f"seed{seed} scenario=tuned_no_preprocess train_best")
        train_best_cmd = [
            py,
            str(here / "train_best_from_random.py"),
            "--best-json",
            str(no_pre_random_out / "best_trial.json"),
            "--seed",
            str(seed),
            "--gpu",
            str(args.gpu),
            "--run-dir",
            str(no_pre_train_run),
            "--scenario-name",
            "tuned_no_preprocess",
        ]
        rc = _run(train_best_cmd, flowpolicy_pkg, env, no_pre_dir / "train_best.log", args.dry_run)
        if rc != 0:
            return rc
        no_pre_ckpt = no_pre_train_run / "checkpoints" / "latest.ckpt"

        # scenario: tuned_preprocess
        pre_dir = seed_dir / "tuned_preprocess"
        pre_random_out = pre_dir / "random_search"
        pre_train_run = pre_dir / "train_run"
        pre_dir.mkdir(parents=True, exist_ok=True)
        step += 1
        _progress(step, total_steps, f"seed{seed} scenario=tuned_preprocess random_search")
        random_pre_cmd = [
            py,
            str(here / "random_search_kitchen.py"),
            "--seed",
            str(seed),
            "--n-iter",
            str(args.random_n_iter),
            "--cv",
            str(args.random_cv),
            "--out-root",
            str(pre_random_out),
            "--gpu",
            str(args.gpu),
            "--random-state",
            str(args.random_state),
            "--preprocess",
            "--scenario-name",
            "tuned_preprocess",
        ]
        if args.dry_run:
            random_pre_cmd.append("--dry-run")
        rc = _run(random_pre_cmd, flowpolicy_pkg, env, pre_dir / "random_search.log", args.dry_run)
        if rc != 0:
            return rc

        step += 1
        _progress(step, total_steps, f"seed{seed} scenario=tuned_preprocess train_best")
        train_best_pre_cmd = [
            py,
            str(here / "train_best_from_random.py"),
            "--best-json",
            str(pre_random_out / "best_trial.json"),
            "--seed",
            str(seed),
            "--gpu",
            str(args.gpu),
            "--run-dir",
            str(pre_train_run),
            "--preprocess",
            "--scenario-name",
            "tuned_preprocess",
        ]
        rc = _run(train_best_pre_cmd, flowpolicy_pkg, env, pre_dir / "train_best.log", args.dry_run)
        if rc != 0:
            return rc
        pre_ckpt = pre_train_run / "checkpoints" / "latest.ckpt"

        scenario_rows: List[Dict[str, Any]] = []
        for scenario, ckpt in [
            ("tuned_no_preprocess", no_pre_ckpt),
            ("tuned_preprocess", pre_ckpt),
        ]:
            step += 1
            _progress(step, total_steps, f"seed{seed} scenario={scenario} infer_eval")
            if not args.dry_run and not ckpt.is_file():
                print(f"[exp] ERROR ckpt tidak ditemukan: {ckpt}", file=sys.stderr)
                return 1
            eval_row = _eval_model(
                py=py,
                flowpolicy_pkg=flowpolicy_pkg,
                env=env,
                ckpt_path=ckpt,
                episodes=args.eval_episodes,
                log_path=seed_dir / scenario / "eval.log",
                dry_run=args.dry_run,
            )
            row = {
                "seed": seed,
                "scenario": scenario,
                "run_dir": str(ckpt.parent.parent),
                "ckpt": str(ckpt),
                **eval_row,
            }
            scenario_rows.append(row)
            all_rows.append(row)

        if not args.dry_run:
            valid = [
                r
                for r in scenario_rows
                if r.get("eval_rc") == 0 and math.isfinite(float(r.get("eval_test_mean_score", float("nan"))))
            ]
            if not valid:
                print(f"[exp] ERROR tidak ada eval sukses untuk seed {seed}", file=sys.stderr)
                return 1
            winner = max(valid, key=lambda r: float(r["eval_test_mean_score"]))
            seed_winners[str(seed)] = winner

    all_csv = out_root / "all_models_eval.csv"
    if all_rows:
        step += 1
        _progress(step, total_steps, "menyimpan all_models_eval.csv")
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
    step += 1
    _progress(step, total_steps, "inferensi final global_winner")
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
        "--strict-gpu",
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

    plots_dir = out_root / "plots"
    step += 1
    _progress(step, total_steps, "generate plot")
    all_plot = _plot_all_models(all_rows, plots_dir)
    winner_plot = _plot_global_winner(summary, plots_dir)
    if all_plot:
        summary["success_rate_plot"] = str(all_plot)
    if winner_plot:
        summary["winner_sr_latency_plot"] = str(winner_plot)
    with open(out_root / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_root / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "success_rate_plot": str(all_plot) if all_plot else None,
                "winner_sr_latency_plot": str(winner_plot) if winner_plot else None,
            },
            f,
            indent=2,
            default=str,
        )

    step += 1
    _progress(step, total_steps, "selesai")
    print(f"[exp] seed winners -> {out_root / 'seed_winners.json'}")
    print(f"[exp] global winner -> {out_root / 'global_winner.json'}")
    print(f"[exp] final summary -> {out_root / 'final_summary.json'}")
    if all_plot:
        print(f"[exp] plot all models -> {all_plot}")
    if winner_plot:
        print(f"[exp] plot winner -> {winner_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
