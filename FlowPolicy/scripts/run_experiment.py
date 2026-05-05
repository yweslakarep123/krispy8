#!/usr/bin/env python3
"""Orkestrator eksperimen: seed × preprocessing × random configs × CV fold.

Setiap invokasi train+infer memakai folder run unik ``runs/<sel>__<timestamp>_<uuid8>/``,
``training.resume=false``, dan ``logging.resume=false`` + ``logging.id`` unik agar tidak
terhubung ke run atau checkpoint lain.

Menulis ``configs.json``, ``cv_splits.json``, menjalankan ``train.py`` +
``infer_kitchen.py`` per sel, lalu ``summarize.py`` + ``plot_results.py``.

Profil ``standard`` / ``minimal``: lipatan CV memakai indeks episode penuh
(tanpa sliding-window 70/20/10). Windowing + 70/20/10 hanya untuk training
tanpa ``train_episode_indices`` / ``val_episode_indices`` (mis. ``train_kitchen.sh``).

Contoh:
  cd FlowPolicy
  python scripts/run_experiment.py --gpu 0 --dry-run
  python scripts/run_experiment.py --gpu 0 --output-dir data/outputs/experiment_demo \\
      --n-configs 2 --n-folds 2 --seeds 0 --profiles standard --skip-plots
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
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

# Impor util CV dari skrip sebelah
_SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from cv_splits import build_cv_splits  # noqa: E402

FLOW_REPO = _SCRIPTS.parent
FLOW_PKG = FLOW_REPO / "FlowPolicy"

SEARCH_SPACE: Dict[str, List[Any]] = {
    "training.num_epochs": [500, 1000, 3000, 5000],
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 1.0],
    "policy.Conditional_ConsistencyFM.delta": [1e-4, 1e-3, 1e-2, 1.0],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [4, 6, 8, 16],
    "policy.diffusion_step_embed_dim": [128, 256, 512, 1024],
    "_state_mlp_hidden": [128, 256, 512, 1024],
}


def sample_configs(n: int, sampling_seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(sampling_seed)
    return [{k: rng.choice(v) for k, v in SEARCH_SPACE.items()} for _ in range(n)]


def format_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, float):
        return f"{key}={value:.6g}"
    return f"{key}={value}"


def fmt_episode_list(key: str, indices: Sequence[int]) -> str:
    return f"{key}=[{','.join(str(int(i)) for i in indices)}]"


def _expand_policy_arch(cfg: Dict[str, Any]) -> List[str]:
    extra: List[str] = []
    h = cfg.get("_state_mlp_hidden")
    if h is not None:
        hi = int(h)
        extra.append(f"policy.state_mlp_size=[{hi},{hi}]")
    return extra


def build_train_cmd(
    cfg: Dict[str, Any],
    seed: int,
    run_dir: pathlib.Path,
    profile: str,
    train_episodes: Sequence[int],
    val_episodes: Sequence[int],
) -> List[str]:
    # Setiap subproses: tidak resume ckpt Hydra, tidak resume run WandB, id run unik.
    wandb_run_id = uuid.uuid4().hex
    overrides = [
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={seed}",
        "training.device=cuda",
        "training.debug=False",
        "training.resume=false",
        "exp_name=experiment_cv",
        f"logging.name=exp_{run_dir.name}",
        "logging.mode=offline",
        "logging.resume=false",
        f"logging.id={wandb_run_id}",
        "checkpoint.save_ckpt=True",
        f"task.dataset.preprocessing_profile={profile}",
        fmt_episode_list("task.dataset.train_episode_indices", train_episodes),
        fmt_episode_list("task.dataset.val_episode_indices", val_episodes),
    ]
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        overrides.append(format_override(k, v))
        if k == "dataloader.batch_size":
            overrides.append(format_override("val_dataloader.batch_size", v))
    overrides.extend(_expand_policy_arch(cfg))
    # -u: stdout/stderr tak di-buffer agar tqdm terlihat saat output di-tee ke terminal
    return [sys.executable, "-u", "train.py", *overrides]


def build_infer_cmd(
    ckpt_path: pathlib.Path,
    episodes: int,
    output_subdir: str,
    device: str = "cuda:0",
) -> List[str]:
    return [
        sys.executable,
        "infer_kitchen.py",
        "--checkpoint",
        str(ckpt_path),
        "--episodes",
        str(episodes),
        "--device",
        device,
        "--output-subdir",
        output_subdir,
    ]


def metrics_path(run_dir: pathlib.Path, infer_subdir: str) -> pathlib.Path:
    return run_dir / infer_subdir / "metrics.json"


def _metrics_has_test_score(mpath: pathlib.Path) -> bool:
    if not mpath.is_file():
        return False
    try:
        mj = read_json(mpath)
        return mj.get("test_mean_score") is not None
    except Exception:
        return False


def cell_already_completed(
    runs_root: pathlib.Path,
    cell_name: str,
    infer_subdir: str,
) -> bool:
    """True jika folder lama ``cell_name`` atau ``cell_name__*`` sudah punya metrik inferensi."""
    legacy = runs_root / cell_name
    if legacy.is_dir() and _metrics_has_test_score(metrics_path(legacy, infer_subdir)):
        return True
    for p in runs_root.glob(f"{cell_name}__*"):
        if p.is_dir() and _metrics_has_test_score(metrics_path(p, infer_subdir)):
            return True
    return False


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_results_row(
    results_csv: pathlib.Path,
    row: Dict[str, Any],
) -> None:
    write_header = not results_csv.exists()
    fieldnames = list(row.keys())
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_subprocess(cmd: List[str], log_path: pathlib.Path, env: dict) -> int:
    """Jalankan subproses, tulis log ke ``log_path`` dan salin ke stdout (tee)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        cmd,
        cwd=str(FLOW_PKG),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    def _tee() -> None:
        assert p.stdout is not None
        with open(log_path, "wb") as lf:
            while True:
                chunk = p.stdout.read(8192)
                if not chunk:
                    break
                lf.write(chunk)
                lf.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

    th = threading.Thread(target=_tee)
    th.start()
    rc = int(p.wait())
    th.join(timeout=300)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="Eksperimen seed × profile × CV × HP")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 101])
    p.add_argument(
        "--profiles",
        nargs="+",
        default=["standard", "minimal"],
        choices=("standard", "minimal", "legacy_minimal", "raw"),
    )
    p.add_argument("--n-configs", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-episodes", type=int, default=19)
    p.add_argument("--n-test", type=int, default=1)
    p.add_argument("--sampling-seed", type=int, default=99)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--test-seed", type=int, default=0)
    p.add_argument("--n-infer-episodes", type=int, default=50)
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs/experiment",
        help="Relatif terhadap FlowPolicy/FlowPolicy/",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-summary", action="store_true")
    args = p.parse_args()

    assert FLOW_PKG.is_dir(), f"Tidak ditemukan: {FLOW_PKG}"

    out_root = (FLOW_PKG / args.output_dir).resolve()
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    configs_path = out_root / "configs.json"
    cv_path = out_root / "cv_splits.json"
    results_csv = out_root / "results.csv"

    configs = sample_configs(args.n_configs, args.sampling_seed)
    configs_path.write_text(json.dumps(configs, indent=2), encoding="utf-8")

    cv_data = build_cv_splits(
        n_episodes=args.n_episodes,
        n_folds=args.n_folds,
        n_test=args.n_test,
        cv_seed=args.cv_seed,
        test_seed=args.test_seed,
    )
    cv_path.write_text(json.dumps(cv_data, indent=2), encoding="utf-8")

    infer_subdir = f"infer_ep{args.n_infer_episodes}"

    if args.dry_run:
        print(f"[dry-run] output: {out_root}")
        print(f"  configs: {len(configs)}")
        print(f"  folds: {len(cv_data['folds'])}")
        print(f"  seeds: {args.seeds}")
        print(f"  profiles: {args.profiles}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"

    device_infer = "cuda:0"

    for ci, hp_cfg in enumerate(configs):
        for profile in args.profiles:
            for fold_entry in cv_data["folds"]:
                fold = int(fold_entry["fold"])
                train_ep = fold_entry["train_episodes"]
                val_ep = fold_entry["val_episodes"]
                for seed in args.seeds:
                    name = f"cfg{ci:02d}_seed{seed}_{profile}_fold{fold}"
                    run_uid = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
                    run_slug = f"{name}__{run_uid}"
                    run_dir = (runs_root / run_slug).resolve()
                    run_dir.mkdir(parents=True, exist_ok=True)
                    mpath = metrics_path(run_dir, infer_subdir)
                    if cell_already_completed(runs_root, name, infer_subdir):
                        print(f"[skip] {name} (inferensi sudah selesai di folder run sebelumnya)")
                        continue

                    print(f"\n=== train {run_slug} ===")
                    t0 = time.time()
                    ckpt = run_dir / "checkpoints" / "latest.ckpt"
                    # #region agent log
                    try:
                        with open(
                            "/home/daffa/Documents/krispy8/.cursor/debug-3a4aa7.log",
                            "a",
                            encoding="utf-8",
                        ) as _df:
                            _df.write(
                                json.dumps(
                                    {
                                        "sessionId": "3a4aa7",
                                        "runId": "pre-fix",
                                        "hypothesisId": "H3",
                                        "location": "run_experiment.py:before_train",
                                        "message": "train cell",
                                        "timestamp": int(time.time() * 1000),
                                        "data": {
                                            "run_name": name,
                                            "checkpoint_exists": ckpt.is_file(),
                                        },
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion
                    cmd = build_train_cmd(
                        hp_cfg, seed, run_dir, profile, train_ep, val_ep,
                    )
                    rc = run_subprocess(cmd, run_dir / "train_stdout.log", env)
                    # #region agent log
                    try:
                        with open(
                            "/home/daffa/Documents/krispy8/.cursor/debug-3a4aa7.log",
                            "a",
                            encoding="utf-8",
                        ) as _df:
                            _df.write(
                                json.dumps(
                                    {
                                        "sessionId": "3a4aa7",
                                        "runId": "pre-fix",
                                        "hypothesisId": "H4",
                                        "location": "run_experiment.py:after_train_subprocess",
                                        "message": "train subprocess finished",
                                        "timestamp": int(time.time() * 1000),
                                        "data": {
                                            "run_name": name,
                                            "run_slug": run_slug,
                                            "returncode": int(rc),
                                            "elapsed_sec": round(time.time() - t0, 3),
                                            "ckpt_written": ckpt.is_file(),
                                        },
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion
                    if rc != 0 or not ckpt.is_file():
                        row = _result_row(
                            ci, seed, profile, fold, hp_cfg,
                            args.n_infer_episodes,
                            status=f"train_failed_{rc}",
                            t_train=time.time() - t0,
                            run_folder=run_slug,
                        )
                        append_results_row(results_csv, row)
                        continue
                    t_train = time.time() - t0

                    print(f"=== infer {run_slug} ===")
                    t1 = time.time()
                    icmd = build_infer_cmd(
                        ckpt, args.n_infer_episodes, infer_subdir, device_infer,
                    )
                    rc = run_subprocess(icmd, run_dir / "infer_stdout.log", env)
                    t_infer = time.time() - t1

                    metrics: Dict[str, Any] = {}
                    if mpath.is_file():
                        try:
                            metrics = read_json(mpath)
                        except Exception:
                            pass
                    train_end = run_dir / "training_end_metrics.json"
                    tlf = vlf = ""
                    if train_end.is_file():
                        try:
                            te = read_json(train_end)
                            tlf = te.get("train_loss_final", "")
                            vlf = te.get("val_loss_final", "")
                        except Exception:
                            pass

                    row = _result_row(
                        ci,
                        seed,
                        profile,
                        fold,
                        hp_cfg,
                        args.n_infer_episodes,
                        metrics=metrics,
                        t_train=t_train,
                        t_infer=t_infer,
                        status="ok" if rc == 0 and metrics else f"infer_failed_{rc}",
                        train_loss_final=tlf,
                        val_loss_final=vlf,
                        ckpt=str(ckpt),
                        run_folder=run_slug,
                    )
                    append_results_row(results_csv, row)

    if not args.skip_summary:
        summ = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "summarize.py"),
                "--output-dir",
                str(out_root),
            ],
            cwd=str(FLOW_REPO),
        )
        if summ.returncode != 0:
            print("[warn] summarize.py gagal")

    if not args.skip_plots and not args.skip_summary:
        pl = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "plot_results.py"),
                "--output-dir",
                str(out_root),
            ],
            cwd=str(FLOW_REPO),
        )
        if pl.returncode != 0:
            print("[warn] plot_results.py gagal (matplotlib hilang?)")

    print(f"\n[done] {out_root}")
    return 0


def _result_row(
    cfg_idx: int,
    seed: int,
    profile: str,
    fold: int,
    hp_cfg: Dict[str, Any],
    n_infer_episodes: int,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "",
    t_train: float = 0.0,
    t_infer: float = 0.0,
    train_loss_final: Any = "",
    val_loss_final: Any = "",
    ckpt: str = "",
    run_folder: str = "",
) -> Dict[str, Any]:
    metrics = metrics or {}
    row: Dict[str, Any] = {
        "cfg_idx": cfg_idx,
        "seed": seed,
        "profile": profile,
        "fold": fold,
        "run_folder": run_folder,
    }
    for k in SEARCH_SPACE:
        row[k] = hp_cfg.get(k)
    row["success_rate_k1"] = metrics.get("success_rate_k1", "")
    row["success_rate_k2"] = metrics.get("success_rate_k2", "")
    row["success_rate_k3"] = metrics.get("success_rate_k3", "")
    row["success_rate_k4"] = metrics.get("success_rate_k4", "")
    row["mean_inference_latency_ms"] = metrics.get("mean_inference_latency_ms", "")
    row["std_inference_latency_ms"] = metrics.get("std_inference_latency_ms", "")
    row["trade_off"] = metrics.get("trade_off", "")
    row["trade_off_k4_per_ms"] = metrics.get("trade_off_k4_per_ms", "")
    row["train_loss_final"] = train_loss_final
    row["val_loss_final"] = val_loss_final
    row["n_infer_episodes"] = int(n_infer_episodes)
    row["checkpoint_path"] = ckpt
    row["status"] = status
    row["t_train_s"] = round(t_train, 2)
    row["t_infer_s"] = round(t_infer, 2)
    row["timestamp"] = int(time.time())
    return row


if __name__ == "__main__":
    os.chdir(FLOW_PKG)
    sys.exit(main())
