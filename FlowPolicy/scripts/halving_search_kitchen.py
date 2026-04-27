#!/usr/bin/env python3
"""Hyperparameter tuning FlowPolicy dengan HalvingRandomSearchCV.

Pipeline evaluasi tiap kandidat:
1) train.py dengan override hyperparameter
2) infer_kitchen.py untuk menghitung test_mean_score
3) skor final = rata-rata test_mean_score lintas seed
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from hparam_search_common import (
    DEFAULT_SEEDS,
    FlowPolicyEstimator,
    HalvingAppCtx,
    SEARCH_SPACE_BASE,
    SEARCH_SPACE_MODEL,
    TRIAL_CSV_FIELDNAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HalvingRandomSearchCV untuk FlowPolicy kitchen")
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--min-episodes", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out-root", type=str, default="data/outputs/halving_search")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-mem-cap", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--preprocess-window-ratio", type=float, default=None)
    parser.add_argument("--preprocess-stride", type=int, default=None)
    parser.add_argument("--preprocess-train-ratio", type=float, default=None)
    parser.add_argument("--preprocess-val-ratio", type=float, default=None)
    parser.add_argument("--preprocess-test-ratio", type=float, default=None)
    parser.add_argument("--preprocess-split-seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.min_episodes < 1 or args.max_episodes < args.min_episodes:
        raise SystemExit("Invalid range episodes: pastikan 1 <= min-episodes <= max-episodes")

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    if not flowpolicy_dir.is_dir():
        raise SystemExit(f"Tidak menemukan {flowpolicy_dir}")
    os.chdir(flowpolicy_dir)

    out_root = (flowpolicy_dir / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = __import__("time").time()
    ctx = HalvingAppCtx(
        out_root=out_root,
        progress_log=out_root / "progress.log",
        trials_csv=out_root / "trials.csv",
        best_json=out_root / "best_trial.json",
        env=os.environ.copy(),
        seeds=args.seeds,
        no_mem_cap=args.no_mem_cap,
        t0=t0,
        preprocess_overrides=None,
    )
    ctx.env["CUDA_VISIBLE_DEVICES"] = args.gpu
    ctx.env["HYDRA_FULL_ERROR"] = "1"
    ctx.env["PYTHONUNBUFFERED"] = "1"

    if args.preprocess:
        ctx.preprocess_overrides = {"enabled": True}
        if args.preprocess_window_ratio is not None:
            ctx.preprocess_overrides["window_ratio"] = args.preprocess_window_ratio
        if args.preprocess_stride is not None:
            ctx.preprocess_overrides["stride"] = args.preprocess_stride
        if args.preprocess_train_ratio is not None:
            ctx.preprocess_overrides["train_ratio"] = args.preprocess_train_ratio
        if args.preprocess_val_ratio is not None:
            ctx.preprocess_overrides["val_ratio"] = args.preprocess_val_ratio
        if args.preprocess_test_ratio is not None:
            ctx.preprocess_overrides["test_ratio"] = args.preprocess_test_ratio
        if args.preprocess_split_seed is not None:
            ctx.preprocess_overrides["split_seed"] = args.preprocess_split_seed

    if not ctx.trials_csv.is_file():
        with open(ctx.trials_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=TRIAL_CSV_FIELDNAMES).writeheader()

    from hparam_search_common import log_line

    if args.dry_run:
        log_line("[halving] dry-run: konfigurasi search akan dijalankan tanpa train/infer", ctx.progress_log)
        log_line(f"  out-root={ctx.out_root}", ctx.progress_log)
        log_line(f"  n-candidates={args.n_candidates}, factor={args.factor}", ctx.progress_log)
        log_line(f"  episodes resource={args.min_episodes}..{args.max_episodes}", ctx.progress_log)
        return 0

    FlowPolicyEstimator.trial_counter = 0
    FlowPolicyEstimator.app_ctx = ctx
    estimator = FlowPolicyEstimator()

    param_distributions = {
        "training_num_epochs": SEARCH_SPACE_BASE["training.num_epochs"],
        "optimizer_lr": SEARCH_SPACE_BASE["optimizer.lr"],
        "dataloader_batch_size": SEARCH_SPACE_BASE["dataloader.batch_size"],
        "policy_num_segments": SEARCH_SPACE_BASE["policy.Conditional_ConsistencyFM.num_segments"],
        "policy_eps": SEARCH_SPACE_BASE["policy.Conditional_ConsistencyFM.eps"],
        "policy_delta": SEARCH_SPACE_BASE["policy.Conditional_ConsistencyFM.delta"],
        "n_action_steps": SEARCH_SPACE_BASE["n_action_steps"],
        "n_obs_steps": SEARCH_SPACE_BASE["n_obs_steps"],
        "policy_down_dims": SEARCH_SPACE_MODEL["policy.down_dims"],
        "policy_diffusion_step_embed_dim": SEARCH_SPACE_MODEL["policy.diffusion_step_embed_dim"],
        "policy_kernel_size": SEARCH_SPACE_MODEL["policy.kernel_size"],
        "policy_n_groups": SEARCH_SPACE_MODEL["policy.n_groups"],
        "policy_encoder_output_dim": SEARCH_SPACE_MODEL["policy.encoder_output_dim"],
        "policy_state_mlp_size": SEARCH_SPACE_MODEL["policy.state_mlp_size"],
        "policy_action_clip": SEARCH_SPACE_MODEL["policy.action_clip"],
        "policy_encoder_use_layernorm": SEARCH_SPACE_MODEL["policy.encoder_use_layernorm"],
        "policy_use_down_condition": SEARCH_SPACE_MODEL["policy.use_down_condition"],
        "policy_use_mid_condition": SEARCH_SPACE_MODEL["policy.use_mid_condition"],
        "policy_use_up_condition": SEARCH_SPACE_MODEL["policy.use_up_condition"],
    }

    X = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2,), dtype=np.float32)
    cv = [(np.array([0]), np.array([1]))]

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_candidates=args.n_candidates,
        factor=args.factor,
        resource="episodes",
        min_resources=args.min_episodes,
        max_resources=args.max_episodes,
        cv=cv,
        refit=False,
        random_state=args.random_state,
        n_jobs=1,
        verbose=1,
        error_score=0.0,
        return_train_score=False,
    )

    log_line(
        f"[halving] mulai search n_candidates={args.n_candidates} factor={args.factor} "
        f"episodes={args.min_episodes}..{args.max_episodes}",
        ctx.progress_log,
    )
    search.fit(X, y)
    log_line("[halving] search selesai", ctx.progress_log)

    best = {
        "score": float(search.best_score_),
        "params": search.best_params_,
        "n_iterations": int(search.n_iterations_),
        "n_candidates": list(map(int, search.n_candidates_)),
        "n_resources": list(map(int, search.n_resources_)),
    }
    with open(ctx.best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, default=str)
    log_line(f"[halving] best score={best['score']:.4f}", ctx.progress_log)
    log_line(f"[halving] best_trial.json -> {ctx.best_json}", ctx.progress_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
