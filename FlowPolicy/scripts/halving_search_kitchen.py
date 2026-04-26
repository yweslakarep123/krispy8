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
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

TRIAL_CSV_FIELDNAMES = [
    "trial_number",
    "value",
    "elapsed_s",
    "episodes",
    "config_json",
]

SEARCH_SPACE_BASE: Dict[str, List[Any]] = {
    "training.num_epochs": [500, 1000, 3000, 5000],
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-2, 1e-3, 1e-4, 0.05],
    "policy.Conditional_ConsistencyFM.delta": [1e-2, 1e-3, 1e-4, 0.05],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [4, 6, 8, 16],
}

SEARCH_SPACE_MODEL: Dict[str, List[Any]] = {
    "policy.down_dims": [
        [128, 256, 512],
        [192, 384, 768],
        [256, 512, 1024],
        [384, 768, 1536],
    ],
    "policy.diffusion_step_embed_dim": [64, 128, 256, 512],
    "policy.kernel_size": [3, 5, 7, 9],
    "policy.n_groups": [1, 2, 4, 8],
    "policy.encoder_output_dim": [32, 64, 128, 256],
    "policy.state_mlp_size": [
        [128, 128],
        [256, 256],
        [512, 512],
        [256, 256, 256],
    ],
    "policy.action_clip": [0.5, 1.0, 2.0, 5.0],
    "policy.encoder_use_layernorm": [True, False],
    "policy.use_down_condition": [True, False],
    "policy.use_mid_condition": [True, False],
    "policy.use_up_condition": [True, False],
}

DEFAULT_SEEDS = [0, 42, 101]


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


def _n_groups_compatible(n_groups: int, down_dims: List[int]) -> bool:
    return all(d % n_groups == 0 for d in down_dims)


def is_cfg_valid(cfg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    n_groups = int(cfg.get("policy.n_groups", 8))
    down_dims = list(cfg.get("policy.down_dims", [256, 512, 1024]))
    if not _n_groups_compatible(n_groups, down_dims):
        return False, f"n_groups={n_groups} tidak kompatibel dengan down_dims={down_dims}"
    eps_c = float(cfg.get("policy.Conditional_ConsistencyFM.eps", 1e-2))
    if not (1e-9 < eps_c < 1.0):
        return False, f"eps={eps_c} harus di rentang (0,1)"
    dlt = float(cfg.get("policy.Conditional_ConsistencyFM.delta", 1e-2))
    if not (1e-9 < dlt < 1.0):
        return False, f"delta={dlt} harus di rentang (0,1)"
    return True, None


def _dataloader_batch_cap_for_vram(down_dims: Sequence[int]) -> int:
    mx = int(max(down_dims))
    if mx >= 1536:
        return 64
    if mx >= 1024:
        return 128
    if mx >= 768:
        return 128
    if mx >= 512:
        return 256
    return 512


def _apply_vram_safe_batch(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    out = dict(cfg)
    dd = list(out.get("policy.down_dims", [256, 512, 1024]))
    cap = _dataloader_batch_cap_for_vram(dd)
    req = int(out.get("dataloader.batch_size", 128))
    if req > cap:
        out["dataloader.batch_size"] = cap
        return out, f"dataloader.batch_size {req} -> {cap} (down_dims max {max(dd)})"
    return out, None


def run_subprocess(cmd: List[str], log_path: pathlib.Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def read_metrics(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FlowPolicyEstimator(BaseEstimator):
    trial_counter = 0

    def __init__(
        self,
        training_num_epochs=1000,
        optimizer_lr=1e-4,
        dataloader_batch_size=128,
        policy_num_segments=2,
        policy_eps=1e-2,
        policy_delta=1e-2,
        n_action_steps=4,
        n_obs_steps=4,
        policy_down_dims=None,
        policy_diffusion_step_embed_dim=128,
        policy_kernel_size=5,
        policy_n_groups=8,
        policy_encoder_output_dim=64,
        policy_state_mlp_size=None,
        policy_action_clip=1.0,
        policy_encoder_use_layernorm=True,
        policy_use_down_condition=True,
        policy_use_mid_condition=True,
        policy_use_up_condition=True,
        episodes=50,
    ):
        self.training_num_epochs = training_num_epochs
        self.optimizer_lr = optimizer_lr
        self.dataloader_batch_size = dataloader_batch_size
        self.policy_num_segments = policy_num_segments
        self.policy_eps = policy_eps
        self.policy_delta = policy_delta
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.policy_down_dims = policy_down_dims or [256, 512, 1024]
        self.policy_diffusion_step_embed_dim = policy_diffusion_step_embed_dim
        self.policy_kernel_size = policy_kernel_size
        self.policy_n_groups = policy_n_groups
        self.policy_encoder_output_dim = policy_encoder_output_dim
        self.policy_state_mlp_size = policy_state_mlp_size or [256, 256]
        self.policy_action_clip = policy_action_clip
        self.policy_encoder_use_layernorm = policy_encoder_use_layernorm
        self.policy_use_down_condition = policy_use_down_condition
        self.policy_use_mid_condition = policy_use_mid_condition
        self.policy_use_up_condition = policy_use_up_condition
        self.episodes = episodes

    def fit(self, X, y=None):  # noqa: D401, ANN001
        cfg: Dict[str, Any] = {
            "training.num_epochs": int(self.training_num_epochs),
            "optimizer.lr": float(self.optimizer_lr),
            "dataloader.batch_size": int(self.dataloader_batch_size),
            "policy.Conditional_ConsistencyFM.num_segments": int(self.policy_num_segments),
            "policy.Conditional_ConsistencyFM.eps": float(self.policy_eps),
            "policy.Conditional_ConsistencyFM.delta": float(self.policy_delta),
            "n_action_steps": int(self.n_action_steps),
            "n_obs_steps": int(self.n_obs_steps),
            "policy.down_dims": list(self.policy_down_dims),
            "policy.diffusion_step_embed_dim": int(self.policy_diffusion_step_embed_dim),
            "policy.kernel_size": int(self.policy_kernel_size),
            "policy.n_groups": int(self.policy_n_groups),
            "policy.encoder_output_dim": int(self.policy_encoder_output_dim),
            "policy.state_mlp_size": list(self.policy_state_mlp_size),
            "policy.action_clip": float(self.policy_action_clip),
            "policy.encoder_use_layernorm": bool(self.policy_encoder_use_layernorm),
            "policy.use_down_condition": bool(self.policy_use_down_condition),
            "policy.use_mid_condition": bool(self.policy_use_mid_condition),
            "policy.use_up_condition": bool(self.policy_use_up_condition),
        }

        valid, reason = is_cfg_valid(cfg)
        trial_id = FlowPolicyEstimator.trial_counter
        FlowPolicyEstimator.trial_counter += 1
        _pr(f"[halving] trial {trial_id} start episodes={self.episodes}", APP.progress_log)
        if not valid:
            _pr(f"[halving] trial {trial_id} invalid: {reason}", APP.progress_log)
            self._score = 0.0
            _append_trial_row(trial_id, self._score, self.episodes, cfg, APP.t0)
            return self

        train_cfg = cfg
        if not APP.no_mem_cap:
            train_cfg, note = _apply_vram_safe_batch(cfg)
            if note:
                _pr(f"  [mem-cap] {note}", APP.progress_log)

        scores: List[float] = []
        for seed in APP.seeds:
            run_dir = APP.out_root / f"trial_{trial_id:05d}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
            infer_subdir = f"inference_ep{int(self.episodes)}"
            metrics_path = run_dir / infer_subdir / "metrics.json"

            train_cmd = [
                sys.executable,
                "train.py",
                "--config-name=flowpolicy",
                "task=kitchen_complete",
                f"hydra.run.dir={run_dir}",
                f"training.seed={seed}",
                "training.device=cuda",
                "training.debug=False",
                "exp_name=halving_search",
                f"logging.name=halving_{run_dir.name}",
                "logging.mode=offline",
                "checkpoint.save_ckpt=True",
                "training.resume=False",
            ]
            for k, v in train_cfg.items():
                train_cmd.append(format_override(k, v))
                if k == "dataloader.batch_size":
                    train_cmd.append(format_override("val_dataloader.batch_size", v))
            if APP.preprocess_overrides:
                for k, v in APP.preprocess_overrides.items():
                    train_cmd.append(format_override(f"task.dataset.preprocess.{k}", v))

            _pr(f"[halving] trial {trial_id} seed {seed} train...", APP.progress_log)
            rc = run_subprocess(train_cmd, run_dir / "train_stdout.log", APP.env)
            if rc != 0 or not ckpt_path.is_file():
                _pr(f"[halving] trial {trial_id} seed {seed} TRAIN FAIL rc={rc}", APP.progress_log)
                scores.append(0.0)
                continue

            infer_cmd = [
                sys.executable,
                "infer_kitchen.py",
                "--checkpoint",
                str(ckpt_path),
                "--episodes",
                str(int(self.episodes)),
                "--device",
                "cuda:0",
                "--output-subdir",
                infer_subdir,
                "--no-video",
            ]
            rc = run_subprocess(infer_cmd, run_dir / "infer_stdout.log", APP.env)
            if rc != 0 or not metrics_path.is_file():
                _pr(f"[halving] trial {trial_id} seed {seed} INFER FAIL rc={rc}", APP.progress_log)
                scores.append(0.0)
                continue
            try:
                metrics = read_metrics(metrics_path)
                scores.append(float(metrics.get("test_mean_score", 0.0)))
            except Exception:
                scores.append(0.0)

        self._score = float(sum(scores) / len(scores)) if scores else 0.0
        _pr(f"[halving] trial {trial_id} mean SR={self._score:.4f}", APP.progress_log)
        _append_trial_row(trial_id, self._score, self.episodes, cfg, APP.t0)
        return self

    def score(self, X, y=None):  # noqa: D401, ANN001
        return float(getattr(self, "_score", 0.0))


class _AppCtx:
    out_root: pathlib.Path
    progress_log: pathlib.Path
    trials_csv: pathlib.Path
    best_json: pathlib.Path
    env: Dict[str, str]
    seeds: List[int]
    no_mem_cap: bool
    t0: float
    preprocess_overrides: Optional[Dict[str, Any]]


APP = _AppCtx()


def _pr(msg: str, progress_log: pathlib.Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(progress_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_trial_row(trial_n: int, score: float, episodes: int, cfg: Dict[str, Any], t0: float) -> None:
    row = {
        "trial_number": trial_n,
        "value": score,
        "elapsed_s": round(time.time() - t0, 2),
        "episodes": int(episodes),
        "config_json": json.dumps(cfg, default=str),
    }
    with open(APP.trials_csv, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=TRIAL_CSV_FIELDNAMES).writerow(row)


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

    APP.out_root = (flowpolicy_dir / args.out_root).resolve()
    APP.out_root.mkdir(parents=True, exist_ok=True)
    APP.progress_log = APP.out_root / "progress.log"
    APP.trials_csv = APP.out_root / "trials.csv"
    APP.best_json = APP.out_root / "best_trial.json"
    APP.seeds = args.seeds
    APP.no_mem_cap = args.no_mem_cap
    APP.t0 = time.time()
    APP.env = os.environ.copy()
    APP.env["CUDA_VISIBLE_DEVICES"] = args.gpu
    APP.env["HYDRA_FULL_ERROR"] = "1"
    APP.env["PYTHONUNBUFFERED"] = "1"
    APP.preprocess_overrides = None

    if args.preprocess:
        APP.preprocess_overrides = {"enabled": True}
        if args.preprocess_window_ratio is not None:
            APP.preprocess_overrides["window_ratio"] = args.preprocess_window_ratio
        if args.preprocess_stride is not None:
            APP.preprocess_overrides["stride"] = args.preprocess_stride
        if args.preprocess_train_ratio is not None:
            APP.preprocess_overrides["train_ratio"] = args.preprocess_train_ratio
        if args.preprocess_val_ratio is not None:
            APP.preprocess_overrides["val_ratio"] = args.preprocess_val_ratio
        if args.preprocess_test_ratio is not None:
            APP.preprocess_overrides["test_ratio"] = args.preprocess_test_ratio
        if args.preprocess_split_seed is not None:
            APP.preprocess_overrides["split_seed"] = args.preprocess_split_seed

    if not APP.trials_csv.is_file():
        with open(APP.trials_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=TRIAL_CSV_FIELDNAMES).writeheader()

    if args.dry_run:
        _pr("[halving] dry-run: konfigurasi search akan dijalankan tanpa train/infer", APP.progress_log)
        _pr(f"  out-root={APP.out_root}", APP.progress_log)
        _pr(f"  n-candidates={args.n_candidates}, factor={args.factor}", APP.progress_log)
        _pr(f"  episodes resource={args.min_episodes}..{args.max_episodes}", APP.progress_log)
        return 0

    FlowPolicyEstimator.trial_counter = 0
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

    # Dummy data: score estimator tidak memakai X/y.
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

    _pr(
        f"[halving] mulai search n_candidates={args.n_candidates} factor={args.factor} "
        f"episodes={args.min_episodes}..{args.max_episodes}",
        APP.progress_log,
    )
    search.fit(X, y)
    _pr("[halving] search selesai", APP.progress_log)

    best = {
        "score": float(search.best_score_),
        "params": search.best_params_,
        "n_iterations": int(search.n_iterations_),
        "n_candidates": list(map(int, search.n_candidates_)),
        "n_resources": list(map(int, search.n_resources_)),
    }
    with open(APP.best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, default=str)
    _pr(f"[halving] best score={best['score']:.4f}", APP.progress_log)
    _pr(f"[halving] best_trial.json -> {APP.best_json}", APP.progress_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
