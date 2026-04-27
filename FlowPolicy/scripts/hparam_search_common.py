#!/usr/bin/env python3
"""Utilitas bersama untuk Halving/Grid search hyperparameter FlowPolicy (Franka Kitchen)."""
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover - grid search tidak memakai estimators sklearn
    class BaseEstimator:  # type: ignore[misc, no-redef]
        pass

# --- Halving: ruang pencarian asli (sample acak) ---
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

TRIAL_CSV_FIELDNAMES = [
    "trial_number",
    "value",
    "elapsed_s",
    "episodes",
    "config_json",
]


@dataclass
class HalvingAppCtx:
    out_root: pathlib.Path
    progress_log: pathlib.Path
    trials_csv: pathlib.Path
    best_json: pathlib.Path
    env: Dict[str, str]
    seeds: List[int]
    no_mem_cap: bool
    t0: float
    preprocess_overrides: Optional[Dict[str, Any]] = None


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


def apply_vram_safe_batch(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
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


def log_line(msg: str, progress_log: pathlib.Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(progress_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_halving_trial_row(
    ctx: HalvingAppCtx,
    trial_n: int,
    score: float,
    episodes: int,
    cfg: Dict[str, Any],
) -> None:
    row = {
        "trial_number": trial_n,
        "value": score,
        "elapsed_s": round(time.time() - ctx.t0, 2),
        "episodes": int(episodes),
        "config_json": json.dumps(cfg, default=str),
    }
    with open(ctx.trials_csv, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=TRIAL_CSV_FIELDNAMES).writerow(row)


def build_train_cmd(
    train_cfg: Dict[str, Any],
    run_dir: pathlib.Path,
    seed: int,
    exp_name: str,
    logging_name: str,
    preprocess_overrides: Optional[Dict[str, Any]],
) -> List[str]:
    train_cmd: List[str] = [
        sys.executable,
        "train.py",
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={seed}",
        "training.device=cuda",
        "training.debug=False",
        f"exp_name={exp_name}",
        f"logging.name={logging_name}",
        "logging.mode=offline",
        "checkpoint.save_ckpt=True",
        "training.resume=False",
    ]
    for k, v in train_cfg.items():
        train_cmd.append(format_override(k, v))
        if k == "dataloader.batch_size":
            train_cmd.append(format_override("val_dataloader.batch_size", v))
    if preprocess_overrides:
        for k, v in preprocess_overrides.items():
            train_cmd.append(format_override(f"task.dataset.preprocess.{k}", v))
    return train_cmd


def build_infer_cmd(
    ckpt_path: pathlib.Path,
    episodes: int,
    infer_subdir: str,
    no_video: bool = True,
) -> List[str]:
    infer_cmd: List[str] = [
        sys.executable,
        "infer_kitchen.py",
        "--checkpoint",
        str(ckpt_path),
        "--episodes",
        str(int(episodes)),
        "--device",
        "cuda:0",
        "--output-subdir",
        infer_subdir,
    ]
    if no_video:
        infer_cmd.append("--no-video")
    return infer_cmd


def run_one_train_infer(
    train_cfg: Dict[str, Any],
    seed: int,
    run_dir: pathlib.Path,
    episodes: int,
    base_env: Dict[str, str],
    exp_name: str,
    logging_name: str,
    preprocess_overrides: Optional[Dict[str, Any]],
    infer_subdir: str,
) -> Tuple[float, float, int, int]:
    """Train lalu infer satu seed. Return (test_mean_score, mean_time, rc_train, rc_infer)."""
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
    metrics_path = run_dir / infer_subdir / "metrics.json"
    train_cmd = build_train_cmd(
        train_cfg, run_dir, seed, exp_name, logging_name, preprocess_overrides
    )
    rc = run_subprocess(train_cmd, run_dir / "train_stdout.log", base_env)
    if rc != 0 or not ckpt_path.is_file():
        return 0.0, 0.0, rc, -1
    infer_cmd = build_infer_cmd(ckpt_path, episodes, infer_subdir, no_video=True)
    rc_i = run_subprocess(infer_cmd, run_dir / "infer_stdout.log", base_env)
    if rc_i != 0 or not metrics_path.is_file():
        return 0.0, 0.0, rc, rc_i
    try:
        metrics = read_metrics(metrics_path)
        sr = float(metrics.get("test_mean_score", 0.0))
        lat = float(metrics.get("mean_time", 0.0))
        return sr, lat, rc, rc_i
    except Exception:
        return 0.0, 0.0, rc, rc_i


def cfg_from_flowpolicy_estimator(self: Any) -> Dict[str, Any]:
    return {
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


def hydra_cfg_to_estimator_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parameter names yang dipakai `train_best_trial` / `best_trial.json` `params`."""
    return {
        "training_num_epochs": int(cfg["training.num_epochs"]),
        "optimizer_lr": float(cfg["optimizer.lr"]),
        "dataloader_batch_size": int(cfg["dataloader.batch_size"]),
        "policy_num_segments": int(cfg["policy.Conditional_ConsistencyFM.num_segments"]),
        "policy_eps": float(cfg["policy.Conditional_ConsistencyFM.eps"]),
        "policy_delta": float(cfg["policy.Conditional_ConsistencyFM.delta"]),
        "n_action_steps": int(cfg["n_action_steps"]),
        "n_obs_steps": int(cfg["n_obs_steps"]),
        "policy_down_dims": list(cfg["policy.down_dims"]),
        "policy_diffusion_step_embed_dim": int(cfg["policy.diffusion_step_embed_dim"]),
        "policy_kernel_size": int(cfg["policy.kernel_size"]),
        "policy_n_groups": int(cfg["policy.n_groups"]),
        "policy_encoder_output_dim": int(cfg["policy.encoder_output_dim"]),
        "policy_state_mlp_size": list(cfg["policy.state_mlp_size"]),
        "policy_action_clip": float(cfg["policy.action_clip"]),
        "policy_encoder_use_layernorm": bool(cfg["policy.encoder_use_layernorm"]),
        "policy_use_down_condition": bool(cfg["policy.use_down_condition"]),
        "policy_use_mid_condition": bool(cfg["policy.use_mid_condition"]),
        "policy_use_up_condition": bool(cfg["policy.use_up_condition"]),
    }


class FlowPolicyEstimator(BaseEstimator):
    """Kandidat HParam untuk HalvingRandomSearchCV; butuh `app_ctx` (HalvingAppCtx) diset sebelum fit."""

    app_ctx: Optional[HalvingAppCtx] = None
    trial_counter = 0

    def __init__(
        self,
        training_num_epochs: int = 1000,
        optimizer_lr: float = 1e-4,
        dataloader_batch_size: int = 128,
        policy_num_segments: int = 2,
        policy_eps: float = 1e-2,
        policy_delta: float = 1e-2,
        n_action_steps: int = 4,
        n_obs_steps: int = 4,
        policy_down_dims: Optional[List[int]] = None,
        policy_diffusion_step_embed_dim: int = 128,
        policy_kernel_size: int = 5,
        policy_n_groups: int = 8,
        policy_encoder_output_dim: int = 64,
        policy_state_mlp_size: Optional[List[int]] = None,
        policy_action_clip: float = 1.0,
        policy_encoder_use_layernorm: bool = True,
        policy_use_down_condition: bool = True,
        policy_use_mid_condition: bool = True,
        policy_use_up_condition: bool = True,
        episodes: int = 50,
    ) -> None:
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
        ctx = FlowPolicyEstimator.app_ctx
        if ctx is None:
            raise RuntimeError("FlowPolicyEstimator.app_ctx harus diset (HalvingAppCtx)")
        cfg = cfg_from_flowpolicy_estimator(self)
        valid, reason = is_cfg_valid(cfg)
        trial_id = FlowPolicyEstimator.trial_counter
        FlowPolicyEstimator.trial_counter += 1
        log_line(f"[halving] trial {trial_id} start episodes={self.episodes}", ctx.progress_log)
        if not valid:
            log_line(f"[halving] trial {trial_id} invalid: {reason}", ctx.progress_log)
            self._score = 0.0
            append_halving_trial_row(ctx, trial_id, self._score, self.episodes, cfg)
            return self
        train_cfg = cfg
        if not ctx.no_mem_cap:
            train_cfg, note = apply_vram_safe_batch(cfg)
            if note:
                log_line(f"  [mem-cap] {note}", ctx.progress_log)
        scores: List[float] = []
        for seed in ctx.seeds:
            run_dir = ctx.out_root / f"trial_{trial_id:05d}_seed{seed}"
            infer_subdir = f"inference_ep{int(self.episodes)}"
            env = {**ctx.env, "HYDRA_FULL_ERROR": "1", "PYTHONUNBUFFERED": "1"}
            ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
            metrics_path = run_dir / infer_subdir / "metrics.json"
            log_line(f"[halving] trial {trial_id} seed {seed} train...", ctx.progress_log)
            sr, _lat, rc, rci = run_one_train_infer(
                train_cfg,
                seed,
                run_dir,
                int(self.episodes),
                env,
                "halving_search",
                f"halving_{run_dir.name}",
                ctx.preprocess_overrides,
                infer_subdir,
            )
            if rc != 0 or not ckpt_path.is_file():
                log_line(f"[halving] trial {trial_id} seed {seed} TRAIN FAIL rc={rc}", ctx.progress_log)
                scores.append(0.0)
                continue
            if rci != 0 or not metrics_path.is_file():
                log_line(f"[halving] trial {trial_id} seed {seed} INFER FAIL rc={rci}", ctx.progress_log)
                scores.append(0.0)
                continue
            scores.append(sr)
        self._score = float(sum(scores) / len(scores)) if scores else 0.0
        log_line(f"[halving] trial {trial_id} mean SR={self._score:.4f}", ctx.progress_log)
        append_halving_trial_row(ctx, trial_id, self._score, self.episodes, cfg)
        return self

    def score(self, X, y=None):  # noqa: D401, ANN001
        return float(getattr(self, "_score", 0.0))
