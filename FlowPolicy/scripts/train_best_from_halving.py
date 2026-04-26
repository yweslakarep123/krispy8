#!/usr/bin/env python3
"""Jalankan training final dari best_trial.json hasil Halving search."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict


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


def map_best_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "training.num_epochs": int(best_params["training_num_epochs"]),
        "optimizer.lr": float(best_params["optimizer_lr"]),
        "dataloader.batch_size": int(best_params["dataloader_batch_size"]),
        "policy.Conditional_ConsistencyFM.num_segments": int(best_params["policy_num_segments"]),
        "policy.Conditional_ConsistencyFM.eps": float(best_params["policy_eps"]),
        "policy.Conditional_ConsistencyFM.delta": float(best_params["policy_delta"]),
        "n_action_steps": int(best_params["n_action_steps"]),
        "n_obs_steps": int(best_params["n_obs_steps"]),
        "policy.down_dims": list(best_params["policy_down_dims"]),
        "policy.diffusion_step_embed_dim": int(best_params["policy_diffusion_step_embed_dim"]),
        "policy.kernel_size": int(best_params["policy_kernel_size"]),
        "policy.n_groups": int(best_params["policy_n_groups"]),
        "policy.encoder_output_dim": int(best_params["policy_encoder_output_dim"]),
        "policy.state_mlp_size": list(best_params["policy_state_mlp_size"]),
        "policy.action_clip": float(best_params["policy_action_clip"]),
        "policy.encoder_use_layernorm": bool(best_params["policy_encoder_use_layernorm"]),
        "policy.use_down_condition": bool(best_params["policy_use_down_condition"]),
        "policy.use_mid_condition": bool(best_params["policy_use_mid_condition"]),
        "policy.use_up_condition": bool(best_params["policy_use_up_condition"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train final model dari best_trial Halving")
    ap.add_argument("--best-json", type=str, default="data/outputs/halving_search/best_trial.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--save-ckpt", type=str, default="True")
    ap.add_argument("--preprocess", action="store_true")
    args = ap.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    os.chdir(flowpolicy_dir)

    best_path = (flowpolicy_dir / args.best_json).resolve()
    if not best_path.is_file():
        raise SystemExit(f"Tidak menemukan best_trial.json: {best_path}")

    with open(best_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    best_params = payload.get("params", {})
    if not best_params:
        raise SystemExit("best_trial.json tidak berisi field params")

    cfg = map_best_params(best_params)
    run_dir = args.run_dir or f"data/outputs/kitchen_complete-flowpolicy_halving_best_seed{args.seed}"

    cmd = [
        sys.executable,
        "train.py",
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={args.seed}",
        "training.device=cuda",
        "training.debug=False",
        "exp_name=kitchen_complete-flowpolicy_halving_best",
        f"logging.name=halving_best_seed{args.seed}",
        "logging.mode=online",
        f"checkpoint.save_ckpt={args.save_ckpt}",
        "training.resume=False",
    ]
    for k, v in cfg.items():
        cmd.append(format_override(k, v))
        if k == "dataloader.batch_size":
            cmd.append(format_override("val_dataloader.batch_size", v))
    if args.preprocess:
        cmd.append("task.dataset.preprocess.enabled=true")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print("[train-best] cmd:")
    print(" ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
