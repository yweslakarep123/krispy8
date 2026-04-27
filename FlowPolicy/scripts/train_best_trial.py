#!/usr/bin/env python3
"""Training final dari best_trial.json (Halving / Grid / JSON kompatibel)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict

from hparam_search_common import format_override


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


def _names_for_flavor(flavor: str) -> tuple[str, str]:
    if flavor == "halving":
        return (
            "kitchen_complete-flowpolicy_halving_best",
            "halving_best",
        )
    return (
        "kitchen_complete-flowpolicy_grid_best",
        "grid_best",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train final model dari best_trial.json (params sklearn)",
    )
    ap.add_argument(
        "--best-json",
        type=str,
        default="data/outputs/grid_search_kitchen/best_trial.json",
    )
    ap.add_argument(
        "--flavor",
        choices=("halving", "grid", "auto"),
        default="auto",
        help="auto: pilih dari nama path (halving_search=halving) atau sumber di JSON",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--save-ckpt", type=str, default="True")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--exp-name", type=str, default=None, help="Override penuh exp_name Hydra")
    ap.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Override logging.name; default: {prefix}_seed{seed}",
    )
    ap.add_argument("--log-prefix", type=str, default=None, help="Awar prefix logging.name bila bukan --log-name")
    ap.add_argument(
        "--wandb-mode", type=str, default="online", help="Contoh: online, offline, disabled"
    )
    args = ap.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    os.chdir(flowpolicy_dir)

    best_path = (flowpolicy_dir / args.best_json).resolve()
    if not best_path.is_file():
        raise SystemExit(f"Tidak menemukan best_trial: {best_path}")

    with open(best_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    best_params = payload.get("params", {})
    if not best_params:
        raise SystemExit("JSON tidak berisi field params")

    flavor = args.flavor
    if flavor == "auto":
        src = str(payload.get("source", "")) + " " + str(args.best_json)
        if "halving" in str(args.best_json) or "halving" in src:
            flavor = "halving"
        else:
            flavor = "grid"
    exp_d, lpre = _names_for_flavor(flavor)
    exp = args.exp_name or exp_d
    lp = args.log_prefix or lpre
    if args.log_name is not None:
        logn = args.log_name
    else:
        logn = f"{lp}_seed{args.seed}"

    cfg = map_best_params(best_params)
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        tag = f"{lp}_pre" if args.preprocess else f"{lp}_no_pre"
        run_dir = f"data/outputs/kitchen_complete-flowpolicy_{flavor}_final_{tag}_seed{args.seed}"

    cmd: list[str] = [
        sys.executable,
        "train.py",
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={args.seed}",
        "training.device=cuda",
        "training.debug=False",
        f"exp_name={exp}",
        f"logging.name={logn}",
        f"logging.mode={args.wandb_mode}",
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

    print("[train_best_trial] cmd:")
    print(" ".join(cmd))
    return int(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    raise SystemExit(main())
