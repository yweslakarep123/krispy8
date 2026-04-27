#!/usr/bin/env python3
"""Train final model dari hasil best_trial random search."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List


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


def _run(cmd: List[str], cwd: pathlib.Path, env: Dict[str, str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{stamp} | CMD | {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train final dari best_trial random search")
    p.add_argument("--best-json", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--preprocess", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    here = pathlib.Path(__file__).resolve().parent
    flowpolicy_pkg = here.parent / "FlowPolicy"

    with open(args.best_json, "r", encoding="utf-8") as f:
        best = json.load(f)
    params = best.get("params", {})
    if not isinstance(params, dict) or not params:
        raise SystemExit("best_trial.json tidak memiliki field params yang valid.")

    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        py,
        "train.py",
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        format_override("hydra.run.dir", str(run_dir)),
        format_override("training.seed", args.seed),
        "training.device=cuda",
        "training.debug=False",
        "exp_name=exp_3seed_4arms",
        format_override("logging.name", f"seed{args.seed}_{run_dir.parent.name}"),
        "logging.mode=offline",
        "checkpoint.save_ckpt=True",
        "training.resume=False",
        format_override("task.dataset.preprocess.enabled", bool(args.preprocess)),
    ]
    for k, v in params.items():
        cmd.append(format_override(k, v))
        if k == "dataloader.batch_size":
            cmd.append(format_override("val_dataloader.batch_size", v))

    rc = _run(cmd, flowpolicy_pkg, env, run_dir.parent / "train_best_from_random.log")
    return rc


if __name__ == "__main__":
    sys.exit(main())
