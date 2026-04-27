#!/usr/bin/env python3
"""Grid search (produk Cartesian list terpendek) untuk FlowPolicy Franka Kitchen.

Setiap sel: 3 seed x 2 preprocess = 6 train+infer; skor = rata 6x test_mean_score.
Opsi --gpu-pool 0,1,2: tiga proses paralel, satu proses per seed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import sys
import time
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hparam_search_common import (
    DEFAULT_SEEDS,
    apply_vram_safe_batch,
    hydra_cfg_to_estimator_params,
    is_cfg_valid,
    log_line,
    read_metrics,
    run_one_train_infer,
)

# Produk: 2^6 * 3 = 192 sel (setelah filter n_groups|down_dims)
GRID_EPOCHS = [2000, 5000]
GRID_LR = [1e-4, 1e-3]
GRID_BATCH = [128, 256]
GRID_N_ACTION = [4, 6]
GRID_N_OBS = [4, 16]
GRID_DOWN_NGROUPS: List[Tuple[List[int], int]] = [
    ([128, 256, 512], 4),
    ([256, 512, 1024], 8),
    ([192, 384, 768], 4),
]
GRID_SEGMENTS = [1, 2]
GRID_KERNEL = [3, 7]

BASELINE_POLICY: Dict[str, Any] = {
    "policy.Conditional_ConsistencyFM.eps": 0.01,
    "policy.Conditional_ConsistencyFM.delta": 0.01,
    "policy.diffusion_step_embed_dim": 128,
    "policy.encoder_output_dim": 64,
    "policy.state_mlp_size": [256, 256],
    "policy.action_clip": 1.0,
    "policy.encoder_use_layernorm": True,
    "policy.use_down_condition": True,
    "policy.use_mid_condition": True,
    "policy.use_up_condition": True,
}

GRID_TRIALS_FIELDNAMES = [
    "grid_id",
    "preprocess",
    "seed",
    "value",
    "mean_time",
    "elapsed_s",
    "episodes",
    "config_json",
]
GRID_SUMMARY_FIELDNAMES = [
    "grid_id",
    "mean_value",
    "std_value",
    "n_runs",
    "config_json",
]


def _tup_to_cfg(
    t: Tuple[Any, ...],
) -> Dict[str, Any]:
    ne, lr, bs, na, nobs, dpack, nseg, ksz = t
    dd, ng = dpack
    cfg: Dict[str, Any] = {
        "training.num_epochs": int(ne),
        "optimizer.lr": float(lr),
        "dataloader.batch_size": int(bs),
        "n_action_steps": int(na),
        "n_obs_steps": int(nobs),
        "policy.down_dims": list(dd),
        "policy.n_groups": int(ng),
        "policy.Conditional_ConsistencyFM.num_segments": int(nseg),
        "policy.kernel_size": int(ksz),
    }
    cfg.update(BASELINE_POLICY)
    return cfg


def enumerate_valid_grid() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tup in product(
        GRID_EPOCHS,
        GRID_LR,
        GRID_BATCH,
        GRID_N_ACTION,
        GRID_N_OBS,
        GRID_DOWN_NGROUPS,
        GRID_SEGMENTS,
        GRID_KERNEL,
    ):
        cfg = _tup_to_cfg(
            (tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6], tup[7])
        )
        ok, _ = is_cfg_valid(cfg)
        if ok:
            out.append(cfg)
    return out


def _mp_one_seed(
    flow_dir: str,
    tcfg: Dict[str, Any],
    seed: int,
    run_dir: str,
    ep: int,
    env: Dict[str, str],
    exp_name: str,
    logn: str,
    preprocess: bool,
    infer_sub: str,
) -> Dict[str, Any]:
    os.chdir(flow_dir)
    pr: Optional[Dict[str, Any]] = {"enabled": True} if preprocess else None
    p = pathlib.Path(run_dir)
    sr, lat, rc, rci = run_one_train_infer(
        tcfg,
        seed,
        p,
        ep,
        env,
        exp_name,
        logn,
        pr,
        infer_sub,
    )
    mpath = p / infer_sub / "metrics.json"
    if mpath.is_file() and rc == 0 and rci == 0:
        try:
            m = read_metrics(mpath)
            sr = float(m.get("test_mean_score", sr))
            lat = float(m.get("mean_time", lat))
        except Exception:
            pass
    v, mt = 0.0, 0.0
    if (
        rc == 0
        and (p / "checkpoints" / "latest.ckpt").is_file()
        and rci == 0
        and mpath.is_file()
    ):
        v, mt = float(sr), float(lat)
    return {"seed": seed, "value": v, "mean_time": mt}


def _run_seeds(
    flowpolicy_dir: pathlib.Path,
    out_root: pathlib.Path,
    grid_id: int,
    pre_idx: int,
    train_cfg: Dict[str, Any],
    seeds: Sequence[int],
    episodes: int,
    exp_name: str,
    base_env: Dict[str, str],
    use_gpu: str,
    parallel_gpus: Optional[Sequence[str]],
    preprocess: bool,
) -> List[Dict[str, Any]]:
    infer_sub = f"inference_ep{episodes}"
    tcfg = dict(train_cfg)
    out_rows: List[Dict[str, Any]] = []
    pgp = [g.strip() for g in (parallel_gpus or []) if g and str(g).strip()]
    if len(pgp) >= len(seeds) and len(seeds) >= 1:
        fdir = str(flowpolicy_dir)
        futs = []
        nwk = min(len(seeds), len(pgp))
        with concurrent.futures.ProcessPoolExecutor(max_workers=nwk) as ex:
            for s, g in zip(seeds, pgp[:nwk]):
                run_path = out_root / f"cfg_{grid_id:05d}_pre{pre_idx}_seed{s}"
                e = {**base_env, "CUDA_VISIBLE_DEVICES": g, "HYDRA_FULL_ERROR": "1", "PYTHONUNBUFFERED": "1"}
                futs.append(
                    ex.submit(
                        _mp_one_seed,
                        fdir,
                        tcfg,
                        int(s),
                        str(run_path),
                        int(episodes),
                        e,
                        str(exp_name),
                        f"grid_cfg_{grid_id:05d}_pre{pre_idx}_s{s}",
                        preprocess,
                        infer_sub,
                    )
                )
            for f in futs:
                d = f.result()
                out_rows.append(
                    {
                        "seed": d["seed"],
                        "value": d["value"],
                        "mean_time": d["mean_time"],
                    }
                )
        return out_rows
    e2 = {**base_env, "CUDA_VISIBLE_DEVICES": str(use_gpu), "HYDRA_FULL_ERROR": "1", "PYTHONUNBUFFERED": "1"}
    for s in seeds:
        p = out_root / f"cfg_{grid_id:05d}_pre{pre_idx}_seed{s}"
        d = _mp_one_seed(
            str(flowpolicy_dir),
            tcfg,
            int(s),
            str(p),
            int(episodes),
            e2,
            str(exp_name),
            f"grid_cfg_{grid_id:05d}_pre{pre_idx}_s{s}",
            preprocess,
            infer_sub,
        )
        out_rows.append(
            {
                "seed": d["seed"],
                "value": d["value"],
                "mean_time": d["mean_time"],
            }
        )
    return out_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search FlowPolicy (list padat)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--out-root", type=str, default="data/outputs/grid_search_kitchen")
    p.add_argument(
        "--max-grid-runs",
        type=int,
        default=500,
        help="Gagal bila jumlah sel valid > nilai ini",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU tunggal bila urut (bukan --gpu-pool)",
    )
    p.add_argument(
        "--gpu-pool",
        type=str,
        default=None,
        help="Koma, mis. 0,1,2 — paralel tiga proses, satu proses per seed (untuk 3 seed).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-mem-cap", action="store_true")
    p.add_argument("--start-grid-id", type=int, default=0)
    p.add_argument("--max-grid-cells", type=int, default=None)
    p.add_argument("--exp-name", type=str, default="grid_search", dest="exp_name")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = (script_dir.parent / "FlowPolicy").resolve()
    if not flowpolicy_dir.is_dir():
        raise SystemExit(f"Tidak menemukan {flowpolicy_dir}")

    all_cfgs = enumerate_valid_grid()
    n_sel = len(all_cfgs)
    if n_sel == 0:
        raise SystemExit("Grid valid kosong; periksa GRID_* / is_cfg_valid")

    gslice = all_cfgs[args.start_grid_id :]
    if args.max_grid_cells is not None:
        gslice = gslice[: args.max_grid_cells]
    ncells = len(gslice)
    n_after = n_sel
    if not args.dry_run and n_after > args.max_grid_runs:
        raise SystemExit(
            f"Jumlah sel setelah is_cfg_valid = {n_after} > --max-grid-runs={args.max_grid_runs}."
        )

    os.chdir(flowpolicy_dir)
    out_root = (flowpolicy_dir / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_p = out_root / "progress.log"
    trial_csv = out_root / "grid_trials.csv"
    sum_csv = out_root / "grid_summary.csv"
    best_json = out_root / "best_trial.json"
    t0 = time.time()

    gpu_list: Optional[List[str]] = None
    if args.gpu_pool:
        gpu_list = [x.strip() for x in str(args.gpu_pool).split(",") if x.strip()]

    if args.dry_run:
        log_line(f"[grid] dry-run: n_all={n_after} n_slice={ncells} out={out_root}", log_p)
        if gpu_list:
            log_line(f"[grid] gpu-pool={gpu_list} (3 seed paralel jika >=3 id)", log_p)
        return 0

    seeds: List[int] = list(args.seeds)
    if not trial_csv.is_file():
        with open(trial_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=GRID_TRIALS_FIELDNAMES).writeheader()
    if not sum_csv.is_file():
        with open(sum_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=GRID_SUMMARY_FIELDNAMES).writeheader()

    base_env = {**os.environ}
    if not gpu_list or len(gpu_list) < len(seeds):
        base_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    best_m = -1.0
    best_cfg: Optional[Dict[str, Any]] = None
    best_id = -1

    for j, tcfg0 in enumerate(gslice):
        grid_id = args.start_grid_id + j
        tcfg = dict(tcfg0)
        if not args.no_mem_cap:
            tcfg, mem_note = apply_vram_safe_batch(tcfg)
            if mem_note:
                log_line(f"  [mem-cap] {mem_note}", log_p)

        log_line(f"[grid] cell {grid_id} ...", log_p)
        all_vals: List[float] = []
        batch: List[Dict[str, Any]] = []

        for pre_idx, pre in ((0, False), (1, True)):
            rows = _run_seeds(
                flowpolicy_dir,
                out_root,
                grid_id,
                pre_idx,
                tcfg,
                seeds,
                int(args.episodes),
                str(args.exp_name),
                base_env,
                str(args.gpu),
                (gpu_list if (gpu_list and len(gpu_list) >= len(seeds)) else None),
                pre,
            )
            for r in rows:
                all_vals.append(float(r["value"]))
                batch.append(
                    {
                        **r,
                        "preprocess": pre_idx,
                    }
                )

        m = float(sum(all_vals) / len(all_vals)) if all_vals else 0.0
        std = (
            float(
                (sum((x - m) ** 2 for x in all_vals) / max(len(all_vals), 1)) ** 0.5
            )
            if all_vals
            else 0.0
        )

        for b in batch:
            row = {
                "grid_id": grid_id,
                "preprocess": b["preprocess"],
                "seed": b["seed"],
                "value": b["value"],
                "mean_time": b["mean_time"],
                "elapsed_s": round(time.time() - t0, 2),
                "episodes": int(args.episodes),
                "config_json": json.dumps(tcfg, default=str),
            }
            with open(trial_csv, "a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=GRID_TRIALS_FIELDNAMES).writerow(row)

        with open(sum_csv, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=GRID_SUMMARY_FIELDNAMES).writerow(
                {
                    "grid_id": grid_id,
                    "mean_value": f"{m:.6f}",
                    "std_value": f"{std:.6f}",
                    "n_runs": len(all_vals),
                    "config_json": json.dumps(tcfg, default=str),
                }
            )

        log_line(
            f"[grid] cell {grid_id} done mean_six={m:.4f} std={std:.4f}",
            log_p,
        )
        if m > best_m:
            best_m, best_cfg, best_id = m, dict(tcfg), grid_id

    if best_cfg is not None:
        payload = {
            "score": best_m,
            "grid_id": int(best_id),
            "source": "grid_search_kitchen",
            "params": hydra_cfg_to_estimator_params(best_cfg),
        }
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        log_line(f"[grid] best_trial -> {best_json} (grid {best_id}, score {best_m:.4f})", log_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
