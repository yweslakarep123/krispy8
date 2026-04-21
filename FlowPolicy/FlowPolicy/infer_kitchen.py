#!/usr/bin/env python3
"""
Jalankan inferensi FlowPolicy (low-dim) di FrankaKitchen-v1 dari checkpoint terlatih.

Menyimpan video MP4 per episode, `metrics.json` (success rate, latensi), dan
mencetak ringkasan ke stdout.

Contoh:
  cd /path/ke/FlowPolicy/FlowPolicy
  python infer_kitchen.py --checkpoint data/outputs/kitchen_complete-flowpolicy_seed0/checkpoints/latest.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Samakan layout dengan train.py: cwd = folder yang berisi train.py + infer_kitchen.py
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
os.chdir(_SCRIPT_DIR)

import dill
import hydra
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser(description="Inferensi Franka Kitchen dari checkpoint FlowPolicy")
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path ke file .ckpt (mis. .../checkpoints/latest.ckpt)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Folder keluaran (video + metrics). Default: <run_dir>/inference_<stem_ckpt>",
    )
    parser.add_argument(
        "--output-subdir", type=str, default=None,
        help="Jika diset, pakai nama subfolder ini di bawah run_dir (mis. "
             "inference_ep50). Diabaikan bila --output-dir disediakan.",
    )
    parser.add_argument(
        "--episodes", "-n", type=int, default=10,
        help="Jumlah episode evaluasi",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device PyTorch, mis. cuda:0 atau cpu",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Jika diset, kirim video ke Weights & Biases (perlu wandb login)",
    )
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint tidak ditemukan: {ckpt_path}")

    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    run_dir = ckpt_path.parent.parent
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    elif args.output_subdir:
        out_dir = run_dir / args.output_subdir
    else:
        out_dir = run_dir / f"inference_{ckpt_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    from train import TrainFlowPolicyWorkspace

    workspace = TrainFlowPolicyWorkspace(cfg, output_dir=str(run_dir))
    workspace.load_payload(payload)

    use_ema = bool(cfg.training.get("use_ema", True))
    policy = workspace.ema_model if use_ema and workspace.ema_model is not None else workspace.model
    policy.eval()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA tidak tersedia — memakai CPU.")
    else:
        device = torch.device(args.device)
    policy.to(device)

    # #region agent log
    def _dbg_infer(msg: str, data: dict):
        import time
        try:
            with open(
                "/home/daffa/Documents/skpsi/.cursor/debug-6ac186.log",
                "a",
                encoding="utf-8",
            ) as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "6ac186",
                            "runId": "infer-runner-cfg",
                            "hypothesisId": "H1_struct_merge",
                            "location": "infer_kitchen.py",
                            "message": msg,
                            "timestamp": int(time.time() * 1000),
                            "data": data,
                        },
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            pass

    # #endregion

    # H1: cfg.task.env_runner dari checkpoint sering struct=True → OmegaConf.merge
    # tidak boleh menambah kunci baru (save_videos_dir, wandb_log). Solusi: plain dict.
    _er = cfg.task.env_runner
    _struct = bool(OmegaConf.is_config(_er) and OmegaConf.is_struct(_er))
    _keys_before = list(OmegaConf.to_container(_er, resolve=True).keys())
    _dbg_infer(
        "env_runner before infer overrides",
        {"is_struct": _struct, "keys": _keys_before},
    )

    runner_dict = OmegaConf.to_container(_er, resolve=True)
    runner_dict["eval_episodes"] = args.episodes
    runner_dict["max_video_episodes"] = args.episodes
    runner_dict["record_video"] = True
    runner_dict["save_videos_dir"] = str(video_dir)
    runner_dict["wandb_log"] = bool(args.wandb)
    runner_cfg = OmegaConf.create(runner_dict)
    OmegaConf.set_struct(runner_cfg, False)
    _dbg_infer(
        "runner_cfg built from dict (struct disabled)",
        {"keys": list(runner_dict.keys()), "save_videos_dir": runner_dict.get("save_videos_dir")},
    )

    env_runner = hydra.utils.instantiate(runner_cfg, output_dir=str(out_dir))

    print(f"Checkpoint : {ckpt_path}")
    print(f"Output     : {out_dir}")
    print(f"Episodes   : {args.episodes}")
    print(f"Device     : {device}")
    print("Menjalankan rollout...")
    log = env_runner.run(policy)

    metrics = {}
    for k, v in log.items():
        if isinstance(v, float):
            metrics[k] = v
        elif k == "saved_video_paths":
            metrics[k] = v

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    sr = float(log.get("test_mean_score", 0.0))
    lat_s = float(log.get("mean_time", 0.0))
    lat_ms = lat_s * 1000.0

    print("\n========== HASIL INFERENSI ==========")
    print(f"Success rate (rata-rata fraksi task selesai per episode): {sr:.4f}")
    print(f"Mean inference latency (detik per langkah kontrol / per predict_action): {lat_s:.6f} s")
    print(f"Mean inference latency: {lat_ms:.3f} ms")
    print(f"Mean completed tasks per episode: {log.get('mean_n_completed_tasks', 'n/a')}")
    print(f"Metrik tersimpan di: {metrics_path}")
    vids = log.get("saved_video_paths") or []
    if vids:
        print("Video tersimpan:")
        for p in vids:
            print(f"  - {p}")
    else:
        print("Tidak ada video (render gagal atau record_video=False).")
    print("=====================================\n")


if __name__ == "__main__":
    main()
