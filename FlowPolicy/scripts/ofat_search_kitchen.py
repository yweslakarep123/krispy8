#!/usr/bin/env python3
"""OFAT (One-Factor-At-a-Time) sweep untuk FlowPolicy di FrankaKitchen-v1.

Eksplorasi SEMUA hyperparameter: tiap hyperparameter di-variasi melintasi 4
nilai sambil menahan hyperparameter lain pada BASELINE. Total run:

  8 hyperparameter x 4 nilai x 3 seed = 96 run

Tiap run = training + inferensi 50 episode. Resume-capable (lewat
`metrics.json`) dan cocok untuk Colab Free T4 yang bisa kena timeout 12
jam.

Contoh:
  cd FlowPolicy
  python scripts/ofat_search_kitchen.py --gpu 0
  python scripts/ofat_search_kitchen.py --gpu 0 --dry-run
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
from typing import Any, Dict, List, Sequence, Tuple


SEARCH_SPACE: Dict[str, List[Any]] = {
    "training.num_epochs":                           [500, 1000, 3000, 5000],
    "optimizer.lr":                                  [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size":                         [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps":          [1e-2, 1e-3, 1e-4, 1.0],
    "policy.Conditional_ConsistencyFM.delta":        [1e-2, 1e-3, 1e-4, 1.0],
    "n_action_steps":                                [2, 4, 6, 8],
    "n_obs_steps":                                   [4, 6, 8, 16],
}

# Baseline: satu nilai "center" per hyperparameter. Dipilih dari dalam
# SEARCH_SPACE sehingga kombinasi (hp=baseline_val) ikut tercakup sebagai
# salah satu dari 4 sweep untuk hp itu.
BASELINE: Dict[str, Any] = {
    "training.num_epochs":                           3000,
    "optimizer.lr":                                  1e-4,
    "dataloader.batch_size":                         128,
    "policy.Conditional_ConsistencyFM.num_segments": 2,
    "policy.Conditional_ConsistencyFM.eps":          1e-2,
    "policy.Conditional_ConsistencyFM.delta":        1e-2,
    "n_action_steps":                                4,
    "n_obs_steps":                                   4,
}

DEFAULT_SEEDS: List[int] = [0, 42, 101]
DEFAULT_EVAL_EPISODES: int = 50


def enumerate_configs() -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    """Hasil: list of (global_idx, hp_key, hp_value, full_config_dict).

    Ada 8 HP x 4 nilai = 32 konfigurasi unik. Dikembalikan dalam urutan
    deterministik: per HP, semua nilai (dari SEARCH_SPACE) dijalankan.
    """
    out: List[Tuple[int, str, Any, Dict[str, Any]]] = []
    idx = 0
    for hp_key, values in SEARCH_SPACE.items():
        for v in values:
            cfg = dict(BASELINE)
            cfg[hp_key] = v
            out.append((idx, hp_key, v, cfg))
            idx += 1
    return out


def format_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, float):
        return f"{key}={value:.6g}"
    return f"{key}={value}"


def build_train_cmd(cfg: Dict[str, Any], seed: int, run_dir: pathlib.Path,
                    save_ckpt: bool = True) -> List[str]:
    overrides = [
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={seed}",
        "training.device=cuda",
        "training.debug=False",
        "exp_name=ofat_search",
        f"logging.name=ofat_{run_dir.name}",
        "logging.mode=offline",
        f"checkpoint.save_ckpt={'True' if save_ckpt else 'False'}",
    ]
    for k, v in cfg.items():
        overrides.append(format_override(k, v))
        if k == "dataloader.batch_size":
            overrides.append(format_override("val_dataloader.batch_size", v))
    return [sys.executable, "train.py", *overrides]


def build_infer_cmd(ckpt_path: pathlib.Path, episodes: int,
                    output_subdir: str) -> List[str]:
    return [
        sys.executable, "infer_kitchen.py",
        "--checkpoint", str(ckpt_path),
        "--episodes", str(episodes),
        "--device", "cuda:0",
        "--output-subdir", output_subdir,
    ]


def metrics_path_for(run_dir: pathlib.Path, output_subdir: str) -> pathlib.Path:
    return run_dir / output_subdir / "metrics.json"


def read_metrics(path: pathlib.Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_result_row(results_csv: pathlib.Path, row: Dict[str, Any]) -> None:
    write_header = not results_csv.exists()
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_existing_results(results_csv: pathlib.Path) -> List[Dict[str, str]]:
    if not results_csv.exists():
        return []
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(results_csv: pathlib.Path, summary_csv: pathlib.Path,
                  enum_cfgs: List[Tuple[int, str, Any, Dict[str, Any]]]) -> None:
    rows = load_existing_results(results_csv)
    if not rows:
        print("[summary] results.csv kosong, summary dilewati.")
        return

    grouped: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        try:
            cid = int(r["cfg_idx"])
        except Exception:
            continue
        grouped.setdefault(cid, []).append(r)

    param_keys = list(SEARCH_SPACE.keys())
    metric_keys = ["test_mean_score", "mean_n_completed_tasks", "mean_time"]

    summary_rows: List[Dict[str, Any]] = []
    for cid, runs in grouped.items():
        if not runs:
            continue
        # find which HP/value this cfg belonged to
        hp_key, hp_value, cfg_dict = None, None, None
        for (gidx, k, v, cfg) in enum_cfgs:
            if gidx == cid:
                hp_key, hp_value, cfg_dict = k, v, cfg
                break
        summary_row: Dict[str, Any] = {
            "cfg_idx": cid,
            "swept_hp": hp_key,
            "swept_value": hp_value,
            "n_seeds": len(runs),
        }
        if cfg_dict is not None:
            for k in param_keys:
                summary_row[k] = cfg_dict[k]
        for mk in metric_keys:
            vals: List[float] = []
            for r in runs:
                try:
                    vals.append(float(r.get(mk, "nan")))
                except Exception:
                    pass
            if vals:
                m = sum(vals) / len(vals)
                s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
                summary_row[f"{mk}_mean"] = m
                summary_row[f"{mk}_std"] = s
            else:
                summary_row[f"{mk}_mean"] = float("nan")
                summary_row[f"{mk}_std"] = float("nan")
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda r: r.get("test_mean_score_mean", float("-inf")),
                      reverse=True)

    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"[summary] {summary_csv}  ({len(summary_rows)} konfigurasi)")
    print(f"[summary] Top-5 (mean test_mean_score):")
    for r in summary_rows[:5]:
        print(f"  cfg {r['cfg_idx']:02d} [{r['swept_hp']}={r['swept_value']}]: "
              f"SR={r.get('test_mean_score_mean'):.4f} +- "
              f"{r.get('test_mean_score_std'):.4f}  ({r.get('n_seeds')} seeds)")


def run_subprocess(cmd: Sequence[str], log_path: pathlib.Path,
                   env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    return proc.returncode


def _append_run_row(results_csv: pathlib.Path, cfg_idx: int, seed: int,
                    hp_key: str, hp_value: Any, cfg: Dict[str, Any],
                    metrics: Dict[str, float], status: str,
                    t_train: float = 0.0, t_infer: float = 0.0) -> None:
    row: Dict[str, Any] = {
        "cfg_idx": cfg_idx,
        "swept_hp": hp_key,
        "swept_value": hp_value,
        "seed": seed,
    }
    for k in SEARCH_SPACE.keys():
        row[k] = cfg.get(k)
    for mk in ("test_mean_score", "mean_n_completed_tasks", "mean_time",
               "mean_success_rates", "SR_test_L3", "SR_test_L5"):
        v = metrics.get(mk)
        row[mk] = float(v) if v is not None else ""
    row["status"] = status
    row["t_train_s"] = round(t_train, 2)
    row["t_infer_s"] = round(t_infer, 2)
    row["timestamp"] = int(time.time())
    append_result_row(results_csv, row)


# ----- progress reporting (Colab-friendly) -----------------------------------

_PROGRESS_FILE: pathlib.Path = None  # set di main(), dipakai oleh helper di bawah


def _pr(msg: str = "") -> None:
    """Print ke stdout (flush) + append ke progress.log.

    `BrokenPipeError` ditangkap diam-diam supaya aman ketika output
    di-pipe ke `head`, `tail`, dll. di Colab (`set -o pipefail`).
    """
    # #region agent log
    try:
        import json as _json, time as _time
        with open("/home/daffa/Documents/skpsi/.cursor/debug-8dee3e.log",
                  "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({
                "sessionId": "8dee3e",
                "runId": "post-fix-sigpipe",
                "hypothesisId": "H1_broken_pipe",
                "location": "ofat_search_kitchen.py:_pr",
                "message": "print attempt",
                "timestamp": int(_time.time() * 1000),
                "data": {"len": len(msg)},
            }) + "\n")
    except Exception:
        pass
    # #endregion
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        # Stdout ditutup downstream (mis. head). Biarkan sisa eksekusi
        # menulis ke progress.log saja tanpa crash.
        try:
            import sys as _sys
            _sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
    if _PROGRESS_FILE is not None:
        try:
            with open(_PROGRESS_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _progress_line(done: int, total: int, cfg_idx: int, seed: int,
                   hp_key: str, hp_value: Any, status: str,
                   sr: Any, t_train: float, t_infer: float,
                   elapsed_sweep: float) -> str:
    if isinstance(sr, (int, float)):
        sr_str = f"SR={sr:.4f}"
    else:
        sr_str = "SR=   n/a "
    eta_str = ""
    if done > 0 and total > 0:
        avg = elapsed_sweep / done
        eta = avg * (total - done)
        eta_str = f" eta={_fmt_hms(eta)}"
    return (f"[PROGRESS {done:3d}/{total}] cfg_{cfg_idx:02d} seed={seed:<3d} "
            f"[{hp_key}={hp_value}] {sr_str} status={status} "
            f"t_train={t_train:5.0f}s t_infer={t_infer:4.0f}s "
            f"elapsed={_fmt_hms(elapsed_sweep)}{eta_str}")


def main() -> int:
    # Standard unix behaviour for broken pipes (e.g. `| head`). Python
    # defaults to raising BrokenPipeError; we ingin diam seperti `cat`.
    try:
        import signal as _signal
        if hasattr(_signal, "SIGPIPE"):
            _signal.signal(_signal.SIGPIPE, _signal.SIG_DFL)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="OFAT sweep FlowPolicy kitchen")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/ofat_search")
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--only-cfg", type=int, nargs="+", default=None,
                        help="Hanya jalankan cfg_idx ini (debugging / partial resume)")
    parser.add_argument("--only-hp", type=str, default=None,
                        help="Hanya jalankan sweep untuk 1 hyperparameter ini "
                             "(mis. 'optimizer.lr')")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Auto-stop setelah N menit (berguna untuk Colab "
                             "free T4 yang bisa kena timeout 12 jam).")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    assert flowpolicy_dir.is_dir(), \
        f"Tidak menemukan folder FlowPolicy/FlowPolicy di {flowpolicy_dir}"

    out_root = (flowpolicy_dir / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "results.csv"
    summary_csv = out_root / "summary.csv"
    configs_json = out_root / "configs.json"
    progress_log = out_root / "progress.log"

    # Aktifkan penulisan progress ke file supaya bisa di-tail dari Colab cell
    # lain walaupun stdout %%bash di-buffer.
    global _PROGRESS_FILE
    _PROGRESS_FILE = progress_log

    enum_cfgs = enumerate_configs()

    # hitung total run setelah filter (--only-hp / --only-cfg)
    filtered_cfgs = [
        (gidx, k, v, cfg) for (gidx, k, v, cfg) in enum_cfgs
        if (not args.only_cfg or gidx in args.only_cfg)
        and (not args.only_hp or k == args.only_hp)
    ]
    total_planned = len(filtered_cfgs) * len(args.seeds)

    # dump konfigurasi yang akan dijalankan
    cfg_dump = [
        {"cfg_idx": gidx, "swept_hp": k, "swept_value": v, "full_cfg": cfg}
        for (gidx, k, v, cfg) in enum_cfgs
    ]
    with open(configs_json, "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, default=str)
    _pr(f"[ofat] {len(enum_cfgs)} konfigurasi unik x {len(args.seeds)} seed "
        f"= {len(enum_cfgs)*len(args.seeds)} run total "
        f"(akan dijalankan/di-resume: {total_planned})")
    _pr(f"[ofat] konfigurasi ditulis ke {configs_json}")
    _pr(f"[ofat] progress log    : {progress_log}")
    _pr(f"[ofat] results.csv     : {results_csv}")
    _pr(f"[ofat] baseline: {BASELINE}")

    if args.dry_run:
        _pr("[dry-run] list konfigurasi:")
        for (gidx, k, v, cfg) in enum_cfgs:
            if args.only_cfg and gidx not in args.only_cfg:
                continue
            if args.only_hp and k != args.only_hp:
                continue
            _pr(f"  cfg {gidx:02d}: sweep[{k}={v}] -> {cfg}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    # Paksa child python unbuffered supaya log di-stream ke stdout secara
    # real-time walau output diteruskan ke file.
    env["PYTHONUNBUFFERED"] = "1"

    t_sweep_start = time.time()
    done = 0
    for (gidx, hp_key, hp_value, cfg) in filtered_cfgs:
        for seed in args.seeds:
            done += 1
            elapsed = time.time() - t_sweep_start

            # time-budget check
            if args.max_minutes is not None:
                elapsed_min = elapsed / 60.0
                if elapsed_min >= args.max_minutes:
                    _pr(f"\n[ofat] --max-minutes={args.max_minutes} tercapai "
                        f"({elapsed_min:.1f} min), stop di {done-1}/"
                        f"{total_planned}.")
                    if not args.skip_summary:
                        write_summary(results_csv, summary_csv, enum_cfgs)
                    return 0

            run_dir = (out_root / f"cfg_{gidx:02d}_seed{seed}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            infer_subdir = f"inference_ep{args.episodes}"
            mpath = metrics_path_for(run_dir, infer_subdir)

            _pr(f"\n========== [cfg {gidx:02d} / seed {seed}] "
                f"({done}/{total_planned}) sweep[{hp_key}={hp_value}] "
                f"==========")
            _pr(f"  cfg      : {cfg}")
            _pr(f"  run_dir  : {run_dir}")

            # Resume: metrics sudah ada -> skip
            if mpath.is_file():
                try:
                    metrics = read_metrics(mpath)
                    if "test_mean_score" in metrics:
                        sr = metrics["test_mean_score"]
                        _pr(f"  [skip] metrics sudah ada: SR={sr:.4f}")
                        _append_run_row(results_csv, gidx, seed, hp_key,
                                        hp_value, cfg, metrics,
                                        status="skip_resume")
                        _pr(_progress_line(
                            done, total_planned, gidx, seed, hp_key, hp_value,
                            "skip_resume", sr, 0.0, 0.0,
                            time.time() - t_sweep_start))
                        continue
                except Exception:
                    pass

            # ---- training ----
            t0 = time.time()
            ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
            if ckpt_path.is_file():
                _pr(f"  [skip-train] ckpt sudah ada: {ckpt_path}")
            else:
                train_cmd = build_train_cmd(cfg, seed, run_dir, save_ckpt=True)
                _pr(f"  [train] {' '.join(train_cmd)}")
                rc = run_subprocess(
                    train_cmd, run_dir / "train_stdout.log", env=env)
                if rc != 0 or not ckpt_path.is_file():
                    t_train = time.time() - t0
                    _pr(f"  [ERROR] training gagal (rc={rc}), skip inferensi.")
                    _append_run_row(results_csv, gidx, seed, hp_key, hp_value,
                                    cfg, {},
                                    status=f"train_failed_rc{rc}",
                                    t_train=t_train)
                    _pr(_progress_line(
                        done, total_planned, gidx, seed, hp_key, hp_value,
                        f"train_failed_rc{rc}", None, t_train, 0.0,
                        time.time() - t_sweep_start))
                    continue
            t_train = time.time() - t0

            # ---- inference ----
            t1 = time.time()
            infer_cmd = build_infer_cmd(
                ckpt_path, episodes=args.episodes, output_subdir=infer_subdir)
            _pr(f"  [infer] {' '.join(infer_cmd)}")
            rc = run_subprocess(
                infer_cmd, run_dir / "infer_stdout.log", env=env)
            t_infer = time.time() - t1

            metrics: Dict[str, float] = {}
            if rc == 0 and mpath.is_file():
                try:
                    metrics = read_metrics(mpath)
                except Exception as exc:
                    _pr(f"  [WARN] gagal parse metrics: {exc}")
            else:
                _pr(f"  [ERROR] inferensi gagal (rc={rc})")

            status = "ok" if metrics else f"infer_failed_rc{rc}"
            _append_run_row(results_csv, gidx, seed, hp_key, hp_value, cfg,
                            metrics, status=status,
                            t_train=t_train, t_infer=t_infer)
            _pr(_progress_line(
                done, total_planned, gidx, seed, hp_key, hp_value, status,
                metrics.get("test_mean_score") if metrics else None,
                t_train, t_infer, time.time() - t_sweep_start))

    if not args.skip_summary:
        write_summary(results_csv, summary_csv, enum_cfgs)
    _pr(f"[done] Selesai. Total elapsed: "
        f"{_fmt_hms(time.time() - t_sweep_start)}")
    return 0


if __name__ == "__main__":
    _here = pathlib.Path(__file__).resolve().parent
    _flowpolicy_pkg = _here.parent / "FlowPolicy"
    os.chdir(_flowpolicy_pkg)
    sys.exit(main())
