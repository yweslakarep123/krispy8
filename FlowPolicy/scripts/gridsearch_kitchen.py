#!/usr/bin/env python3
"""Grid Search (full cartesian) untuk FlowPolicy di FrankaKitchen-v1.

Berbeda dengan OFAT (`ofat_search_kitchen.py`) yang hanya mengubah satu
hyperparameter sekaligus, skrip ini menjalankan **full cartesian product**
untuk subset hyperparameter yang dipilih, lalu setiap konfigurasi
dijalankan di beberapa seed (default 3).

- Semua HP non-subset dikunci ke BASELINE (sejajar dengan nilai default
  di `flow_policy_3d/config/flowpolicy.yaml`).
- Tiap kombinasi (cfg_idx, seed) mendapat `run_dir` unik dan checkpoint
  sendiri, sehingga TIDAK ada run yang melanjutkan bobot run lain
  (isolation). Auto-resume dari `metrics.json` tetap ada untuk recovery.

Contoh:
  cd FlowPolicy
  # preset default subset 7 HP (~1296 run)
  bash scripts/gridsearch_kitchen.sh 0 --dry-run
  bash scripts/gridsearch_kitchen.sh 0

  # mode preprocessing (out-root otomatis ke gridsearch_preprocess)
  bash scripts/gridsearch_kitchen.sh 0 --preprocess

  # override subset: tambahkan nilai tambahan untuk satu HP
  bash scripts/gridsearch_kitchen.sh 0 --override-hp optimizer.lr=1e-3:5e-4:1e-4

  # subset preset kecil (untuk smoke test)
  bash scripts/gridsearch_kitchen.sh 0 --subset smoke
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# HP lama (sebagian besar sama dengan OFAT). Kalau HP tidak ada di subset,
# BASELINE yang dipakai.
SEARCH_SPACE_BASE: Dict[str, List[Any]] = {
    "training.num_epochs":                           [500, 1000, 3000, 5000],
    "optimizer.lr":                                  [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size":                         [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps":          [1e-2, 1e-3, 1e-4, 1.0],
    "policy.Conditional_ConsistencyFM.delta":        [1e-2, 1e-3, 1e-4, 1.0],
    "n_action_steps":                                [2, 4, 6, 8],
    "n_obs_steps":                                   [4, 6, 8, 16],
}

# 11 HP model baru (minus `condition_type` sesuai permintaan).
# Non-boolean: 4 nilai. Boolean: 2 nilai.
SEARCH_SPACE_MODEL: Dict[str, List[Any]] = {
    "policy.down_dims": [
        [128, 256, 512],
        [192, 384, 768],
        [256, 512, 1024],
        [384, 768, 1536],
    ],
    "policy.diffusion_step_embed_dim": [64, 128, 256, 512],
    "policy.kernel_size":              [3, 5, 7, 9],
    # n_groups harus membagi semua channel di down_dims. [1,2,4,8] aman
    # untuk seluruh down_dims di atas (semua channel habis dibagi 8).
    "policy.n_groups":                 [1, 2, 4, 8],
    "policy.encoder_output_dim":       [32, 64, 128, 256],
    "policy.state_mlp_size": [
        [128, 128],
        [256, 256],
        [512, 512],
        [256, 256, 256],
    ],
    "policy.action_clip":              [0.5, 1.0, 2.0, 5.0],
    # booleans (2 nilai)
    "policy.encoder_use_layernorm":    [True, False],
    "policy.use_down_condition":       [True, False],
    "policy.use_mid_condition":        [True, False],
    "policy.use_up_condition":         [True, False],
}

# BASELINE: nilai default untuk SEMUA HP sweepable. Dipilih sejalan
# dengan `FlowPolicy/flow_policy_3d/config/flowpolicy.yaml` saat ini.
BASELINE: Dict[str, Any] = {
    # base
    "training.num_epochs":                           3000,
    "optimizer.lr":                                  1e-4,
    "dataloader.batch_size":                         128,
    "policy.Conditional_ConsistencyFM.num_segments": 2,
    "policy.Conditional_ConsistencyFM.eps":          1e-2,
    "policy.Conditional_ConsistencyFM.delta":        1e-2,
    "n_action_steps":                                4,
    "n_obs_steps":                                   4,
    # model
    "policy.down_dims":              [256, 512, 1024],
    "policy.diffusion_step_embed_dim": 128,
    "policy.kernel_size":            5,
    "policy.n_groups":               8,
    "policy.encoder_output_dim":     64,
    "policy.state_mlp_size":         [256, 256],
    "policy.action_clip":            1.0,
    "policy.encoder_use_layernorm":  True,
    "policy.use_down_condition":     True,
    "policy.use_mid_condition":      True,
    "policy.use_up_condition":       True,
}


# ---------------------------------------------------------------------------
# Subset presets
# ---------------------------------------------------------------------------

# Subset default sesuai plan: 7 HP, campuran (lama + model), ~1296 run × 3 seed.
SUBSETS: Dict[str, Dict[str, List[Any]]] = {
    "default": {
        "optimizer.lr":                                  [1e-3, 1e-4, 1e-5],
        "dataloader.batch_size":                         [64, 128, 256],
        "policy.Conditional_ConsistencyFM.num_segments": [2, 3],
        "policy.down_dims": [
            [128, 256, 512],
            [256, 512, 1024],
            [384, 768, 1536],
        ],
        "policy.diffusion_step_embed_dim":               [128, 256],
        "policy.kernel_size":                            [3, 5],
        "policy.use_mid_condition":                      [True, False],
    },
    # Preset kecil (~16 kombinasi × 3 seed = 48 run) untuk smoke test.
    "smoke": {
        "optimizer.lr":              [1e-3, 1e-4],
        "policy.down_dims":          [[128, 256, 512], [256, 512, 1024]],
        "policy.kernel_size":        [3, 5],
        "policy.use_mid_condition":  [True, False],
    },
}

DEFAULT_SEEDS: List[int] = [0, 42, 101]
DEFAULT_EVAL_EPISODES: int = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_value(v: Any) -> str:
    """Format satu nilai Python ke sintaks Hydra override."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ",".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def format_override(key: str, value: Any) -> str:
    return f"{key}={_fmt_value(value)}"


def _parse_literal(text: str) -> Any:
    """Parse '1e-3', 'true', '[128,256,512]' dll. menjadi nilai Python."""
    text = text.strip()
    # Booleans (case-insensitive)
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # Angka / list literal
    try:
        return ast.literal_eval(text)
    except Exception:
        return text  # fallback: string mentah


def _split_respect_brackets(text: str, sep: str = ":") -> List[str]:
    """Split `text` pada `sep` tapi respect [ ] ( ) nesting."""
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch in "[(":
            depth += 1
            buf.append(ch)
        elif ch in "])":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [s for s in (x.strip() for x in out) if s]


def parse_override_spec(spec: str) -> Tuple[str, List[Any]]:
    """Parse `key=v1:v2:v3`. Nilai list pakai kurung siku. Contoh:
    `optimizer.lr=1e-3:1e-4`, `policy.down_dims=[128,256,512]:[256,512,1024]`.
    """
    if "=" not in spec:
        raise ValueError(f"--override-hp butuh 'key=v1:v2:...'; dapat: {spec!r}")
    key, rhs = spec.split("=", 1)
    key = key.strip()
    parts = _split_respect_brackets(rhs, sep=":")
    if not parts:
        raise ValueError(f"--override-hp tidak punya nilai: {spec!r}")
    vals = [_parse_literal(p) for p in parts]
    return key, vals


def _n_groups_compatible(n_groups: int, down_dims: List[int]) -> bool:
    """GroupNorm di ConditionalResidualBlock1D butuh n_groups membagi
    semua channel di down_dims."""
    return all(d % n_groups == 0 for d in down_dims)


def is_cfg_valid(cfg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Sanity check untuk kombinasi HP yang tidak kompatibel.

    Kembalikan (valid, reason). `reason` hanya diisi bila invalid.
    """
    n_groups = int(cfg.get("policy.n_groups", 8))
    down_dims = list(cfg.get("policy.down_dims", [256, 512, 1024]))
    if not _n_groups_compatible(n_groups, down_dims):
        return False, (
            f"n_groups={n_groups} tidak membagi semua channel di "
            f"down_dims={down_dims}")
    # `n_obs_steps + n_action_steps - 1` → horizon; horizon harus >= 4 dan
    # kelipatan 4 setelah pembulatan (sudah di-handle di yaml). Tidak di-enforce di sini.
    return True, None


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_subset(preset_name: str,
                 override_specs: Sequence[str]) -> Dict[str, List[Any]]:
    """Gabungkan preset + override CLI menjadi subset final."""
    if preset_name not in SUBSETS:
        raise KeyError(
            f"preset '{preset_name}' tidak dikenal. Pilih dari: "
            f"{sorted(SUBSETS)} atau pakai preset='empty' lalu --override-hp")
    subset: Dict[str, List[Any]] = {
        k: list(v) for k, v in SUBSETS[preset_name].items()
    }
    for spec in override_specs or []:
        key, vals = parse_override_spec(spec)
        if key not in BASELINE:
            raise KeyError(
                f"HP '{key}' tidak ada di BASELINE / bukan HP sweepable. "
                f"Periksa ejaan. Valid: {sorted(BASELINE.keys())}")
        subset[key] = vals
    if not subset:
        raise ValueError("subset kosong: setidaknya 1 HP harus di-sweep.")
    return subset


def enumerate_configs_grid(subset: Dict[str, List[Any]]
                           ) -> List[Tuple[int, Dict[str, Any]]]:
    """Cartesian product. Hasil: list (cfg_idx, full_config_dict).

    `full_config_dict` = BASELINE dengan HP di subset di-overwrite.
    """
    keys = list(subset.keys())
    value_lists = [subset[k] for k in keys]
    out: List[Tuple[int, Dict[str, Any]]] = []
    idx = 0
    for combo in itertools.product(*value_lists):
        cfg = dict(BASELINE)
        for k, v in zip(keys, combo):
            cfg[k] = v
        out.append((idx, cfg))
        idx += 1
    return out


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def build_train_cmd(cfg: Dict[str, Any], seed: int, run_dir: pathlib.Path,
                    save_ckpt: bool,
                    preprocess_overrides: Optional[Dict[str, Any]] = None,
                    resume: bool = False) -> List[str]:
    overrides = [
        "--config-name=flowpolicy",
        "task=kitchen_complete",
        f"hydra.run.dir={run_dir}",
        f"training.seed={seed}",
        "training.device=cuda",
        "training.debug=False",
        "exp_name=gridsearch",
        f"logging.name=grid_{run_dir.name}",
        "logging.mode=offline",
        f"checkpoint.save_ckpt={'True' if save_ckpt else 'False'}",
        f"training.resume={'True' if resume else 'False'}",
    ]
    for k, v in cfg.items():
        overrides.append(format_override(k, v))
        if k == "dataloader.batch_size":
            overrides.append(format_override("val_dataloader.batch_size", v))
    if preprocess_overrides:
        for k, v in preprocess_overrides.items():
            overrides.append(format_override(f"task.dataset.preprocess.{k}", v))
    return [sys.executable, "train.py", *overrides]


def build_infer_cmd(ckpt_path: pathlib.Path, episodes: int,
                    output_subdir: str) -> List[str]:
    return [
        sys.executable, "infer_kitchen.py",
        "--checkpoint", str(ckpt_path),
        "--episodes", str(episodes),
        "--device", "cuda:0",
        "--output-subdir", output_subdir,
        "--no-video",   # sweep training: skip video untuk hemat waktu
    ]


def metrics_path_for(run_dir: pathlib.Path, output_subdir: str) -> pathlib.Path:
    return run_dir / output_subdir / "metrics.json"


def read_metrics(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# ---------------------------------------------------------------------------
# CSV & summary
# ---------------------------------------------------------------------------

def append_result_row(results_csv: pathlib.Path, row: Dict[str, Any],
                      fieldnames: List[str]) -> None:
    write_header = not results_csv.exists()
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_existing_results(results_csv: pathlib.Path) -> List[Dict[str, str]]:
    if not results_csv.exists():
        return []
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(results_csv: pathlib.Path, summary_csv: pathlib.Path,
                  subset: Dict[str, List[Any]]) -> None:
    rows = load_existing_results(results_csv)
    if not rows:
        print("[summary] results.csv kosong, summary dilewati.")
        return
    # groupkey: tuple dari nilai subset
    keys = list(subset.keys())
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for r in rows:
        gk = tuple(str(r.get(k, "")) for k in keys)
        groups.setdefault(gk, []).append(r)

    metric_keys = ["test_mean_score", "mean_n_completed_tasks", "mean_time"]
    summary_rows: List[Dict[str, Any]] = []
    for gk, runs in groups.items():
        if not runs:
            continue
        row: Dict[str, Any] = {
            "cfg_idx": runs[0].get("cfg_idx"),
            "n_seeds": len(runs),
        }
        for k, v in zip(keys, gk):
            row[k] = v
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
                row[f"{mk}_mean"] = m
                row[f"{mk}_std"] = s
            else:
                row[f"{mk}_mean"] = float("nan")
                row[f"{mk}_std"] = float("nan")
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: r.get("test_mean_score_mean",
                                          float("-inf")), reverse=True)
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
        sr_mean = r.get("test_mean_score_mean", float("nan"))
        sr_std = r.get("test_mean_score_std", float("nan"))
        kv = ", ".join(f"{k}={r.get(k)}" for k in keys)
        print(f"  cfg {r.get('cfg_idx')}: SR={sr_mean:.4f} ± {sr_std:.4f} "
              f"({r.get('n_seeds')} seeds) | {kv}")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

_PROGRESS_FILE: Optional[pathlib.Path] = None


def _pr(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        import signal as _signal
        if hasattr(_signal, "SIGPIPE"):
            _signal.signal(_signal.SIGPIPE, _signal.SIG_DFL)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Grid search FlowPolicy kitchen (full cartesian)")
    parser.add_argument("--subset", default="default", choices=list(SUBSETS),
                        help="Preset subset HP yang di-sweep.")
    parser.add_argument("--override-hp", action="append", default=[],
                        help="Override satu HP dengan daftar nilai. "
                             "Format: key=v1:v2:v3. "
                             "Untuk list gunakan kurung siku: "
                             "policy.down_dims=[128,256,512]:[256,512,1024]. "
                             "Bisa diulang untuk HP berbeda.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/gridsearch")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--only-cfg", type=int, nargs="+", default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    # preprocessing sliding-window (opsional)
    parser.add_argument("--preprocess", action="store_true",
                        help="Aktifkan preprocessing sliding-window. Bila "
                             "--out-root default, out-root otomatis ke "
                             "data/outputs/gridsearch_preprocess.")
    parser.add_argument("--preprocess-window-ratio", type=float, default=None)
    parser.add_argument("--preprocess-stride", type=int, default=None)
    parser.add_argument("--preprocess-train-ratio", type=float, default=None)
    parser.add_argument("--preprocess-val-ratio", type=float, default=None)
    parser.add_argument("--preprocess-test-ratio", type=float, default=None)
    parser.add_argument("--preprocess-split-seed", type=int, default=None)
    args = parser.parse_args()

    # build subset
    subset = build_subset(args.subset, args.override_hp)

    # preprocess overrides
    preprocess_overrides: Optional[Dict[str, Any]] = None
    if args.preprocess:
        preprocess_overrides = {"enabled": True}
        if args.preprocess_window_ratio is not None:
            preprocess_overrides["window_ratio"] = args.preprocess_window_ratio
        if args.preprocess_stride is not None:
            preprocess_overrides["stride"] = args.preprocess_stride
        if args.preprocess_train_ratio is not None:
            preprocess_overrides["train_ratio"] = args.preprocess_train_ratio
        if args.preprocess_val_ratio is not None:
            preprocess_overrides["val_ratio"] = args.preprocess_val_ratio
        if args.preprocess_test_ratio is not None:
            preprocess_overrides["test_ratio"] = args.preprocess_test_ratio
        if args.preprocess_split_seed is not None:
            preprocess_overrides["split_seed"] = args.preprocess_split_seed

    # auto-suffix out-root untuk mode preprocess
    _default_out_root = "data/outputs/gridsearch"
    if args.preprocess and args.out_root == _default_out_root:
        args.out_root = _default_out_root + "_preprocess"

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

    global _PROGRESS_FILE
    _PROGRESS_FILE = progress_log

    enum_cfgs = enumerate_configs_grid(subset)

    # filter by only-cfg
    filtered = [
        (gidx, cfg) for (gidx, cfg) in enum_cfgs
        if (not args.only_cfg or gidx in args.only_cfg)
    ]
    total_planned = len(filtered) * len(args.seeds)

    # dump konfigurasi yang akan dijalankan
    cfg_dump = [
        {"cfg_idx": gidx, "sweep_subset": subset, "full_cfg": cfg,
         # highlight nilai subset untuk memudahkan plotting nanti
         "subset_values": {k: cfg[k] for k in subset.keys()}}
        for (gidx, cfg) in enum_cfgs
    ]
    with open(configs_json, "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, default=str)

    _pr(f"[grid] subset preset   : {args.subset}")
    _pr(f"[grid] subset HP ({len(subset)}): "
        f"{ {k: len(v) for k, v in subset.items()} }")
    _pr(f"[grid] total konfig    : {len(enum_cfgs)}")
    _pr(f"[grid] seeds           : {args.seeds}")
    _pr(f"[grid] total run       : {len(enum_cfgs) * len(args.seeds)} "
        f"(planned: {total_planned})")
    _pr(f"[grid] out-root        : {out_root}")
    _pr(f"[grid] configs.json    : {configs_json}")
    _pr(f"[grid] results.csv     : {results_csv}")
    if preprocess_overrides is not None:
        _pr(f"[grid] preprocessing   : ON -> {preprocess_overrides}")
    else:
        _pr(f"[grid] preprocessing   : OFF (pipeline episode-level)")

    if args.dry_run:
        _pr("[dry-run] list konfigurasi (subset values saja):")
        for (gidx, cfg) in filtered:
            valid, reason = is_cfg_valid(cfg)
            tag = "" if valid else f"  [INVALID: {reason}]"
            sv = {k: cfg[k] for k in subset.keys()}
            _pr(f"  cfg {gidx:04d}: {sv}{tag}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # CSV fieldnames tetap
    metric_cols = ["test_mean_score", "mean_n_completed_tasks", "mean_time"]
    subset_keys = list(subset.keys())
    fieldnames = [
        "cfg_idx", "seed",
        *subset_keys,
        *metric_cols,
        "status", "t_train_s", "t_infer_s",
        "run_dir", "timestamp",
    ]

    t_sweep_start = time.time()
    done = 0
    for (gidx, cfg) in filtered:
        for seed in args.seeds:
            done += 1
            elapsed = time.time() - t_sweep_start
            if args.max_minutes is not None:
                if elapsed / 60.0 >= args.max_minutes:
                    _pr(f"\n[grid] --max-minutes={args.max_minutes} tercapai "
                        f"({elapsed / 60.0:.1f} min), stop di {done-1}/"
                        f"{total_planned}.")
                    if not args.skip_summary:
                        write_summary(results_csv, summary_csv, subset)
                    return 0

            run_dir = (out_root / f"cfg_{gidx:04d}_seed{seed}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            infer_subdir = f"inference_ep{args.episodes}"
            mpath = metrics_path_for(run_dir, infer_subdir)

            sv = {k: cfg[k] for k in subset_keys}
            _pr(f"\n========== [{done}/{total_planned}] cfg {gidx:04d} "
                f"seed {seed} ==========")
            _pr(f"  subset   : {sv}")
            _pr(f"  run_dir  : {run_dir}")

            # validasi kombinasi
            valid, reason = is_cfg_valid(cfg)
            if not valid:
                _pr(f"  [skip invalid] {reason}")
                row = _base_row(gidx, seed, cfg, subset_keys, run_dir,
                                fieldnames, status="skipped_invalid")
                append_result_row(results_csv, row, fieldnames)
                continue

            # resume-by-metrics (tetap skip bila metrics.json sudah ada valid)
            if mpath.is_file():
                try:
                    metrics = read_metrics(mpath)
                    if "test_mean_score" in metrics:
                        sr = metrics["test_mean_score"]
                        _pr(f"  [skip-resume] metrics sudah ada: SR={sr:.4f}")
                        row = _fill_metrics_row(gidx, seed, cfg, subset_keys,
                                                run_dir, fieldnames,
                                                metrics=metrics,
                                                status="skip_resume",
                                                t_train=0.0, t_infer=0.0)
                        append_result_row(results_csv, row, fieldnames)
                        continue
                except Exception:
                    pass

            # ---- training ----
            t0 = time.time()
            ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
            if ckpt_path.is_file():
                _pr(f"  [skip-train] ckpt sudah ada: {ckpt_path}")
            else:
                train_cmd = build_train_cmd(
                    cfg, seed, run_dir, save_ckpt=True,
                    preprocess_overrides=preprocess_overrides,
                    resume=False)
                _pr(f"  [train] {' '.join(train_cmd)}")
                rc = run_subprocess(
                    train_cmd, run_dir / "train_stdout.log", env=env)
                if rc != 0 or not ckpt_path.is_file():
                    t_train = time.time() - t0
                    _pr(f"  [ERROR] training gagal (rc={rc}), skip inferensi.")
                    row = _base_row(gidx, seed, cfg, subset_keys, run_dir,
                                    fieldnames,
                                    status=f"train_failed_rc{rc}",
                                    t_train=t_train)
                    append_result_row(results_csv, row, fieldnames)
                    continue
            t_train = time.time() - t0

            # ---- inference (mini-eval untuk summary) ----
            t1 = time.time()
            infer_cmd = build_infer_cmd(
                ckpt_path, episodes=args.episodes, output_subdir=infer_subdir)
            _pr(f"  [infer] {' '.join(infer_cmd)}")
            rc = run_subprocess(
                infer_cmd, run_dir / "infer_stdout.log", env=env)
            t_infer = time.time() - t1

            metrics: Dict[str, Any] = {}
            if rc == 0 and mpath.is_file():
                try:
                    metrics = read_metrics(mpath)
                except Exception as exc:
                    _pr(f"  [WARN] gagal parse metrics: {exc}")
            else:
                _pr(f"  [ERROR] inferensi gagal (rc={rc})")

            status = "ok" if metrics else f"infer_failed_rc{rc}"
            row = _fill_metrics_row(gidx, seed, cfg, subset_keys, run_dir,
                                    fieldnames, metrics=metrics,
                                    status=status,
                                    t_train=t_train, t_infer=t_infer)
            append_result_row(results_csv, row, fieldnames)
            _pr(f"  [{status}] t_train={t_train:.0f}s t_infer={t_infer:.0f}s "
                f"elapsed={_fmt_hms(time.time() - t_sweep_start)}")

    if not args.skip_summary:
        write_summary(results_csv, summary_csv, subset)
    _pr(f"[done] Selesai. Total elapsed: "
        f"{_fmt_hms(time.time() - t_sweep_start)}")
    return 0


def _base_row(cfg_idx: int, seed: int, cfg: Dict[str, Any],
              subset_keys: List[str], run_dir: pathlib.Path,
              fieldnames: List[str], status: str,
              t_train: float = 0.0, t_infer: float = 0.0) -> Dict[str, Any]:
    row = {k: "" for k in fieldnames}
    row["cfg_idx"] = cfg_idx
    row["seed"] = seed
    for k in subset_keys:
        row[k] = _fmt_value(cfg[k]) if isinstance(cfg[k], list) else cfg[k]
    row["status"] = status
    row["t_train_s"] = round(float(t_train), 2)
    row["t_infer_s"] = round(float(t_infer), 2)
    row["run_dir"] = str(run_dir)
    row["timestamp"] = int(time.time())
    return row


def _fill_metrics_row(cfg_idx: int, seed: int, cfg: Dict[str, Any],
                      subset_keys: List[str], run_dir: pathlib.Path,
                      fieldnames: List[str], metrics: Dict[str, Any],
                      status: str, t_train: float, t_infer: float
                      ) -> Dict[str, Any]:
    row = _base_row(cfg_idx, seed, cfg, subset_keys, run_dir, fieldnames,
                    status=status, t_train=t_train, t_infer=t_infer)
    for mk in ("test_mean_score", "mean_n_completed_tasks", "mean_time"):
        v = metrics.get(mk)
        row[mk] = float(v) if v is not None else ""
    return row


if __name__ == "__main__":
    _here = pathlib.Path(__file__).resolve().parent
    _flowpolicy_pkg = _here.parent / "FlowPolicy"
    os.chdir(_flowpolicy_pkg)
    sys.exit(main())
