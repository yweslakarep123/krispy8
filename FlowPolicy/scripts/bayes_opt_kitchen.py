#!/usr/bin/env python3
"""Bayesian optimization untuk hyperparameter FlowPolicy — FrankaKitchen-v1.

**Konsep (sesuai literatur BO):** evaluasi objektif (train+infer) mahal; kita
ingin **belajar dari trial sebelumnya** dan **mengarahkan** percobaan berikutnya
ke wilayah HP yang menjanjikan (penyeimbangan eksplorasi vs eksploitasi),
bukan brute-force grid.

**Implementasi:** `Optuna TPESampler` (Tree-structured Parzen Estimator). TPE
bukan Gaussian Process + Expected Improvement secara eksplisit, tetapi termasuk
*sequential model-based optimization* / variasi BO yang umum dipakai di deep
learning: memisahkan kepadatan ``p(x | y baik)`` vs ``p(x | y buruk)`` dari
riwayat trial, lalu mengambil kandidat berikutnya yang memaksimalkan rasio
tersebut—peran setara *surrogate* + *acquisition* untuk ruang HP **kategorikal
dan campuran** (Bergstra et al., Hyperopt). Untuk ruang kontinu murni,
alternatif klasik GP+EI tersedia di ekosistem lain (BoTorch); di sini diskrit
dominan sehingga TPE adalah pilihan yang stabil.

Prasyarat: ``pip install optuna``

Contoh:
  cd FlowPolicy
  bash scripts/bayes_opt_kitchen.sh 0 --n-trials 20 --dry-run
  bash scripts/bayes_opt_kitchen.sh 0 --n-trials 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import warnings
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

TRIAL_CSV_FIELDNAMES = [
    "trial_number", "value", "state", "elapsed_s", "config_json",
]

# ---------------------------------------------------------------------------
# Ruang pencarian (diskrit) — sama dengan grid search lama
# ---------------------------------------------------------------------------

SEARCH_SPACE_BASE: Dict[str, List[Any]] = {
    "training.num_epochs":                           [500, 1000, 3000, 5000],
    "optimizer.lr":                                  [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size":                         [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    # Catatan: eps/delta harus < 1 agar CFM predict_action tidak bagi (1-num_t)^2 -> 0.
    "policy.Conditional_ConsistencyFM.eps":          [1e-2, 1e-3, 1e-4, 0.05],
    "policy.Conditional_ConsistencyFM.delta":        [1e-2, 1e-3, 1e-4, 0.05],
    "n_action_steps":                                [2, 4, 6, 8],
    "n_obs_steps":                                   [4, 6, 8, 16],
}

SEARCH_SPACE_MODEL: Dict[str, List[Any]] = {
    "policy.down_dims": [
        [128, 256, 512],
        [192, 384, 768],
        [256, 512, 1024],
        [384, 768, 1536],
    ],
    "policy.diffusion_step_embed_dim": [64, 128, 256, 512],
    "policy.kernel_size":              [3, 5, 7, 9],
    "policy.n_groups":                 [1, 2, 4, 8],
    "policy.encoder_output_dim":       [32, 64, 128, 256],
    "policy.state_mlp_size": [
        [128, 128],
        [256, 256],
        [512, 512],
        [256, 256, 256],
    ],
    "policy.action_clip":              [0.5, 1.0, 2.0, 5.0],
    "policy.encoder_use_layernorm":    [True, False],
    "policy.use_down_condition":       [True, False],
    "policy.use_mid_condition":        [True, False],
    "policy.use_up_condition":         [True, False],
}

SEARCH_SPACE_FULL: Dict[str, List[Any]] = {
    **SEARCH_SPACE_BASE,
    **SEARCH_SPACE_MODEL,
}

BASELINE: Dict[str, Any] = {
    "training.num_epochs":                           3000,
    "optimizer.lr":                                  1e-4,
    "dataloader.batch_size":                         128,
    "policy.Conditional_ConsistencyFM.num_segments": 2,
    "policy.Conditional_ConsistencyFM.eps":          1e-2,
    "policy.Conditional_ConsistencyFM.delta":        1e-2,
    "n_action_steps":                                4,
    "n_obs_steps":                                   4,
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

DEFAULT_SEEDS: List[int] = [0, 42, 101]
DEFAULT_EVAL_EPISODES: int = 50


def _pname(key: str) -> str:
    return key.replace(".", "__")


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
        return False, (
            f"n_groups={n_groups} tidak membagi semua channel di "
            f"down_dims={down_dims}")
    eps_c = float(cfg.get("policy.Conditional_ConsistencyFM.eps", 1e-2))
    if not (1e-9 < eps_c < 1.0):
        return False, (
            f"eps={eps_c} harus di (0,1); eps>=1 memicu ZeroDivisionError di "
            "predict_action ((1-num_t)^2 di penyebut).")
    dlt = float(cfg.get("policy.Conditional_ConsistencyFM.delta", 1e-2))
    if not (1e-9 < dlt < 1.0):
        return False, (
            f"delta={dlt} harus di (0,1) untuk CFM stabil pada rollout eval.")
    return True, None


# #region agent log
_DBG_LOG = pathlib.Path(
    "/home/daffa/Documents/krispy8/.cursor/debug-a60755.log")


def _dbg_train_fail(
        *,
        trial_n: int,
        seed: int,
        rc: int,
        ckpt_ok: bool,
        train_log: pathlib.Path) -> None:
    tail = ""
    try:
        if train_log.is_file():
            with open(train_log, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-100:])
    except OSError:
        tail = ""
    tl = tail.lower()
    payload = {
        "sessionId": "a60755",
        "runId": "bayes_bo",
        "hypothesisId": "train_fail_ndjson",
        "location": "bayes_opt_kitchen.py:objective",
        "message": "train subprocess failed",
        "data": {
            "trial": trial_n,
            "seed": seed,
            "rc": rc,
            "ckpt_exists": ckpt_ok,
            "H_eps_one": "eps=1" in tail or "eps=1.0" in tail,
            "H_zero_div": "zerodivisionerror" in tl,
            "H_cuda_oom": "cuda out of memory" in tl,
            "H_traceback": "traceback" in tl,
            "tail_suffix": tail[-4000:] if tail else "",
        },
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DBG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DBG_LOG, "a", encoding="utf-8") as df:
            df.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


# #endregion


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


def _apply_vram_safe_batch(cfg: Dict[str, Any]
                           ) -> Tuple[Dict[str, Any], Optional[str]]:
    out = dict(cfg)
    dd = list(out.get("policy.down_dims", [256, 512, 1024]))
    cap = _dataloader_batch_cap_for_vram(dd)
    req = int(out.get("dataloader.batch_size", 128))
    if req > cap:
        out["dataloader.batch_size"] = cap
        return out, (
            f"dataloader.batch_size {req} → {cap} "
            f"(max policy.down_dims={max(dd)})")
    return out, None


def suggest_cfg(trial: Any) -> Dict[str, Any]:
    """Bangun cfg penuh dari satu Optuna trial (semua HP di SEARCH_SPACE_FULL).

    Semua dimensi memakai ``suggest_int(..., 0, n-1)`` + pemetaan lokal, **bukan**
    ``suggest_categorical`` langsung pada nilai float/list. Alasannya: SQLite
    studi Optuna menyimpan ``CategoricalDistribution`` per nama parameter; bila
    daftar pilihan berubah (mis. mengganti ``1.0``), Optuna memunculkan
    ``ValueError: CategoricalDistribution does not support dynamic value space``.
    Indeks integer + mapping menjaga distribusi tetap stabil saat nilai
    kategorikal di repo diubah.
    """
    cfg = dict(BASELINE)
    for key, choices in SEARCH_SPACE_FULL.items():
        pname = _pname(key)
        n = len(choices)
        idx = trial.suggest_int(f"{pname}_idx", 0, n - 1)
        val = choices[idx]
        cfg[key] = list(val) if isinstance(val, list) else val
    return cfg


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
        "exp_name=bayes_opt",
        f"logging.name=bayes_{run_dir.name}",
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
        "--no-video",
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


def _pr(msg: str, log_file: Optional[pathlib.Path]) -> None:
    print(msg, flush=True)
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    try:
        import signal as _signal
        if hasattr(_signal, "SIGPIPE"):
            _signal.signal(_signal.SIGPIPE, _signal.SIG_DFL)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Bayesian optimization (Optuna TPE) untuk FlowPolicy kitchen")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Jumlah trial Optuna (setiap trial = semua seed).")
    parser.add_argument("--study-name", type=str, default="flowpolicy_kitchen_bo")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="URL storage Optuna, mis. sqlite:///abs/path/study.db. "
             "Kosong = otomatis ke <out-root>/study.db")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/bayes_opt")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-startup-trials", type=int, default=12,
                        help="Fase inisialisasi: trial acak sebelum model TPE "
                             "membimbing pilihan (setara eksplorasi awal BO).")
    parser.add_argument(
        "--no-tpe-multivariate",
        action="store_true",
        help="Nonaktifkan TPE multivariat (default: aktif). Multivariat "
             "memodelkan dependensi antar dimensi HP lebih kuat, mirip surrogate "
             "joint untuk beberapa HP sekaligus.")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument(
        "--no-mem-cap",
        action="store_true",
        help="Matikan penurunan otomatis batch untuk UNet lebar.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Aktifkan preprocessing sliding-window untuk semua train.")
    parser.add_argument("--preprocess-window-ratio", type=float, default=None)
    parser.add_argument("--preprocess-stride", type=int, default=None)
    parser.add_argument("--preprocess-train-ratio", type=float, default=None)
    parser.add_argument("--preprocess-val-ratio", type=float, default=None)
    parser.add_argument("--preprocess-test-ratio", type=float, default=None)
    parser.add_argument("--preprocess-split-seed", type=int, default=None)
    args = parser.parse_args()

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

    _default_out_root = "data/outputs/bayes_opt"
    if args.preprocess and args.out_root == _default_out_root:
        args.out_root = _default_out_root + "_preprocess"

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    assert flowpolicy_dir.is_dir(), (
        f"Tidak menemukan folder FlowPolicy/FlowPolicy di {flowpolicy_dir}")

    out_root = (flowpolicy_dir / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log = out_root / "progress.log"
    trials_csv = out_root / "trials.csv"
    best_json = out_root / "best_trial.json"

    if not args.storage:
        storage = f"sqlite:///{out_root / 'study.db'}"
    else:
        storage = args.storage

    # Header trials.csv sejak awal (termasuk dry-run) agar plot & tooling
    # menemukan file di out-root yang sama dengan training.
    if not trials_csv.is_file():
        with open(trials_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=TRIAL_CSV_FIELDNAMES).writeheader()

    if args.dry_run:
        rng = random.Random(0)

        class _DryTrial:
            def suggest_int(self, name: str, low: int, high: int) -> int:
                return rng.randint(low, high)

            def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
                return choices[rng.randint(0, len(choices) - 1)]

        _pr("[bo] dry-run (mock trial, tidak menulis study.db)", progress_log)
        for i in range(min(3, args.n_trials)):
            cfg = suggest_cfg(_DryTrial())
            valid, reason = is_cfg_valid(cfg)
            _pr(f"  sample {i}: valid={valid} {reason or ''} cfg={cfg}", progress_log)
        _pr("[bo] dry-run selesai.", progress_log)
        return 0

    import optuna  # noqa: PLC0415

    try:
        from optuna.exceptions import ExperimentalWarning as _OptExpWarn
        warnings.filterwarnings("ignore", category=_OptExpWarn)
    except Exception:  # pragma: no cover
        warnings.filterwarnings(
            "ignore", message=".*multivariate.*experimental.*")

    def _make_bo_sampler() -> Any:
        """TPESampler = BO berbasis kepadatan untuk HP campuran/kategorikal."""
        kw: Dict[str, Any] = {
            "seed": 0,
            "n_startup_trials": args.n_startup_trials,
            "consider_prior": True,
        }
        if not args.no_tpe_multivariate:
            kw["multivariate"] = True
        try:
            return optuna.samplers.TPESampler(**kw)
        except TypeError:
            kw.pop("multivariate", None)
            return optuna.samplers.TPESampler(**kw)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    infer_subdir = f"inference_ep{args.episodes}"
    t_sweep_start = time.time()

    sampler = _make_bo_sampler()
    _pr(f"[bo] sampler: TPE (BO berbasis kepadatan), multivariate="
        f"{not args.no_tpe_multivariate}", progress_log)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        if args.max_minutes is not None:
            if (time.time() - t_sweep_start) / 60.0 >= args.max_minutes:
                trial.study.stop()

        cfg = suggest_cfg(trial)
        valid, reason = is_cfg_valid(cfg)
        if not valid:
            _pr(f"[bo] trial {trial.number} PRUNED: {reason}", progress_log)
            raise optuna.TrialPruned()

        scores: List[float] = []
        for seed in args.seeds:
            if args.max_minutes is not None:
                if (time.time() - t_sweep_start) / 60.0 >= args.max_minutes:
                    trial.study.stop()
                    break

            run_dir = (out_root / f"trial_{trial.number:05d}_seed{seed}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            mpath = metrics_path_for(run_dir, infer_subdir)

            if mpath.is_file():
                try:
                    m = read_metrics(mpath)
                    if "test_mean_score" in m:
                        scores.append(float(m["test_mean_score"]))
                        _pr(f"[bo] trial {trial.number} seed {seed} skip-resume "
                            f"SR={m['test_mean_score']}", progress_log)
                        continue
                except Exception:
                    pass

            if args.no_mem_cap:
                train_cfg, mem_note = cfg, None
            else:
                train_cfg, mem_note = _apply_vram_safe_batch(cfg)
            if mem_note:
                _pr(f"  [mem-cap] {mem_note}", progress_log)

            ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
            train_cmd = build_train_cmd(
                train_cfg, seed, run_dir, save_ckpt=True,
                preprocess_overrides=preprocess_overrides, resume=False)
            _pr(f"[bo] trial {trial.number} seed {seed} train ...", progress_log)
            rc = run_subprocess(
                train_cmd, run_dir / "train_stdout.log", env=env)
            if rc != 0 or not ckpt_path.is_file():
                _pr(f"[bo] trial {trial.number} seed {seed} TRAIN FAIL rc={rc}",
                     progress_log)
                # #region agent log
                _dbg_train_fail(
                    trial_n=trial.number,
                    seed=seed,
                    rc=rc,
                    ckpt_ok=ckpt_path.is_file(),
                    train_log=run_dir / "train_stdout.log",
                )
                # #endregion
                scores.append(0.0)
                continue

            infer_cmd = build_infer_cmd(
                ckpt_path, episodes=args.episodes, output_subdir=infer_subdir)
            rc = run_subprocess(
                infer_cmd, run_dir / "infer_stdout.log", env=env)
            if rc != 0 or not mpath.is_file():
                _pr(f"[bo] trial {trial.number} seed {seed} INFER FAIL rc={rc}",
                     progress_log)
                scores.append(0.0)
                continue
            try:
                m = read_metrics(mpath)
                scores.append(float(m.get("test_mean_score", 0.0)))
            except Exception:
                scores.append(0.0)

        if not scores:
            return 0.0
        mean_sr = sum(scores) / len(scores)
        row: Dict[str, Any] = {
            "trial_number": trial.number,
            "value": mean_sr,
            "state": "COMPLETE",
            "elapsed_s": round(time.time() - t_sweep_start, 2),
            "config_json": json.dumps(cfg, default=str),
        }
        with open(trials_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRIAL_CSV_FIELDNAMES)
            w.writerow(row)

        _pr(f"[bo] trial {trial.number} value(mean SR)={mean_sr:.4f} "
            f"elapsed={_fmt_hms(time.time() - t_sweep_start)}", progress_log)
        return mean_sr

    _pr(f"[bo] study={args.study_name} storage={storage}", progress_log)
    _pr(f"[bo] n_trials={args.n_trials} seeds={args.seeds} out-root={out_root}",
        progress_log)
    _pr(f"[bo] VRAM batch cap: {'OFF' if args.no_mem_cap else 'ON'}", progress_log)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    try:
        best = study.best_trial
    except ValueError:
        _pr("[bo] selesai tanpa trial complete (semua prune/gagal).", progress_log)
        return 1
    out_best = {
        "number": best.number,
        "value": best.value,
        "params": best.params,
        "user_attrs": dict(best.user_attrs),
    }
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(out_best, f, indent=2, default=str)

    _pr(f"[bo] selesai. best trial={best.number} value={best.value}", progress_log)
    _pr(f"[bo] best_trial.json -> {best_json}", progress_log)
    return 0


if __name__ == "__main__":
    _here = pathlib.Path(__file__).resolve().parent
    _flowpolicy_pkg = _here.parent / "FlowPolicy"
    os.chdir(_flowpolicy_pkg)
    sys.exit(main())
