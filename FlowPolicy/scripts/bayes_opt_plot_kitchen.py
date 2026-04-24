#!/usr/bin/env python3
"""Plot ringkas riwayat Bayesian optimization dari ``trials.csv`` atau ``study.db``."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from typing import List, Tuple


# #region agent log
def _agent_dbg_plot(
        hypothesis_id: str,
        location: str,
        message: str,
        data: dict) -> None:
    dbg = pathlib.Path(
        "/home/daffa/Documents/krispy8/.cursor/debug-a60755.log")
    payload = {
        "sessionId": "a60755",
        "runId": "bayes_plot",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        dbg.parent.mkdir(parents=True, exist_ok=True)
        with open(dbg, "a", encoding="utf-8") as df:
            df.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


# #endregion


def _load_trials_from_csv(csv_path: pathlib.Path) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append((int(row["trial_number"]), float(row["value"])))
            except (KeyError, ValueError):
                continue
    return out


def _load_trials_from_study(study_db: pathlib.Path,
                           study_name: str) -> List[Tuple[int, float]]:
    import optuna
    from optuna.trial import TrialState

    storage = f"sqlite:///{study_db.resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    out: List[Tuple[int, float]] = []
    for t in study.trials:
        if t.state == TrialState.COMPLETE and t.value is not None:
            out.append((t.number, float(t.value)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials-csv",
        type=str,
        default="",
        help="Path ke trials.csv (default: <out-root>/trials.csv)")
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/outputs/bayes_opt",
        help="Root relatif ke FlowPolicy/ (dipakai jika --trials-csv kosong).")
    parser.add_argument(
        "--study-name",
        type=str,
        default="flowpolicy_kitchen_bo",
        help="Nama studi Optuna (untuk fallback baca study.db).")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    if not flowpolicy_dir.is_dir():
        print(f"Tidak ada {flowpolicy_dir}", file=sys.stderr)
        return 1

    out_dir_base = flowpolicy_dir / args.out_root

    if args.trials_csv:
        csv_path = pathlib.Path(args.trials_csv)
    else:
        csv_path = (out_dir_base / "trials.csv").resolve()

    study_db = csv_path.parent / "study.db"
    source = "csv"
    trials: List[Tuple[int, float]] = []

    if csv_path.is_file():
        trials = _load_trials_from_csv(csv_path)

    if not trials and study_db.is_file():
        try:
            trials = _load_trials_from_study(study_db, args.study_name)
            source = "study_db"
        except Exception as exc:
            # #region agent log
            _agent_dbg_plot(
                "H4_optuna_load_fail",
                "bayes_opt_plot_kitchen.py:main",
                "gagal load study",
                {"error": str(exc), "study_db": str(study_db)},
            )
            # #endregion

    if not trials:
        children: List[str] = []
        if out_dir_base.is_dir():
            children = sorted(p.name for p in out_dir_base.iterdir())[:30]
        # #region agent log
        _agent_dbg_plot(
            "H_missing_data",
            "bayes_opt_plot_kitchen.py:main",
            "tidak ada titik plot",
            {
                "csv_path": str(csv_path),
                "csv_exists": csv_path.is_file(),
                "study_db_exists": study_db.is_file(),
                "out_root": str(out_dir_base),
                "out_root_children_sample": children,
                "H1_csv_only_header": csv_path.is_file() and not trials,
                "H2_no_study_db": not study_db.is_file(),
                "H3_wrong_out_root": args.out_root,
            },
        )
        # #endregion
        print(
            "Tidak ada data trial untuk di-plot.\n"
            f"  Diharap: {csv_path}\n"
            "  - Jalankan BO dulu: bash scripts/bayes_opt_kitchen.sh 0 --n-trials 1\n"
            "  - Atau setelah dry-run, jalankan lagi bayes_opt (trials.csv kini "
            "selalu dibuat berisi header).\n"
            "  - Mode preprocess: bash scripts/bayes_opt_plot_kitchen.sh "
            "--out-root data/outputs/bayes_opt_preprocess\n"
            "  - Studi lain: --study-name ... --trials-csv /path/trials.csv",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Butuh: pip install matplotlib", file=sys.stderr)
        return 1

    trials.sort(key=lambda x: x[0])
    xs = [t[0] for t in trials]
    ys = [t[1] for t in trials]
    best_so_far: List[float] = []
    cur = float("-inf")
    for y in ys:
        cur = max(cur, y)
        best_so_far.append(cur)

    plot_out_parent = csv_path.parent if csv_path.is_file() else out_dir_base
    out_dir = plot_out_parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, "o-", alpha=0.6, label="trial value (mean SR)")
    ax.plot(xs, best_so_far, "-", linewidth=2, label="best so far")
    ax.set_xlabel("trial_number")
    ax.set_ylabel("test_mean_score (mean over seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"sumber data: {source}", fontsize=9, y=1.02)
    fig.tight_layout()
    outp = out_dir / "bayes_optimization_history.png"
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {outp} (sumber: {source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
