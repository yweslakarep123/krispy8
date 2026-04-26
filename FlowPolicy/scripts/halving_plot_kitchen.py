#!/usr/bin/env python3
"""Plot riwayat tuning dari trials.csv (HalvingRandomSearchCV)."""
from __future__ import annotations

import argparse
import csv
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_trials(csv_path: pathlib.Path) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                xs.append(int(row["trial_number"]))
                ys.append(float(row["value"]))
            except Exception:
                continue
    return xs, ys


def best_so_far(vals: List[float]) -> List[float]:
    out: List[float] = []
    cur = float("-inf")
    for v in vals:
        cur = max(cur, v)
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot history Halving search kitchen")
    ap.add_argument("--out-root", type=str, default="data/outputs/halving_search")
    ap.add_argument("--output", type=str, default="plots/halving_search_history.png")
    args = ap.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    flowpolicy_dir = script_dir.parent / "FlowPolicy"
    csv_path = (flowpolicy_dir / args.out_root / "trials.csv").resolve()
    out_path = (flowpolicy_dir / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"Tidak menemukan trials.csv: {csv_path}")

    xs, ys = read_trials(csv_path)
    if not xs:
        raise SystemExit(f"trials.csv kosong / tidak valid: {csv_path}")
    bs = best_so_far(ys)

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, "o-", alpha=0.65, label="trial value (mean SR)")
    plt.plot(xs, bs, "-", linewidth=2.2, label="best so far")
    plt.xlabel("Trial Number")
    plt.ylabel("Score (test_mean_score)")
    plt.title("HalvingRandomSearchCV History")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[plot] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
