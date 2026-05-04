#!/usr/bin/env python3
"""Agregasi ``results.csv`` → ``summary.csv`` (mean ± std per cfg, profile, fold)."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys
from typing import Any, Dict, List, Tuple


METRIC_COLS = (
    "success_rate_k1",
    "success_rate_k2",
    "success_rate_k3",
    "success_rate_k4",
    "mean_inference_latency_ms",
    "std_inference_latency_ms",
    "trade_off",
    "trade_off_k4_per_ms",
    "train_loss_final",
    "val_loss_final",
)


def _f(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder eksperimen (berisi results.csv), relatif ke FlowPolicy/ atau absolut",
    )
    args = p.parse_args()

    root = pathlib.Path(args.output_dir).expanduser().resolve()
    if not root.is_absolute():
        flow_pkg = pathlib.Path(__file__).resolve().parent.parent / "FlowPolicy"
        root = (flow_pkg / args.output_dir).resolve()

    results_csv = root / "results.csv"
    summary_csv = root / "summary.csv"
    if not results_csv.is_file():
        print(f"[error] tidak ada {results_csv}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(open(results_csv, "r", encoding="utf-8")))
    if not rows:
        print("[error] results.csv kosong", file=sys.stderr)
        return 1

    # Beberapa baris per (cfg, profile, fold, seed) jika eksperimen diulang — ambil run terbaru.
    def _dedupe_latest(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        best: Dict[Tuple[int, str, int, int], Tuple[int, Dict[str, str]]] = {}
        for r in rows:
            try:
                key = (
                    int(r["cfg_idx"]),
                    str(r["profile"]),
                    int(r["fold"]),
                    int(r["seed"]),
                )
            except Exception:
                continue
            try:
                ts = int(r.get("timestamp") or 0)
            except Exception:
                ts = 0
            old = best.get(key)
            if old is None or ts >= old[0]:
                best[key] = (ts, r)
        return [t[1] for t in best.values()]

    rows = _dedupe_latest(rows)

    grouped: Dict[Tuple[int, str, int], List[Dict[str, str]]] = {}
    for r in rows:
        try:
            key = (int(r["cfg_idx"]), str(r["profile"]), int(r["fold"]))
        except Exception:
            continue
        grouped.setdefault(key, []).append(r)

    out_rows: List[Dict[str, Any]] = []
    for (cfg_idx, profile, fold), g in grouped.items():
        out: Dict[str, Any] = {
            "cfg_idx": cfg_idx,
            "profile": profile,
            "fold": fold,
            "n_seeds": len(g),
        }
        for mk in METRIC_COLS:
            vals = [_f(x.get(mk)) for x in g]
            vals = [v for v in vals if v == v and math.isfinite(v)]  # drop NaN
            if vals:
                out[f"{mk}_mean"] = statistics.mean(vals)
                out[f"{mk}_std"] = (
                    statistics.stdev(vals) if len(vals) > 1 else 0.0
                )
            else:
                out[f"{mk}_mean"] = float("nan")
                out[f"{mk}_std"] = float("nan")
        out_rows.append(out)

    sort_key = "trade_off_k4_per_ms_mean"
    if all(math.isnan(float(r.get(sort_key, float("nan")))) for r in out_rows):
        sort_key = "trade_off_mean"

    def _sort_val(r: Dict[str, Any]) -> float:
        v = r.get(sort_key, float("-inf"))
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("-inf")
        if not math.isfinite(x):
            return float("-inf")
        return x

    out_rows.sort(key=_sort_val, reverse=True)

    fieldnames = list(out_rows[0].keys())
    for r in out_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[summarize] wrote {summary_csv} ({len(out_rows)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
