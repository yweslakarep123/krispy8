#!/usr/bin/env python3
"""Plot ringkasan grid (mean success rate) + plot SR & latensi infer final."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 9


def _read_json_metrics(p: pathlib.Path) -> Tuple[Optional[float], Optional[float]]:
    with open(p, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = json.load(f)
    sr = d.get("test_mean_score")
    ms = d.get("mean_time")
    try:
        a = float(sr) if sr is not None else None
    except (TypeError, ValueError):
        a = None
    try:
        b = float(ms) if ms is not None else None
    except (TypeError, ValueError):
        b = None
    return a, b


def _load_summary(csv_path: pathlib.Path) -> Tuple[List[str], List[float]]:
    gids: List[str] = []
    sr: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                gids.append(str(int(row.get("grid_id", 0))))
            except (TypeError, ValueError):
                gids.append(str(row.get("grid_id", "?")))
            try:
                sr.append(float(row.get("mean_value", 0.0) or 0.0))
            except (TypeError, ValueError):
                sr.append(0.0)
    return gids, sr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot hasil grid search + infer final")
    p.add_argument(
        "--out-root",
        type=str,
        default="data/outputs/grid_search_kitchen",
    )
    p.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path grid_summary.csv (default: <out>/grid_summary.csv)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="plots/grid_search_kitchen_plots.png",
    )
    p.add_argument(
        "--final-metrics",
        nargs="*",
        default=(),
        help="Satu atau lebih path metrics.json hasil infer final (dengan video).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    fdir = (script_dir.parent / "FlowPolicy").resolve()
    oroot = fdir / args.out_root
    s_path = (oroot / "grid_summary.csv" if not args.summary else fdir / args.summary).resolve()
    out_path = (fdir / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_plots = 1 + (1 if args.final_metrics else 0)
    W = 12
    H = 4.5 * n_plots
    fig, axs = plt.subplots(n_plots, 1, figsize=(W, H), squeeze=False)
    ax0 = axs[0, 0]

    if s_path.is_file():
        gids, vals = _load_summary(s_path)
        if gids and vals:
            x = list(range(len(gids)))
            ax0.bar(
                [str(g) for g in gids],
                vals,
                color="tab:blue",
                alpha=0.8,
            )
            ax0.set_ylabel("Mean test_mean_score (6 runs)")
            ax0.set_xlabel("grid_id")
            ax0.set_title("Grid search — mean success rate per config")
        else:
            ax0.text(0.5, 0.5, f"Kosong: {s_path}", ha="center")
    else:
        ax0.text(0.5, 0.5, f"File tidak ditemukan: {s_path}", ha="center")

    ax0.grid(axis="y", alpha=0.3)
    plt.sca(ax0)
    plt.xticks(rotation=45, ha="right", fontsize=7)

    if args.final_metrics and n_plots > 1:
        a1 = axs[1, 0]
        names: List[str] = []
        ss: List[float] = []
        lts: List[float] = []  # ms
        for i, s in enumerate(args.final_metrics):
            p = (pathlib.Path(s).expanduser().resolve())
            if not p.is_file() and (fdir / s).is_file():
                p = fdir / s
            n = p.parent.name
            sra, tsec = _read_json_metrics(p)
            names.append(f"{i}:{n}")
            ss.append(float(sra) if sra is not None else 0.0)
            lts.append(
                (float(tsec) * 1000.0) if tsec is not None and tsec > 0 else 0.0
            )
        tw = 0.35
        r = [j - 0.2 for j in range(len(names))]
        a1b = a1.twinx()
        a1.bar(
            r,
            ss,
            width=tw,
            color="seagreen",
            alpha=0.9,
            label="success rate",
        )
        a1.set_ylabel("test_mean_score", color="seagreen")
        a1b.bar(
            [j + 0.2 for j in range(len(names))],
            lts,
            width=tw,
            color="coral",
            alpha=0.8,
            label="latency (ms)",
        )
        a1b.set_ylabel("mean_time (ms)", color="coral")
        a1.set_xlabel("run (metrics.json)")
        a1.set_title("Inferensi final: SR + latensi")
        a1.set_xticks(range(len(names)))
        a1.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[plot] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
