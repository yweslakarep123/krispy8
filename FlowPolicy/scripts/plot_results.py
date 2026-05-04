#!/usr/bin/env python3
"""Plot ringkasan eksperimen ke ``plots/`` (PNG + PDF)."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Any, Dict, List, Tuple


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[plot_results] matplotlib tidak tersedia: {exc}", file=sys.stderr)
        return 0

    root = pathlib.Path(args.output_dir).expanduser().resolve()
    if not root.is_absolute():
        flow_pkg = pathlib.Path(__file__).resolve().parent.parent / "FlowPolicy"
        root = (flow_pkg / args.output_dir).resolve()

    results_csv = root / "results.csv"
    summary_csv = root / "summary.csv"
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not results_csv.is_file():
        print(f"[error] {results_csv}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(open(results_csv, "r", encoding="utf-8")))

    def _dedupe_latest(rows_in: List[Dict[str, str]]) -> List[Dict[str, str]]:
        best: Dict[Tuple[int, str, int, int], Tuple[int, Dict[str, str]]] = {}
        for r in rows_in:
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
    summary_rows: List[Dict[str, str]] = []
    if summary_csv.is_file():
        summary_rows = list(csv.DictReader(open(summary_csv, "r", encoding="utf-8")))

    def _f(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    # --- Plot 1: latency vs success_rate_k4 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for profile, marker in (
        ("standard", "o"),
        ("minimal", "s"),
        ("legacy_minimal", "^"),
        ("raw", "D"),
    ):
        xs, ys, cs = [], [], []
        for r in rows:
            if r.get("profile") != profile:
                continue
            if r.get("status") != "ok":
                continue
            xs.append(_f(r.get("mean_inference_latency_ms")))
            ys.append(_f(r.get("success_rate_k4")))
            cs.append(_f(r.get("cfg_idx")))
        if xs:
            ax.scatter(xs, ys, marker=marker, label=profile, alpha=0.7)
    ax.set_xlabel("mean_inference_latency_ms")
    ax.set_ylabel("success_rate_k4")
    ax.set_title("Trade-off scatter (per run)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"tradeoff_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: bar top configs (from summary) ---
    if summary_rows:
        top = summary_rows[: min(10, len(summary_rows))]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax_i, prof in enumerate(("standard", "minimal")):
            ax = axes[ax_i]
            sub = [r for r in top if r.get("profile") == prof]
            if not sub:
                ax.set_visible(False)
                continue
            labels = [f"c{r['cfg_idx']}" for r in sub]
            x = np.arange(len(sub))
            w = 0.2
            for j, k in enumerate(range(1, 5)):
                vals = [_f(r.get(f"success_rate_k{k}_mean")) for r in sub]
                ax.bar(x + (j - 1.5) * w, vals, width=w, label=f"k{k}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("success rate (mean over seeds)")
            ax.set_title(f"Top summary — {prof}")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle("Success rate per sub-task (summary)")
        for ext in ("png", "pdf"):
            fig.savefig(plot_dir / f"success_rate_bar.{ext}", bbox_inches="tight")
        plt.close(fig)

    # --- Plot 3: box-ish bar for top-5 cfg variance across runs ---
    if summary_rows:
        top5 = summary_rows[: min(5, len(summary_rows))]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"{r['profile'][:3]}/c{r['cfg_idx']}/f{r['fold']}" for r in top5]
        ys = [_f(r.get("success_rate_k4_mean")) for r in top5]
        es = [_f(r.get("success_rate_k4_std")) for r in top5]
        ax.barh(labels[::-1], ys[::-1], xerr=es[::-1], capsize=3, alpha=0.75)
        ax.set_xlabel("success_rate_k4 (mean ± std across seeds)")
        ax.set_title("Top-5 summary groups (error bar = seed std)")
        ax.grid(True, axis="x", alpha=0.3)
        for ext in ("png", "pdf"):
            fig.savefig(plot_dir / f"cv_fold_variance.{ext}", bbox_inches="tight")
        plt.close(fig)

    # --- Plot 4: simple sensitivity: cfg_idx vs trade_off_k4_per_ms ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile, marker in (
        ("standard", "o"),
        ("minimal", "s"),
        ("legacy_minimal", "^"),
        ("raw", "D"),
    ):
        xs, ys = [], []
        for r in rows:
            if r.get("profile") != profile or r.get("status") != "ok":
                continue
            xs.append(_f(r.get("cfg_idx")))
            ys.append(_f(r.get("trade_off_k4_per_ms")))
        if xs:
            ax.scatter(xs, ys, marker=marker, alpha=0.6, label=profile)
    ax.set_xlabel("cfg_idx")
    ax.set_ylabel("trade_off_k4_per_ms")
    ax.set_title("Hyperparameter configs vs trade-off (raw runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"hyperparam_sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_results] wrote PNG/PDF under {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
