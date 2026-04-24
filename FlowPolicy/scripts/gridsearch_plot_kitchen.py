#!/usr/bin/env python3
"""Plot perbandingan Grid Search: success rate & inference latency.

Input:
  - `infer_results.csv` dari `gridsearch_infer_kitchen.py` (default), atau
  - `eval_results.csv` dari `gridsearch_eval_kitchen.py`, atau
  - `results.csv` dari `gridsearch_kitchen.py` (training sweep).
  - `configs.json` di out-root untuk mengetahui HP yang disweep.

Output (di <out-root>/plots/):
  - `grid_sr_marginal.png`       : subplot per HP, SR mean±std
                                   (marginalized atas HP lain).
  - `grid_latency_marginal.png`  : subplot per HP, latency mean±std.
  - `grid_pair_heatmap.png`      : heatmap SR untuk pair HP teratas
                                   (ranking berdasarkan varians marginal SR).
  - `grid_sr_vs_latency.png`     : scatter Pareto SR vs latency.
  - `grid_summary.csv`           : rekap mean±std per kombinasi.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple


def load_rows(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_subset_keys(configs_json: pathlib.Path) -> List[str]:
    with open(configs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return []
    return list(data[0].get("sweep_subset", {}).keys())


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_label(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_label(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def aggregate_marginal(rows: List[Dict[str, str]], subset_keys: List[str],
                       sr_key: str, lat_key: str, lat_scale_ms: bool
                       ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return map[hp_key][str(value)] -> stats."""
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for r in rows:
        if r.get("status", "") not in ("", "ok", "skip_resume"):
            continue
        sr = _to_float(r.get(sr_key))
        lat = _to_float(r.get(lat_key))
        if sr is None and lat is None:
            continue
        if lat is not None and lat_scale_ms:
            lat = lat * 1000.0
        for k in subset_keys:
            v = r.get(k, "")
            b = out.setdefault(k, {}).setdefault(str(v),
                                                 {"sr": [], "lat": []})
            if sr is not None:
                b["sr"].append(sr)
            if lat is not None:
                b["lat"].append(lat)
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for k, vm in out.items():
        result[k] = {}
        for vv, s in vm.items():
            result[k][vv] = {
                "sr_mean": (sum(s["sr"]) / len(s["sr"])
                            if s["sr"] else float("nan")),
                "sr_std": _std(s["sr"]),
                "lat_mean": (sum(s["lat"]) / len(s["lat"])
                             if s["lat"] else float("nan")),
                "lat_std": _std(s["lat"]),
                "n": max(len(s["sr"]), len(s["lat"])),
            }
    return result


def aggregate_joint(rows: List[Dict[str, str]], subset_keys: List[str],
                    sr_key: str, lat_key: str, lat_scale_ms: bool
                    ) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """Return map[tuple_of_values] -> stats."""
    out: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    for r in rows:
        if r.get("status", "") not in ("", "ok", "skip_resume"):
            continue
        sr = _to_float(r.get(sr_key))
        lat = _to_float(r.get(lat_key))
        if sr is None and lat is None:
            continue
        if lat is not None and lat_scale_ms:
            lat = lat * 1000.0
        key = tuple(r.get(k, "") for k in subset_keys)
        b = out.setdefault(key, {"sr": [], "lat": []})
        if sr is not None:
            b["sr"].append(sr)
        if lat is not None:
            b["lat"].append(lat)
    result: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for key, s in out.items():
        result[key] = {
            "sr_mean": (sum(s["sr"]) / len(s["sr"])
                        if s["sr"] else float("nan")),
            "sr_std": _std(s["sr"]),
            "lat_mean": (sum(s["lat"]) / len(s["lat"])
                         if s["lat"] else float("nan")),
            "lat_std": _std(s["lat"]),
            "n": max(len(s["sr"]), len(s["lat"])),
        }
    return result


def write_summary(agg_joint: Dict[Tuple[str, ...], Dict[str, float]],
                  subset_keys: List[str],
                  out_csv: pathlib.Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, s in agg_joint.items():
        row: Dict[str, Any] = dict(zip(subset_keys, key))
        row["n_seeds"] = s["n"]
        row["sr_mean"] = s["sr_mean"]
        row["sr_std"] = s["sr_std"]
        row["lat_mean_ms"] = s["lat_mean"]
        row["lat_std_ms"] = s["lat_std"]
        rows.append(row)
    rows.sort(key=lambda r: (-(r["sr_mean"] if not math.isnan(r["sr_mean"])
                                else float("-inf")),
                             r["lat_mean_ms"] if not math.isnan(r["lat_mean_ms"])
                                else float("inf")))
    fieldnames = [*subset_keys, "n_seeds", "sr_mean", "sr_std",
                  "lat_mean_ms", "lat_std_ms"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _ensure_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise SystemExit(
            "[plot] matplotlib belum terpasang. Install: pip install matplotlib\n"
            f"       (error asli: {exc})") from exc


def plot_marginal(agg_marginal: Dict[str, Dict[str, Dict[str, float]]],
                  subset_keys: List[str], out_dir: pathlib.Path,
                  title_suffix: str = "") -> Tuple[pathlib.Path, pathlib.Path]:
    plt = _ensure_mpl()

    hp_keys = [k for k in subset_keys if k in agg_marginal]
    n = len(hp_keys)
    if n == 0:
        raise SystemExit("[plot] tidak ada data marginal.")
    ncols = 2 if n >= 2 else 1
    nrows = (n + ncols - 1) // ncols
    out_dir.mkdir(parents=True, exist_ok=True)

    def _panel(ax, hp_key, series: str):
        vals = sorted(agg_marginal[hp_key].keys())
        x = list(range(len(vals)))
        if series == "sr":
            ys = [agg_marginal[hp_key][v]["sr_mean"] for v in vals]
            es = [agg_marginal[hp_key][v]["sr_std"] for v in vals]
            color = "tab:blue"
            ylab = "Success rate (marginal)"
        else:
            ys = [agg_marginal[hp_key][v]["lat_mean"] for v in vals]
            es = [agg_marginal[hp_key][v]["lat_std"] for v in vals]
            color = "tab:orange"
            ylab = "Latency ms (marginal)"
        ax.errorbar(x, ys, yerr=es, fmt="o-", color=color, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=20, ha="right")
        ax.set_xlabel(hp_key)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    fig_sr, axes_sr = plt.subplots(nrows, ncols,
                                   figsize=(5.2 * ncols, 3.6 * nrows),
                                   squeeze=False)
    axes_flat = axes_sr.flatten()
    for ax, hp in zip(axes_flat, hp_keys):
        _panel(ax, hp, "sr")
    for ax in axes_flat[len(hp_keys):]:
        ax.axis("off")
    fig_sr.suptitle(f"Grid search: Success rate marginal vs HP {title_suffix}".strip())
    fig_sr.tight_layout(rect=(0, 0, 1, 0.97))
    p_sr = out_dir / "grid_sr_marginal.png"
    fig_sr.savefig(p_sr, dpi=140)
    plt.close(fig_sr)

    fig_lat, axes_lat = plt.subplots(nrows, ncols,
                                     figsize=(5.2 * ncols, 3.6 * nrows),
                                     squeeze=False)
    axes_flat = axes_lat.flatten()
    for ax, hp in zip(axes_flat, hp_keys):
        _panel(ax, hp, "lat")
    for ax in axes_flat[len(hp_keys):]:
        ax.axis("off")
    fig_lat.suptitle(
        f"Grid search: Inference latency marginal vs HP {title_suffix}".strip())
    fig_lat.tight_layout(rect=(0, 0, 1, 0.97))
    p_lat = out_dir / "grid_latency_marginal.png"
    fig_lat.savefig(p_lat, dpi=140)
    plt.close(fig_lat)
    return p_sr, p_lat


def pick_top2_hp(agg_marginal: Dict[str, Dict[str, Dict[str, float]]],
                 subset_keys: List[str]) -> List[str]:
    """Pilih 2 HP dengan varians marginal SR terbesar (dampak terbesar)."""
    scored: List[Tuple[str, float]] = []
    for k in subset_keys:
        if k not in agg_marginal:
            continue
        srs = [s["sr_mean"] for s in agg_marginal[k].values()
               if not math.isnan(s["sr_mean"])]
        if len(srs) < 2:
            continue
        mean = sum(srs) / len(srs)
        var = sum((x - mean) ** 2 for x in srs) / len(srs)
        scored.append((k, var))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in scored[:2]]


def plot_pair_heatmap(rows: List[Dict[str, str]], subset_keys: List[str],
                      top2: List[str], sr_key: str,
                      out_dir: pathlib.Path) -> Optional[pathlib.Path]:
    if len(top2) < 2:
        print("[plot] tidak cukup HP (<2) untuk heatmap, dilewati.")
        return None
    plt = _ensure_mpl()
    import numpy as np  # shipped dengan matplotlib
    a, b = top2
    cells: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        if r.get("status", "") not in ("", "ok", "skip_resume"):
            continue
        sr = _to_float(r.get(sr_key))
        if sr is None:
            continue
        cells.setdefault((r.get(a, ""), r.get(b, "")), []).append(sr)
    if not cells:
        return None
    xs = sorted({x for (x, _) in cells})
    ys = sorted({y for (_, y) in cells})
    grid = np.full((len(ys), len(xs)), float("nan"))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vals = cells.get((x, y))
            if vals:
                grid[i, j] = sum(vals) / len(vals)

    fig, ax = plt.subplots(figsize=(1.2 * len(xs) + 3, 1.0 * len(ys) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="SR mean")
    ax.set_title(f"Grid search SR heatmap: {a} x {b}")
    fig.tight_layout()
    p = out_dir / "grid_pair_heatmap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_pareto(agg_joint: Dict[Tuple[str, ...], Dict[str, float]],
                subset_keys: List[str], out_dir: pathlib.Path
                ) -> pathlib.Path:
    plt = _ensure_mpl()
    xs, ys, labels = [], [], []
    for i, (key, s) in enumerate(agg_joint.items()):
        if math.isnan(s["sr_mean"]) or math.isnan(s["lat_mean"]):
            continue
        xs.append(s["lat_mean"])
        ys.append(s["sr_mean"])
        labels.append(i)
    if not xs:
        raise SystemExit("[plot] tidak ada data untuk scatter.")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(xs, ys, s=18, alpha=0.7)
    # Beri label 5 titik teratas SR
    top = sorted(zip(ys, xs, labels), reverse=True)[:5]
    for (y, x, lab) in top:
        ax.annotate(f"#{lab}", (x, y), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Success rate (mean)")
    ax.set_title("Grid search: SR vs latency (tiap titik = 1 kombinasi HP)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "grid_sr_vs_latency.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> int:
    _here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot Grid Search")
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/gridsearch")
    parser.add_argument("--preprocess", action="store_true",
                        help="Bila diset dan --out-root default, alihkan ke "
                             "data/outputs/gridsearch_preprocess.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Override path CSV input.")
    parser.add_argument("--metric-source", choices=["infer", "eval", "search"],
                        default="infer")
    args = parser.parse_args()

    default_root = "data/outputs/gridsearch"
    if args.preprocess and args.out_root == default_root:
        args.out_root = default_root + "_preprocess"

    flowpolicy_dir = (_here.parent / "FlowPolicy").resolve()
    out_root = (flowpolicy_dir / args.out_root).resolve()
    configs_json = out_root / "configs.json"
    if not configs_json.is_file():
        print(f"ERROR: configs.json tidak ada di {out_root}", file=sys.stderr)
        return 2
    subset_keys = load_subset_keys(configs_json)
    if not subset_keys:
        print(f"ERROR: configs.json kosong atau tidak punya sweep_subset",
              file=sys.stderr)
        return 3

    if args.csv:
        csv_path = pathlib.Path(args.csv).expanduser().resolve()
    else:
        csv_path = out_root / {
            "infer": "infer_results.csv",
            "eval": "eval_results.csv",
            "search": "results.csv",
        }[args.metric_source]

    if not csv_path.is_file():
        print(f"ERROR: CSV tidak ditemukan: {csv_path}", file=sys.stderr)
        return 4
    rows = load_rows(csv_path)
    if not rows:
        print(f"ERROR: CSV kosong: {csv_path}", file=sys.stderr)
        return 5

    headers = rows[0].keys()
    sr_key = "test_mean_score"
    if "mean_time_s" in headers:
        lat_key, lat_scale_ms = "mean_time_s", True
    elif "mean_time" in headers:
        lat_key, lat_scale_ms = "mean_time", True
    elif "mean_time_ms" in headers:
        lat_key, lat_scale_ms = "mean_time_ms", False
    else:
        print(f"ERROR: tidak menemukan kolom latency di {csv_path}",
              file=sys.stderr)
        return 6

    agg_marginal = aggregate_marginal(rows, subset_keys, sr_key, lat_key,
                                      lat_scale_ms)
    agg_joint = aggregate_joint(rows, subset_keys, sr_key, lat_key,
                                lat_scale_ms)

    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_summary(agg_joint, subset_keys, plots_dir / "grid_summary.csv")
    p_sr, p_lat = plot_marginal(agg_marginal, subset_keys, plots_dir,
                                title_suffix=f"[{csv_path.name}]")
    top2 = pick_top2_hp(agg_marginal, subset_keys)
    p_heat = plot_pair_heatmap(rows, subset_keys, top2, sr_key, plots_dir)
    p_pareto = plot_pareto(agg_joint, subset_keys, plots_dir)

    print(f"[plot] {p_sr}")
    print(f"[plot] {p_lat}")
    if p_heat:
        print(f"[plot] {p_heat}  (pair: {top2})")
    print(f"[plot] {p_pareto}")
    print(f"[plot] summary CSV: {plots_dir / 'grid_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
