#!/usr/bin/env python3
"""Buat plot perbandingan OFAT: success rate & inference latency vs
nilai tiap hyperparameter.

Input:
  - CSV hasil inferensi dari `ofat_infer_kitchen.py`
    (`infer_results.csv` atau `eval_results.csv`), atau
  - `results.csv` dari `ofat_search_kitchen.py` (punya kolom mean_time
    dan test_mean_score juga).

Output (di --out-root/plots/):
  - `ofat_sr_vs_hp.png`        : 8 subplot (2x4) SR mean±std vs nilai HP
  - `ofat_latency_vs_hp.png`   : 8 subplot (2x4) latency mean±std vs nilai HP
  - `ofat_sr_and_latency.png`  : 8 subplot gabungan (dual y-axis SR + latency)
  - `ofat_summary.csv`         : rekap mean±std per (swept_hp, swept_value)

Contoh:
  bash scripts/ofat_plot_kitchen.sh                              # pakai infer_results.csv default
  bash scripts/ofat_plot_kitchen.sh --preprocess                 # out-root = ofat_search_preprocess
  bash scripts/ofat_plot_kitchen.sh --csv path/ke/custom.csv
  bash scripts/ofat_plot_kitchen.sh --metric-source search       # pakai results.csv dari training OFAT
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple


_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ofat_search_kitchen import SEARCH_SPACE  # type: ignore  # noqa: E402


def load_rows(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_val(v: str, ref_values: List[Any]) -> Any:
    """Cocokkan string v ke salah satu value di SEARCH_SPACE agar urutan
    sumbu-X stabil sesuai definisi sweep."""
    if v is None or v == "":
        return None
    for ref in ref_values:
        if isinstance(ref, bool):
            if v.lower() == str(ref).lower():
                return ref
        elif isinstance(ref, int) and not isinstance(ref, bool):
            try:
                if int(float(v)) == ref and float(v).is_integer():
                    return ref
            except Exception:
                pass
        elif isinstance(ref, float):
            try:
                if math.isclose(float(v), float(ref), rel_tol=1e-6,
                                abs_tol=1e-12):
                    return ref
            except Exception:
                pass
        else:
            if str(v) == str(ref):
                return ref
    # fallback: coba float / int
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except Exception:
        return v


def aggregate(rows: List[Dict[str, str]], sr_key: str, lat_key: str,
              lat_scale_ms: bool) -> Dict[str, Dict[Any, Dict[str, float]]]:
    """Return map[hp_key][value] -> {sr_mean, sr_std, lat_mean, lat_std, n}."""
    bucket: Dict[str, Dict[Any, Dict[str, List[float]]]] = {}
    for r in rows:
        status = r.get("status", "")
        if status not in ("", "ok", "skip_resume"):
            continue
        hp_key = r.get("swept_hp")
        if hp_key not in SEARCH_SPACE:
            continue
        val = _parse_val(r.get("swept_value", ""), SEARCH_SPACE[hp_key])
        sr = _to_float(r.get(sr_key))
        lat = _to_float(r.get(lat_key))
        if sr is None and lat is None:
            continue
        if lat is not None and lat_scale_ms:
            # simpan di ms untuk plotting
            lat = lat * 1000.0
        b = bucket.setdefault(hp_key, {}).setdefault(val, {"sr": [], "lat": []})
        if sr is not None:
            b["sr"].append(sr)
        if lat is not None:
            b["lat"].append(lat)

    out: Dict[str, Dict[Any, Dict[str, float]]] = {}
    for hp_key, vm in bucket.items():
        out[hp_key] = {}
        for val, s in vm.items():
            sr_arr, lat_arr = s["sr"], s["lat"]
            out[hp_key][val] = {
                "sr_mean": sum(sr_arr) / len(sr_arr) if sr_arr else float("nan"),
                "sr_std": _std(sr_arr),
                "lat_mean": sum(lat_arr) / len(lat_arr) if lat_arr else float("nan"),
                "lat_std": _std(lat_arr),
                "n": max(len(sr_arr), len(lat_arr)),
            }
    return out


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _sort_values(hp_key: str, values: List[Any]) -> List[Any]:
    ref = SEARCH_SPACE[hp_key]
    order = {v: i for i, v in enumerate(ref)}
    return sorted(values,
                  key=lambda v: order.get(v, 1e9) if v in order
                  else float("inf"))


def _xlabel(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def write_summary_csv(agg: Dict[str, Dict[Any, Dict[str, float]]],
                      out_csv: pathlib.Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["swept_hp", "swept_value", "n_seeds",
                    "sr_mean", "sr_std", "latency_mean_ms", "latency_std_ms"])
        for hp_key in SEARCH_SPACE:
            if hp_key not in agg:
                continue
            for val in _sort_values(hp_key, list(agg[hp_key].keys())):
                s = agg[hp_key][val]
                w.writerow([hp_key, _xlabel(val), s["n"],
                            f"{s['sr_mean']:.6f}", f"{s['sr_std']:.6f}",
                            f"{s['lat_mean']:.6f}", f"{s['lat_std']:.6f}"])


def plot_all(agg: Dict[str, Dict[Any, Dict[str, float]]],
             out_dir: pathlib.Path, title_suffix: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "[plot] matplotlib belum terpasang di environment ini.\n"
            "       Install dengan: pip install matplotlib\n"
            f"       (error asli: {exc})") from exc

    hp_keys = [k for k in SEARCH_SPACE if k in agg]
    n = len(hp_keys)
    if n == 0:
        print("[plot] tidak ada data yang bisa diplot.")
        return
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    out_dir.mkdir(parents=True, exist_ok=True)

    def _panel(ax, hp_key, series: str):
        vals = _sort_values(hp_key, list(agg[hp_key].keys()))
        x = list(range(len(vals)))
        if series == "sr":
            ys = [agg[hp_key][v]["sr_mean"] for v in vals]
            es = [agg[hp_key][v]["sr_std"] for v in vals]
            color = "tab:blue"
            ylab = "Success rate"
        else:
            ys = [agg[hp_key][v]["lat_mean"] for v in vals]
            es = [agg[hp_key][v]["lat_std"] for v in vals]
            color = "tab:orange"
            ylab = "Latency (ms)"
        ax.errorbar(x, ys, yerr=es, fmt="o-", color=color, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_xlabel(v) for v in vals], rotation=20, ha="right")
        ax.set_xlabel(hp_key)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    # 1) SR vs HP (8 subplot)
    fig1, axes1 = plt.subplots(nrows, ncols,
                               figsize=(5.2 * ncols, 3.6 * nrows))
    axes1 = axes1.flatten() if hasattr(axes1, "flatten") else [axes1]
    for ax, hp in zip(axes1, hp_keys):
        _panel(ax, hp, "sr")
    for ax in axes1[len(hp_keys):]:
        ax.axis("off")
    fig1.suptitle(f"OFAT: Success rate vs hyperparameter {title_suffix}".strip())
    fig1.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = out_dir / "ofat_sr_vs_hp.png"
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)

    # 2) Latency vs HP (8 subplot)
    fig2, axes2 = plt.subplots(nrows, ncols,
                               figsize=(5.2 * ncols, 3.6 * nrows))
    axes2 = axes2.flatten() if hasattr(axes2, "flatten") else [axes2]
    for ax, hp in zip(axes2, hp_keys):
        _panel(ax, hp, "lat")
    for ax in axes2[len(hp_keys):]:
        ax.axis("off")
    fig2.suptitle(
        f"OFAT: Inference latency vs hyperparameter {title_suffix}".strip())
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    p2 = out_dir / "ofat_latency_vs_hp.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)

    # 3) Combined (dual y-axis per subplot)
    fig3, axes3 = plt.subplots(nrows, ncols,
                               figsize=(5.5 * ncols, 3.8 * nrows))
    axes3 = axes3.flatten() if hasattr(axes3, "flatten") else [axes3]
    for ax, hp in zip(axes3, hp_keys):
        vals = _sort_values(hp, list(agg[hp].keys()))
        x = list(range(len(vals)))
        sr = [agg[hp][v]["sr_mean"] for v in vals]
        sr_std = [agg[hp][v]["sr_std"] for v in vals]
        lat = [agg[hp][v]["lat_mean"] for v in vals]
        lat_std = [agg[hp][v]["lat_std"] for v in vals]
        ax.errorbar(x, sr, yerr=sr_std, fmt="o-", color="tab:blue",
                    label="SR", capsize=3)
        ax.set_xlabel(hp)
        ax.set_xticks(x)
        ax.set_xticklabels([_xlabel(v) for v in vals], rotation=20, ha="right")
        ax.set_ylabel("Success rate", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.errorbar(x, lat, yerr=lat_std, fmt="s--", color="tab:orange",
                     label="Latency", capsize=3)
        ax2.set_ylabel("Latency (ms)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
    for ax in axes3[len(hp_keys):]:
        ax.axis("off")
    fig3.suptitle(f"OFAT: SR (biru) & Latency ms (jingga) {title_suffix}".strip())
    fig3.tight_layout(rect=(0, 0, 1, 0.97))
    p3 = out_dir / "ofat_sr_and_latency.png"
    fig3.savefig(p3, dpi=140)
    plt.close(fig3)

    print(f"[plot] {p1}")
    print(f"[plot] {p2}")
    print(f"[plot] {p3}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot OFAT (SR & latency)")
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/ofat_search")
    parser.add_argument("--preprocess", action="store_true",
                        help="Bila diset dan --out-root default, alihkan "
                             "out-root ke data/outputs/ofat_search_preprocess.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Override path CSV input secara eksplisit. "
                             "Kalau diset, --metric-source diabaikan.")
    parser.add_argument("--metric-source", choices=["infer", "eval", "search"],
                        default="infer",
                        help="Sumber metrics. `infer` = infer_results.csv "
                             "(default, dari ofat_infer_kitchen.sh), "
                             "`eval` = eval_results.csv (ofat_eval_kitchen.sh), "
                             "`search` = results.csv (dari training sweep).")
    args = parser.parse_args()

    default_root = "data/outputs/ofat_search"
    if args.preprocess and args.out_root == default_root:
        args.out_root = default_root + "_preprocess"

    flowpolicy_dir = (_HERE.parent / "FlowPolicy").resolve()
    out_root = (flowpolicy_dir / args.out_root).resolve()

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
        print("Jalankan inferensi dulu (mis. `bash scripts/ofat_infer_kitchen.sh 0`)",
              file=sys.stderr)
        return 2

    rows = load_rows(csv_path)
    if not rows:
        print(f"ERROR: CSV kosong: {csv_path}", file=sys.stderr)
        return 3

    # Pilih kolom SR & latency: infer/eval pakai mean_time_s; search pakai mean_time
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
        return 4

    agg = aggregate(rows, sr_key=sr_key, lat_key=lat_key,
                    lat_scale_ms=lat_scale_ms)
    if not agg:
        print("ERROR: tidak ada baris valid (status=ok/skip_resume).",
              file=sys.stderr)
        return 5

    plots_dir = out_root / "plots"
    write_summary_csv(agg, plots_dir / "ofat_summary.csv")
    plot_all(agg, plots_dir,
             title_suffix=f"[{csv_path.name}]")
    print(f"[plot] summary CSV: {plots_dir / 'ofat_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
