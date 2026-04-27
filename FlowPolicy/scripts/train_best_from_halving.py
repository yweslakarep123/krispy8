#!/usr/bin/env python3
"""Alias: best_trial hasil Halving (path & flavor default)."""
from __future__ import annotations

import sys
from importlib import util
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    tbt = here / "train_best_trial.py"
    spec = util.spec_from_file_location("train_best_trial_mod", tbt)
    if spec is None or spec.loader is None:
        raise SystemExit("train_best_trial.py tidak ditemukan")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rest = list(sys.argv[1:])
    is_help = "-h" in rest or "--help" in rest
    if not is_help:
        if "--best-json" not in rest:
            rest = ["--best-json", "data/outputs/halving_search/best_trial.json"] + rest
        if "--flavor" not in rest:
            rest = rest + ["--flavor", "halving"]
    old = sys.argv
    sys.argv = [sys.argv[0]] + rest
    try:
        return int(mod.main())
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
