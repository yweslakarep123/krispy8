#!/usr/bin/env python3
"""Episode-level K-fold splits for Kitchen-Complete-v2 style datasets (~19 ep).

Holds out exactly ``n_test`` episode(s) for a fixed test set (never used in CV
train/val). Remaining episodes are partitioned into ``n_folds`` with shuffled
consecutive chunks (approx. equal validation size per fold).

Example output entry::

    {
      "fold": 0,
      "train_episodes": [0, 1, 2, ...],
      "val_episodes": [3, 4, 5],
      "test_episodes": [18]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _k_fold_episode_indices(
    pool: np.ndarray, n_folds: int, cv_seed: int
) -> List[tuple]:
    """Shuffle ``pool`` then assign consecutive chunks as val (one fold each)."""
    rng = np.random.default_rng(cv_seed)
    perm = pool.astype(np.int64).copy()
    rng.shuffle(perm)
    n = len(perm)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=np.int64)
    fold_sizes[: n % n_folds] += 1
    out: List[tuple] = []
    cur = 0
    for fs in fold_sizes:
        fs = int(fs)
        val = perm[cur : cur + fs]
        train = np.concatenate([perm[:cur], perm[cur + fs :]])
        cur += fs
        out.append(
            (
                sorted(train.astype(int).tolist()),
                sorted(val.astype(int).tolist()),
            )
        )
    return out


def build_cv_splits(
    n_episodes: int = 19,
    n_folds: int = 5,
    n_test: int = 1,
    cv_seed: int = 42,
    test_seed: int = 0,
) -> Dict[str, Any]:
    """Return dict with test_episodes and list of fold dicts."""
    if n_episodes < n_folds + n_test:
        raise ValueError(
            f"n_episodes ({n_episodes}) must be >= n_folds + n_test "
            f"({n_folds + n_test})"
        )
    rng_test = np.random.default_rng(test_seed)
    test_episodes = sorted(
        rng_test.choice(n_episodes, size=n_test, replace=False).tolist()
    )
    pool = np.array(
        [i for i in range(n_episodes) if i not in set(test_episodes)],
        dtype=np.int64,
    )
    if len(pool) < n_folds:
        raise ValueError("CV pool smaller than n_folds")

    fold_pairs = _k_fold_episode_indices(pool, n_folds, cv_seed)
    folds: List[Dict[str, Any]] = []
    for fold_idx, (train_episodes, val_episodes) in enumerate(fold_pairs):
        folds.append(
            {
                "fold": fold_idx,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "test_episodes": list(test_episodes),
            }
        )

    return {
        "n_episodes": n_episodes,
        "n_folds": n_folds,
        "n_test": n_test,
        "cv_seed": cv_seed,
        "test_seed": test_seed,
        "test_episodes": test_episodes,
        "folds": folds,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate episode-level CV splits JSON")
    p.add_argument("--n-episodes", type=int, default=19)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-test", type=int, default=1)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--test-seed", type=int, default=0)
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="cv_splits.json",
        help="Output JSON path",
    )
    args = p.parse_args()
    data = build_cv_splits(
        n_episodes=args.n_episodes,
        n_folds=args.n_folds,
        n_test=args.n_test,
        cv_seed=args.cv_seed,
        test_seed=args.test_seed,
    )
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
