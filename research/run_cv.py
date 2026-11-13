"""Out-of-fold pair scores for the experiments: research/cache/X.parquet -> research/cache/oof_<tag>.npy.

.venv/bin/python research/run_cv.py --tag s0 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "linker"))
import cv  # noqa: E402
import harness  # noqa: E402

CACHE = Path(__file__).parent / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=14)
    args = ap.parse_args()
    df = harness.load()
    X = pd.read_parquet(CACHE / "X.parquet")
    meta = pd.read_parquet(CACHE / "meta.parquet")
    fa, fb = cv.pair_folds(df, meta)
    params = dict(
        iterations=800,
        depth=8,
        learning_rate=0.08,
        thread_count=args.threads,
        verbose=0,
        random_seed=args.seed,
    )
    np.save(CACHE / f"oof_{args.tag}.npy", cv.oof_scores(X, meta, fa, fb, params))


if __name__ == "__main__":
    main()
