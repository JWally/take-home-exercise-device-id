"""Candidate pairs, pair features and labels for all 65k sessions -> research/cache/{X,meta}.parquet.

Labels: 1 same labeled device, 0 different labeled devices or labeled vs unlabeled (guaranteed different),
-1 both unlabeled (unknown)."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linker"))
import blocks  # noqa: E402
import features  # noqa: E402
import harness  # noqa: E402

CACHE = Path(__file__).parent / "cache"


def main():
    CACHE.mkdir(exist_ok=True)
    t = time.time()
    df = harness.load()
    c = blocks.candidates(df)
    a, b = blocks.decode(c)
    print("candidates", len(a), round(time.time() - t, 1))
    P = features.prepare(df)
    X = pd.concat(
        [
            features.pair_features(P, a[i : i + 500000], b[i : i + 500000])
            for i in range(0, len(a), 500000)
        ],
        ignore_index=True,
    )
    print("features", X.shape, round(time.time() - t, 1))
    truth = df.truth.to_numpy(object)
    la, lb = pd.notna(truth[a]), pd.notna(truth[b])
    y = np.full(len(a), -1, np.int8)
    both = la & lb
    y[both] = (truth[a][both] == truth[b][both]).astype(np.int8)
    y[la ^ lb] = 0
    meta = pd.DataFrame(dict(a=a, b=b, y=y))
    X.to_parquet(CACHE / "X.parquet")
    meta.to_parquet(CACHE / "meta.parquet")
    print("done", round(time.time() - t, 1), meta.y.value_counts().to_dict())


if __name__ == "__main__":
    main()
