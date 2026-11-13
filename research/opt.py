"""Operating-point search for grade.py's objective: max recall subject to a false-merge budget (5-fold CV).

Fits the fold-respecting merge gates once on the averaged out-of-fold scores (cached), then replays the
partition for each setting and logs it to the leaderboard:
  g:<thr>   fixed gate threshold
  c:<k>     size-aware rule: accept iff p >= 1 / (1 + k * nA*nB / (nA+nB))   (shipped; k per budget)

    .venv/bin/python research/opt.py c:0.25 c:0.9 c:1.9 g:0.4
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "linker"))
sys.path.insert(0, str(Path(__file__).parent))
import board  # noqa: E402
import gated  # noqa: E402
import harness  # noqa: E402

CACHE = Path(__file__).parent / "cache"


def size_aware(k):
    def thr(f):
        return 1.0 / (1.0 + k * f["nA"] * f["nB"] / (f["nA"] + f["nB"]))

    return thr


def main(specs, scores="oof_ens"):
    df = harness.load()
    m = pd.read_parquet(CACHE / "meta.parquet")
    s = np.load(CACHE / f"{scores}.npy")
    a, b = m.a.to_numpy(), m.b.to_numpy()
    fold = harness.assign_folds(df).to_numpy(object)
    cv = pd.Series(fold).str.startswith("f").to_numpy()
    fold_of = [f if c else None for f, c in zip(fold, cv, strict=True)]
    gp = CACHE / f"gates_{scores}.pkl"
    if gp.exists():
        models, cols = pickle.load(open(gp, "rb"))
    else:
        truth = [t if c else None for t, c in zip(df.truth, cv, strict=True)]
        models, cols = gated.fit_gates(df, a, b, s, truth, fold_of, threads=14)
        pickle.dump((models, cols), open(gp, "wb"))
    for spec in specs:
        kind, v = spec.split(":")
        thr = float(v) if kind == "g" else size_aware(float(v))
        pred = gated.partition(df, a, b, s, models, cols, fold_of, thr=thr)
        np.save(CACHE / f"opt_{scores}_{spec.replace(':', '')}.npy", pred)
        print(board.score(pred, f"{scores} {spec}"), flush=True)
    board.show()


if __name__ == "__main__":
    main(sys.argv[1:])
