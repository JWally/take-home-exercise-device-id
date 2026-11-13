"""Device-grouped 5-fold out-of-fold pair scoring (the sealed lockbox never trains a model).

Every labeled pair is scored by models whose training excluded its endpoints' folds; pairs touching only
unlabeled or lockbox sessions get the mean of the 5 models. `link.py --train` uses this to fit the merge
gate on honest scores; `research/run_cv.py` uses it for the experiments.
"""

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402

K = 5


def pair_folds(df, meta):
    fold = harness.assign_folds(df).to_numpy(object)
    return fold[meta.a.to_numpy()], fold[meta.b.to_numpy()]


def oof_scores(X, meta, fa, fb, params):
    y = meta.y.to_numpy()
    known = y >= 0
    touches_lock = (fa == "lockbox") | (fb == "lockbox")
    preds = np.zeros((K, len(X)), np.float32)
    for k in range(K):
        fk = f"f{k}"
        tr = known & ~touches_lock & (fa != fk) & (fb != fk)
        m = CatBoostClassifier(**params)
        m.fit(X[tr], y[tr])
        preds[k] = m.predict_proba(X)[:, 1]
        print(f"  fold {k}: train={tr.sum()} pos={y[tr].sum()}", flush=True)
    use = np.zeros((K, len(X)), bool)
    for k in range(K):
        use[k] = (fa == f"f{k}") | (fb == f"f{k}")
    use[:, ~use.any(0)] = True
    return (preds * use).sum(0) / use.sum(0)
