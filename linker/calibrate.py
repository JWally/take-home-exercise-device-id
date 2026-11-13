"""Calibrate a per-session link confidence: P(session's predicted device holds no other device's session).

Signal: mean pair-model score between the session and its predicted cluster-mates (over scored candidate pairs).
Isotonic map fitted on the 5 CV folds' out-of-fold predictions, one map per --fpr-budget setting; held-out
check fits on some folds and scores the others. Writes linker/confidence.json from the research/ cache.

    python linker/calibrate.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "research" / "cache"
OUT = Path(__file__).parent / "confidence.json"
BUDGETS = {"0.5": 0.25, "1": 0.9, "2": 1.9}


def mean_in_cluster(pred, a, b, s):
    same = pred[a] == pred[b]
    tot = np.zeros(len(pred))
    cnt = np.zeros(len(pred))
    np.add.at(tot, a[same], s[same])
    np.add.at(tot, b[same], s[same])
    np.add.at(cnt, a[same], 1)
    np.add.at(cnt, b[same], 1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def labeled_rows(df, fold, pred, score):
    """(fold, score, correct) for every CV session that sits in a multi-session predicted device."""
    truth = df.truth.to_numpy(object)
    vis = fold == "unlabeled"
    rows = []
    for k in range(5):
        sc = fold == f"f{k}"
        sizes = Counter(pred[sc | vis].tolist())
        cells = Counter(zip(truth[sc].tolist(), pred[sc].tolist(), strict=False))
        for i in np.flatnonzero(sc):
            if sizes[pred[i]] > 1 and not np.isnan(score[i]):
                rows.append((k, score[i], sizes[pred[i]] == cells[(truth[i], pred[i])]))
    return pd.DataFrame(rows, columns=["fold", "score", "correct"])


def fit(R):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(R.score, R.correct.astype(float))
    return iso


def reliability(conf, correct, bins=(0, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0001)):
    d = pd.DataFrame(dict(conf=conf, correct=correct.astype(float)))
    d["bin"] = pd.cut(d.conf, bins, include_lowest=True)
    return d.groupby("bin", observed=True).agg(
        sessions=("correct", "size"), predicted=("conf", "mean"), observed=("correct", "mean")
    )


def main():
    df = harness.load()
    fold = harness.assign_folds(df).to_numpy(object)
    m = pd.read_parquet(CACHE / "meta.parquet")
    a, b = m.a.to_numpy(), m.b.to_numpy()
    s = np.load(CACHE / "oof_ens.npy")
    maps = {}
    for budget, k in BUDGETS.items():
        pred = np.load(CACHE / f"opt_oof_ens_c{k}.npy")
        R = labeled_rows(df, fold, pred, mean_in_cluster(pred, a, b, s))
        # held-out check: fit on 3 folds, score the other 2 (both directions)
        held = []
        for tr in ([0, 1, 2], [2, 3, 4]):
            te = R[~R.fold.isin(tr)]
            iso = fit(R[R.fold.isin(tr)])
            held.append(pd.DataFrame(dict(conf=iso.predict(te.score), correct=te.correct)))
        H = pd.concat(held)
        brier = float(((H.conf - H.correct) ** 2).mean())
        base = float(((H.correct.mean() - H.correct) ** 2).mean())
        print(
            f"\n--fpr-budget {budget}: {len(R)} linked CV sessions, {int((~R.correct).sum())} wrongly linked; "
            f"held-out Brier {brier:.4f} (constant baseline {base:.4f})"
        )
        print(reliability(H.conf, H.correct).to_string(float_format=lambda v: f"{v:.3f}"))
        iso = fit(R)
        maps[budget] = dict(
            x=[float(v) for v in iso.X_thresholds_], y=[float(v) for v in iso.y_thresholds_]
        )
    OUT.write_text(
        json.dumps(
            dict(
                signal="mean pair-model score to predicted cluster-mates (scored candidate pairs)",
                target="P(the session's predicted device contains no session of another device)",
                fitted_on="5-fold CV out-of-fold predictions (research/opt.py size-aware gate)",
                maps=maps,
            ),
            indent=1,
        )
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
