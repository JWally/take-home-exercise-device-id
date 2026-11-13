"""Merge gate: fold-respecting training from an oracle replay, and the gated Kruskal partition."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
import gate  # noqa: E402

K = 5


def oracle_events(n, a, b, s, T, truth_cv, fold, floor, unl_thr):
    """truth_cv: device id for CV-fold sessions, None otherwise (lockbox treated as unlabeled)."""
    lab = [None if t is None else {t} for t in truth_cv]
    folds = [{fold[i]} if truth_cv[i] is not None else set() for i in range(n)]
    rows = []

    def decide(ra, rb, f, C):
        la, lb = lab[ra], lab[rb]
        if la and lb:
            y = int(la == lb)
        elif la or lb:
            y = 0
        else:
            y = -1
        ok = (f["s"] >= unl_thr) if y < 0 else bool(y)
        rows.append(dict(f, y=y, folds="".join(sorted(folds[ra] | folds[rb]))))
        if ok:
            keep, drop = (ra, rb) if len(C.members[ra]) >= len(C.members[rb]) else (rb, ra)
            lab[keep] = (la or set()) | (lb or set()) or None
            folds[keep] = folds[ra] | folds[rb]
        return ok

    gate.replay(n, a, b, s, T, decide, floor=floor)
    return pd.DataFrame(rows)


EDGE_X = None  # optional DataFrame of pair features indexed by global edge id


def with_edge(ev):
    X = ev.drop(columns=[c for c in ("y", "folds", "edge") if c in ev])
    if EDGE_X is not None:
        X = pd.concat(
            [
                X.reset_index(drop=True),
                EDGE_X.iloc[ev.edge.to_numpy()].reset_index(drop=True).add_prefix("x_"),
            ],
            axis=1,
        )
    return X


BAG = 1


class Bag:
    def __init__(self, ms):
        self.ms = ms

    def predict_proba(self, x):
        return np.mean([m.predict_proba(x) for m in self.ms], axis=0)


def train_gates(ev, params):
    global BAG
    BAG = params.get("bag", 1)
    beta = params.pop("wbeta", None)
    X = with_edge(ev)
    # optional grade.py-shaped weights: a true merge recovers nA*nB pairs, a false one taints nA+nB sessions
    w = (
        None
        if beta is None
        else np.where(ev.y == 1, X.nA * X.nB, X.nA + X.nB).astype(float) ** beta
    )
    models = []
    for k in range(K):
        m = (ev.y >= 0) & ~ev.folds.str.contains(f"f{k}")
        bag = []
        for sd in range(params.pop("bag", 1) if k == 0 else BAG):
            cb = CatBoostClassifier(**dict(params, random_seed=params["random_seed"] + sd))
            cb.fit(X[m], ev.y[m], sample_weight=None if w is None else w[m])
            bag.append(cb)
        models.append(Bag(bag))
        print(f"  gate {k}: n={m.sum()} pos={ev.y[m].sum()}", flush=True)
    return models, list(X.columns)


EDGE_ROW = None


def set_edge_features(X):
    global EDGE_X, EDGE_ROW
    EDGE_X = X
    EDGE_ROW = _RowView(X) if X is not None else None


class _RowView:
    def __init__(self, X):
        self.cols = {c: X[c].to_numpy() for c in X.columns}

    def __getitem__(self, e):
        return {c: v[e] for c, v in self.cols.items()}


def gated_partition(n, a, b, s, T, models, cols, fold_of, thr, floor, cache=None):
    folds = [{fold_of[i]} if fold_of[i] is not None else set() for i in range(n)]

    def decide(ra, rb, f, C):
        fs = folds[ra] | folds[rb]
        ks = [int(x[1]) for x in fs] or list(range(K))
        x = [f[c] if c in f else EDGE_ROW[f["edge"]][c[2:]] for c in cols]
        x = np.array([x], dtype=float)
        p = np.mean([models[k].predict_proba(x)[0, 1] for k in ks])
        if hasattr(thr, "accept"):
            ok = thr.accept(p, ra, rb, C, f)
        else:
            ok = p >= (thr(f) if callable(thr) else thr)
        if ok:
            folds[ra] = folds[rb] = fs
        return ok

    return gate.replay(n, a, b, s, T, decide, floor=floor)
