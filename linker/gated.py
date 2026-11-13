"""Public API for the learned-gate partitioner (fork B).

    models, cols = fit_gates(df, a, b, s_oof, truth, fold_of)
    labels = partition(df, a, b, s, models, cols)            # deployment: mean of all gates
    labels = partition(df, a, b, s, models, cols, fold_of)   # CV: fold-respecting gates

df: sessions table (rows aligned with pair indices a, b); s: pair scores in [0, 1] that must
be out-of-fold for any labeled pair used in fit_gates. truth: device id per row or None
(lockbox / unlabeled rows must be None). fold_of: 'f0'..'f4' per row for labeled CV rows,
None otherwise. CV-selected defaults: floor=0.1, thr=0.4, unl_thr=0.6, bag=5.
"""

import gate
import gate_run as G
import numpy as np


def fit_gates(
    df,
    a,
    b,
    s,
    truth,
    fold_of,
    floor=0.1,
    unl_thr=0.6,
    bag=5,
    iters=300,
    depth=6,
    threads=5,
    seed=0,
    wbeta=None,
):
    T = gate.session_table(df)
    fold = np.array([f if f is not None else "unlabeled" for f in fold_of], dtype=object)
    ev = G.oracle_events(
        len(df),
        np.asarray(a),
        np.asarray(b),
        np.asarray(s, float),
        T,
        list(truth),
        fold,
        floor,
        unl_thr,
    )
    params = dict(
        iterations=iters,
        depth=depth,
        learning_rate=0.1,
        thread_count=threads,
        verbose=0,
        random_seed=seed,
        bag=bag,
        wbeta=wbeta,
    )
    return G.train_gates(ev, params)


def partition(df, a, b, s, models, cols, fold_of=None, thr=0.4, floor=0.1):
    T = gate.session_table(df)
    fold_of = list(fold_of) if fold_of is not None else [None] * len(df)
    return G.gated_partition(
        len(df),
        np.asarray(a),
        np.asarray(b),
        np.asarray(s, float),
        T,
        models,
        cols,
        fold_of,
        thr,
        floor,
    )
