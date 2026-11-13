"""Device linker: one command for new sessions, plus the training step that produces its model.

    python linker/link.py NEW.parquet OUTPUT.csv      # writes session_id,device_id for every original
                                                     # unlabeled session + every new session
    python linker/link.py --train                    # refits linker/model.joblib on all labels

Pipeline: blocking candidates -> pair features -> CatBoost pair scorer (seed-averaged) -> learned
Kruskal merge gate with a size-aware acceptance rule tuned to grade.py's objective: max recall
subject to a session false-positive budget (default 1%; see --fpr-budget).
Cookies and labels are never pair-model features. They are used only as certain links (shared cookie or
labeled device = same device) and cannot-links (README: original unlabeled sessions never belong to a labeled
device). New sessions may join either a labeled device or an unlabeled one.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import calibrate  # noqa: E402
import cv  # noqa: E402
import fastgate  # noqa: E402
import features  # noqa: E402
import gated  # noqa: E402
import harness  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(__file__).parent / "model.joblib"
CONFIDENCE = Path(__file__).parent / "confidence.json"
PAIR = dict(iterations=800, depth=8, learning_rate=0.08, verbose=0)
SEEDS = (0, 1)
THR = 0.4  # legacy fixed gate threshold stored in older bundles
# grade.py trade-off: accept a merge of clusters A, B iff p >= 1 / (1 + k * nA*nB / (nA + nB)), i.e. the
# pairs a true merge recovers (nA*nB) outweigh the sessions a false one contaminates (nA+nB).
# k per FPR budget, chosen on 5-fold CV with the official metric (research/opt.py).
BUDGET_K = {"0.5": 0.25, "1": 0.9, "2": 1.9}


def size_aware(k: float):
    return lambda f: 1.0 / (1.0 + k * f["nA"] * f["nB"] / (f["nA"] + f["nB"]))


def pairs_and_features(df: pd.DataFrame):
    a, b = blocks.decode(blocks.candidates(df))
    P = features.prepare(df)
    X = pd.concat(
        [
            features.pair_features(P, a[i : i + 500000], b[i : i + 500000])
            for i in range(0, len(a), 500000)
        ],
        ignore_index=True,
    )
    return a, b, X


def train(threads: int):
    t = time.time()
    df = harness.load()
    a, b, X = pairs_and_features(df)
    truth = df.truth.to_numpy(object)
    la, lb = pd.notna(truth[a]), pd.notna(truth[b])
    y = np.full(len(a), -1, np.int8)
    y[la & lb] = (truth[a] == truth[b])[la & lb]
    y[la ^ lb] = 0  # labeled vs unlabeled: guaranteed different devices
    meta = pd.DataFrame(dict(a=a, b=b, y=y))
    print("pairs", len(a), round(time.time() - t), flush=True)
    # out-of-fold scores over ALL labeled devices, for fitting the merge gate
    fold = np.array(
        [
            f"f{int(hashlib.sha256(d.encode()).hexdigest(), 16) % 5}"
            if isinstance(d, str)
            else "unlabeled"
            for d in truth
        ],
        dtype=object,
    )
    s_oof = cv.oof_scores(
        X, meta, fold[a], fold[b], dict(PAIR, thread_count=threads, random_seed=0)
    )
    print("oof", round(time.time() - t), flush=True)
    labeled = fold != "unlabeled"
    gates, cols = gated.fit_gates(
        df,
        a,
        b,
        s_oof,
        [d if lab else None for d, lab in zip(truth, labeled, strict=False)],
        [f if lab else None for f, lab in zip(fold, labeled, strict=False)],
        threads=threads,
    )
    known = y >= 0
    scorers = []
    for sd in SEEDS:
        m = CatBoostClassifier(**PAIR, thread_count=threads, random_seed=sd)
        m.fit(X[known], y[known])
        scorers.append(m)
    joblib.dump(
        dict(scorers=scorers, gates=gates, gate_cols=cols, thr=THR, columns=list(X.columns)),
        MODEL,
        compress=3,
    )
    print("saved", MODEL, round(time.time() - t), flush=True)


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if path.endswith((".csv", ".csv.gz")) else pd.read_parquet(path)


class Constrained:
    """Merge rule for deployment: size-aware acceptance plus certain links and cannot-links.

    - forced edges (shared cookie, same labeled device) are always accepted;
    - a cluster holding a labeled device never merges with one holding an original unlabeled session
      (README guarantee) or with a different labeled device. New sessions may join either."""

    def __init__(self, k, lab_dev, orig_unl, forced):
        self.rule, self.forced = size_aware(k), forced
        self.lab, self.unl = {}, {}
        self.row_lab, self.row_unl = lab_dev, orig_unl

    def _state(self, r):
        return self.lab.get(r, self.row_lab[r]), self.unl.get(r, bool(self.row_unl[r]))

    def accept(self, p, ra, rb, C, f):
        la, ua = self._state(ra)
        lb, ub = self._state(rb)
        if (
            (la is not None and lb is not None and la != lb)
            or (la is not None and ub)
            or (lb is not None and ua)
        ):
            return False
        ok = f["edge"] in self.forced or p >= self.rule(f)
        if ok:
            st = (la if la is not None else lb, ua or ub)
            for r in (ra, rb):
                self.lab[r], self.unl[r] = st
        return ok


def _cookie(series: pd.Series) -> np.ndarray:
    c = series.astype(object).where(series.notna(), None).to_numpy(object)
    return np.array(
        [
            x.strip()
            if isinstance(x, str) and x.strip().lower() not in ("", "nan", "none")
            else None
            for x in c
        ],
        dtype=object,
    )


def _chain_edges(groups) -> tuple[list, list]:
    """Consecutive-member edges that connect each group (enough to union it)."""
    ea, eb = [], []
    for g in groups:
        ea += g[:-1]
        eb += g[1:]
    return ea, eb


def predict(new_path: str, out_path: str, budget: str = "1", confidence: bool = False):
    base = pd.read_parquet(ROOT / "sessions.parquet")
    new = read(new_path)
    new = new[base.columns.intersection(new.columns)].drop_duplicates("session_id")
    new = new[
        ~new.session_id.isin(base.session_id)
    ]  # already-known sessions keep original measurements
    df = pd.concat(
        [base, new.astype({c: base[c].dtype for c in new.columns}, errors="ignore")],
        ignore_index=True,
    )
    df["ts"] = pd.to_datetime(df.ts)
    devices = json.load(open(ROOT / "labeled_devices.json"))
    s2dev = {x: d for d, ss in devices.items() for x in ss}
    sid = df.session_id.to_numpy(object)
    lab_dev = np.array([s2dev.get(x) for x in sid], dtype=object)
    orig_unl = np.zeros(len(df), bool)
    orig_unl[: len(base)] = lab_dev[: len(base)] == None  # noqa: E711

    bundle = joblib.load(MODEL)
    a, b, X = pairs_and_features(df)
    s = np.mean([m.predict_proba(X[bundle["columns"]])[:, 1] for m in bundle["scorers"]], axis=0)

    # Certain links, replayed first (score 1): sessions of one labeled device, and sessions sharing a cookie
    # (data dictionary: same cookie_id = same device). Original unlabeled sessions carry no cookie.
    by_dev = pd.Series(np.arange(len(df))[lab_dev != None]).groupby(lab_dev[lab_dev != None])  # noqa: E711
    cookie = _cookie(df.cookie_id)
    has_c = np.array([c is not None for c in cookie])
    by_cookie = pd.Series(np.arange(len(df))[has_c]).groupby(cookie[has_c])
    fa, fb = _chain_edges([list(g) for _, g in by_dev] + [list(g) for _, g in by_cookie])
    forced = set(range(len(a), len(a) + len(fa)))
    a = np.concatenate([a, np.asarray(fa, np.int64)])
    b = np.concatenate([b, np.asarray(fb, np.int64)])
    s = np.concatenate([s, np.ones(len(fa))])

    rule = Constrained(BUDGET_K[budget], lab_dev, orig_unl, forced)
    labels = gated.partition(
        df, a, b, s, fastgate.compile_gates(bundle["gates"]), bundle["gate_cols"], thr=rule
    )

    # Device IDs: a cluster containing a labeled device is named after it (so new sessions that return on
    # a known device are recognizable); otherwise after its smallest session_id.
    lab_of = (
        pd.Series(lab_dev)
        .groupby(labels)
        .transform(lambda x: x.dropna().min() if x.notna().any() else None)
    )
    anchor = pd.Series(sid).groupby(labels).transform("min").to_numpy(object)
    name = [
        "labeled:" + d if isinstance(d, str) else x for d, x in zip(lab_of, anchor, strict=False)
    ]
    dev = np.array(["dev_" + hashlib.sha1(x.encode()).hexdigest()[:16] for x in name], dtype=object)
    keep = lab_dev == None  # noqa: E711
    out = pd.DataFrame(dict(session_id=sid[keep], device_id=dev[keep]))
    if confidence:
        # calibrated P(this session's device holds no other device's session); blank for single-session devices
        cmap = json.load(open(CONFIDENCE))["maps"][budget]
        raw = calibrate.mean_in_cluster(labels, a, b, s)[keep]
        conf = np.interp(raw, cmap["x"], cmap["y"])
        out["link_confidence"] = np.where(np.isnan(raw), np.nan, np.round(conf, 4))
    out.to_csv(out_path, index=False)
    joined = int(
        (~orig_unl[keep] & np.array([n.startswith("labeled:") for n in np.array(name)[keep]])).sum()
    )
    print(
        f"wrote {len(out)} sessions, {out.device_id.nunique()} devices -> {out_path}"
        f" ({joined} new sessions joined a known labeled device)"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("new", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument(
        "--fpr-budget",
        choices=sorted(BUDGET_K),
        default="1",
        help="target share of sessions wrongly linked, in percent (default 1)",
    )
    ap.add_argument(
        "--confidence",
        action="store_true",
        help="add link_confidence: calibrated P(session's device is pure); blank if unlinked",
    )
    args = ap.parse_args()
    if args.train:
        train(args.threads)
    else:
        if not (args.new and args.out):
            ap.error("NEW and OUTPUT are required")
        predict(args.new, args.out, args.fpr_budget, args.confidence)


if __name__ == "__main__":
    main()
