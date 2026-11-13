"""Extract the numbers behind the figures from research/cache into docs/figures_data.json.

Needs the cache produced by build_pairs.py, run_cv.py and opt.py. make_figures.py reads only the JSON.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "linker"))
import calibrate  # noqa: E402
import harness  # noqa: E402

CACHE = Path(__file__).parent / "cache"
MAJOR = [
    "Android chrome",
    "Windows chrome",
    "Windows firefox",
    "iOS webview",
    "iOS safari",
    "Windows edge",
    "Android samsung",
    "Android webview",
    "macOS safari",
]


def cohorts(df):
    """OS x browser family per session; small cohorts fold into 'other'."""
    ua = df.user_agent.fillna("")
    os_ = np.select(
        [
            ua.str.contains("iPhone|iPad"),
            ua.str.contains("Android"),
            ua.str.contains("Windows"),
            ua.str.contains("Mac OS X"),
            ua.str.contains("Linux|CrOS"),
        ],
        ["iOS", "Android", "Windows", "macOS", "Linux"],
        "other",
    )
    c = pd.Series(os_, index=df.index) + " " + df.browser_family.fillna("")
    return c.where(c.isin(MAJOR), "other").to_numpy(object)


def main():
    df = harness.load()
    fold = harness.assign_folds(df).to_numpy(object)
    vis = fold == "unlabeled"
    out = {}

    # 1. recall / false-merge frontier (CV, grade.py metric) + lockbox points
    rows = {}
    for line in open(CACHE / "board.jsonl"):
        r = json.loads(line)
        rows[r["name"]] = r
    series = {"size_aware": [], "fixed_gate": [], "components": []}
    for n, r in rows.items():
        pt = [round(r["fpr"] * 100, 3), round(r["recall"] * 100, 3)]
        if n.startswith("oof_ens c:") and n.count(":") == 1:
            series["size_aware"].append(pt)
        elif n.startswith("gate@") and "+" not in n:
            series["fixed_gate"].append(pt)
        elif n.startswith("cc@"):
            series["components"].append(pt)
    out["frontier"] = {k: sorted(v) for k, v in series.items()}
    lock = []
    for k in ["0.25", "0.9", "1.9", "5", "8"]:
        r = harness.official(
            df.truth, np.load(CACHE / f"opt_oof_ens_c{k}.npy"), fold == "lockbox", vis
        )
        lock.append([round(r["fpr"] * 100, 3), round(r["recall"] * 100, 3)])
    out["lockbox"] = lock

    # 2. cohort breakdown at the default (1%) setting
    coh = cohorts(df)
    dev_coh = pd.Series(coh).groupby(df.truth).agg(lambda x: x.mode()[0])
    devc = df.truth.map(dev_coh).to_numpy(object)
    pred = np.load(CACHE / "opt_oof_ens_c0.9.npy")
    cv = pd.Series(fold).str.startswith("f").to_numpy()
    cohort_rows = []
    for c in MAJOR:
        m = cv & (devc == c)
        rs = [harness.official(df.truth, pred, m & (fold == f"f{k}"), vis) for k in range(5)]
        sz = df[m].groupby("truth").size()
        pairs = int((sz * (sz - 1) // 2).sum())
        rec = float(np.mean([r["recall"] for r in rs]))
        cohort_rows.append(
            dict(
                cohort=c,
                sessions=int(m.sum()),
                recall=round(rec * 100, 2),
                fpr=round(float(np.mean([r["fpr"] for r in rs])) * 100, 2),
                missed=round(pairs * (1 - rec)),
            )
        )
    tot = sum(c["missed"] for c in cohort_rows)
    for c in cohort_rows:
        c["share_of_misses"] = round(100 * c["missed"] / tot, 1)
    out["cohorts"] = cohort_rows

    # 3. within-device signal stability: Android Chrome vs Windows Chrome
    df["coh"] = coh
    cols = ["ip", "asn", "user_agent", "languages", "text_shaping_hash", "media_speech_voice_hash"]
    stab = {}
    for c in ["Android chrome", "Windows chrome"]:
        lab = df[(df.coh == c) & df.truth.notna()]
        g = lab.groupby("truth").size()
        mm = lab[lab.truth.isin(g[g > 1].index)]
        stab[c] = {
            col: round(100 * float((mm.groupby("truth")[col].nunique(dropna=False) > 1).mean()), 1)
            for col in cols
        }
    out["stability"] = stab

    # 4. confidence calibration, held-out folds, default budget
    m = pd.read_parquet(CACHE / "meta.parquet")
    a, b = m.a.to_numpy(), m.b.to_numpy()
    s = np.load(CACHE / "oof_ens.npy")
    R = calibrate.labeled_rows(df, fold, pred, calibrate.mean_in_cluster(pred, a, b, s))
    held = []
    for tr in ([0, 1, 2], [2, 3, 4]):
        te = R[~R.fold.isin(tr)]
        iso = calibrate.fit(R[R.fold.isin(tr)])
        held.append(pd.DataFrame(dict(conf=iso.predict(te.score), correct=te.correct)))
    rel = calibrate.reliability(pd.concat(held).conf, pd.concat(held).correct)
    out["calibration"] = [
        dict(
            bin=str(i),
            sessions=int(r.sessions),
            predicted=float(r.predicted),
            observed=float(r.observed),
        )
        for i, r in rel.iterrows()
    ]

    # 5. why one 2k-device split is noisy: F1 on random 2,033-device subsets of CV devices
    p04 = np.load(CACHE / "oof_ens.pred0.4.npy")
    cvd = df.truth[cv].unique()
    rng = np.random.default_rng(0)
    f1s = [
        harness.metrics(
            df.truth, p04, df.truth.isin(set(rng.choice(cvd, 2033, replace=False))).to_numpy()
        )["f1"]
        for _ in range(300)
    ]
    out["split_noise_f1"] = [round(100 * x, 3) for x in f1s]

    (ROOT / "docs" / "figures_data.json").write_text(json.dumps(out, indent=1))
    print("wrote docs/figures_data.json")


if __name__ == "__main__":
    main()
