"""Leaderboard under the README/grade.py objective: max recall s.t. FPR <= budget (CV, official metric)."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linker"))
import harness  # noqa: E402

LOG = Path(__file__).parent / "cache" / "board.jsonl"
_DF = None


def score(pred, name: str) -> dict:
    global _DF
    if _DF is None:
        df = harness.load()
        _DF = (df, harness.assign_folds(df).to_numpy(object))
    df, fold = _DF
    vis = fold == "unlabeled"
    r = [harness.official(df.truth, pred, fold == f"f{k}", vis) for k in range(5)]
    row = dict(
        name=name,
        recall=float(np.mean([x["recall"] for x in r])),
        fpr=float(np.mean([x["fpr"] for x in r])),
        fpr_max=float(max(x["fpr"] for x in r)),
        t=time.strftime("%H:%M:%S"),
    )
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def show():
    rows = [json.loads(x) for x in open(LOG)] if LOG.exists() else []
    print(time.strftime("[%H:%M:%S]"), f"{len(rows)} configs scored")
    for budget in (0.01, 0.02):
        ok = [r for r in rows if r["fpr"] <= budget]
        if ok:
            b = max(ok, key=lambda r: r["recall"])
            print(
                f"  FPR<= {budget:.0%}: recall {b['recall']:.4f}  fpr {b['fpr']:.4f}  ({b['name']})"
            )


if __name__ == "__main__":
    show()
