"""Data loading, fold/lockbox assignment, and metrics.

Every metric scores a set of labeled sessions while the prediction covers all 65,478 sessions, so a link
to any other session counts against it. `official` reproduces grade.py exactly.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "sessions.parquet")
    lab = json.load(open(ROOT / "labeled_devices.json"))
    s2d = {s: d for d, ss in lab.items() for s in ss}
    df["truth"] = df.session_id.map(s2d)
    return df.reset_index(drop=True)


def metrics(truth, pred, mask) -> dict:
    truth, pred, mask = np.asarray(truth, dtype=object), np.asarray(pred), np.asarray(mask, bool)
    t, p = truth[mask], pred[mask]
    cells = Counter(zip(t.tolist(), p.tolist(), strict=False))
    c2 = lambda ns: sum(n * (n - 1) // 2 for n in ns)  # noqa: E731
    pred_sizes, all_sizes = Counter(p.tolist()), Counter(pred.tolist())
    tp, pos = c2(cells.values()), c2(Counter(t.tolist()).values())
    predicted = c2(pred_sizes.values()) + sum(n * (all_sizes[k] - n) for k, n in pred_sizes.items())
    wrong = sum(n for (d, c), n in cells.items() if all_sizes[c] > n)
    return dict(
        f1=round(2 * tp / (pos + predicted), 5),
        precision=round(tp / max(predicted, 1), 5),
        recall=round(tp / max(pos, 1), 5),
        tp=tp,
        fp=predicted - tp,
        fn=pos - tp,
        wrong_sessions=wrong,
        fpr=round(wrong / mask.sum(), 5),
    )


def assign_folds(df: pd.DataFrame, k: int = 5, lockbox_pct: int = 15) -> pd.Series:
    """Evaluation protocol, independent of the earlier round's over-inspected split.
    Device -> 'lockbox' (sealed, scored once at the end) or 'f0'..'f{k-1}' (CV), via a stable hash."""
    import hashlib

    def bucket(d):
        h = int(hashlib.sha256(("fresh-v1:" + d).encode()).hexdigest(), 16) % 100
        return "lockbox" if h < lockbox_pct else f"f{h % k}"

    fold = df.truth.map(lambda d: bucket(d) if isinstance(d, str) else "unlabeled")
    return fold


def official(truth, pred, scored, visible) -> dict:
    """grade.py as the hidden key will apply it: only `visible` sessions are in the submission
    (the scored pool + its fellow unlabeled distractors; other labeled sessions are not written),
    recall = share of same-device pairs together, fpr = share of scored sessions whose predicted
    device also holds a session from another device."""
    truth, pred = np.asarray(truth, dtype=object), np.asarray(pred)
    scored, visible = np.asarray(scored, bool), np.asarray(visible, bool)
    t, p = truth[scored], pred[scored]
    cells = Counter(zip(t.tolist(), p.tolist(), strict=False))
    vis_sizes = Counter(pred[visible | scored].tolist())
    pos = sum(n * (n - 1) // 2 for n in Counter(t.tolist()).values())
    tp = sum(n * (n - 1) // 2 for n in cells.values())
    wrong = sum(n for (d, c), n in cells.items() if vis_sizes[c] > n)
    return dict(
        recall=round(tp / max(pos, 1), 5),
        fpr=round(wrong / max(scored.sum(), 1), 5),
        wrong_sessions=wrong,
        sessions=int(scored.sum()),
    )
