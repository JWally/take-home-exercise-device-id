"""Score a submission against known devices.

    python grade.py SUBMISSION.csv labeled_devices.json

SUBMISSION.csv has the columns session_id, device_id. The devices file maps each device to its sessions, in the
format of labeled_devices.json. Only the sessions the devices file lists are scored. A scored session that the
submission leaves out, or gives a blank or "nan" device_id, is a device of its own.

recall               of the pairs of scored sessions that share a device, the share the submission puts on one device
false positive rate  the share of scored sessions that the submission puts on one device with a session from another
                     device; a session the devices file does not list counts as another device
"""

import csv
import json
import sys
from collections import Counter


def devices(path):
    return {s: d for d, ss in json.load(open(path)).items() for s in ss}


def submission(path):
    sub: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = csv.reader(f)
        head = [h.strip() for h in next(rows)]
        i, j = head.index("session_id"), head.index("device_id")
        for r in rows:
            s, d = r[i].strip(), r[j].strip()
            if d.lower() in ("", "nan", "none", "null"):
                continue
            if sub.setdefault(s, d) != d:
                sys.exit(f"session {s} has two device_ids")
    return sub


def score(key, sub):
    size = Counter(sub.values())
    pred = {s: sub.get(s, (s,)) for s in key}
    cell = Counter((key[s], pred[s]) for s in key)
    pairs = sum(n * (n - 1) // 2 for n in Counter(key.values()).values())
    found = sum(n * (n - 1) // 2 for n in cell.values())
    wrong = sum(size.get(pred[s], 1) > cell[key[s], pred[s]] for s in key)
    return {
        "sessions": len(key),
        "unassigned": sum(s not in sub for s in key),
        "same_device_pairs": pairs,
        "pairs_found": found,
        "recall": round(found / max(pairs, 1), 4),
        "sessions_wrongly_linked": wrong,
        "false_positive_rate": round(wrong / max(len(key), 1), 4),
    }


if __name__ == "__main__":
    print(json.dumps(score(devices(sys.argv[2]), submission(sys.argv[1])), indent=1))
