"""End-to-end checks of `linker/link.py` predict (about a minute: two full runs).

.venv/bin/python -m unittest linker/test_link.py -v
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import link  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class LinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = pd.read_parquet(ROOT / "sessions.parquet")
        cls.devices = json.load(open(ROOT / "labeled_devices.json"))
        cls.labeled = {x for ss in cls.devices.values() for x in ss}
        cls.unl = cls.base[~cls.base.session_id.isin(cls.labeled)]
        cls.tmp = Path(tempfile.mkdtemp())
        cls.base.iloc[:0].to_parquet(cls.tmp / "empty.parquet")
        link.predict(str(cls.tmp / "empty.parquet"), str(cls.tmp / "plain.csv"))
        cls.plain = pd.read_csv(cls.tmp / "plain.csv").set_index("session_id").device_id

        # three new sessions that copy unlabeled sessions on three DIFFERENT predicted devices, sharing one cookie
        picks = cls.unl[cls.unl.session_id.map(cls.plain).duplicated().eq(False)].sample(
            3, random_state=7
        )  # noqa: E712
        shared = picks.assign(session_id="new-ck-" + picks.session_id, cookie_id="cookie-shared")
        # two new sessions carrying the cookie of a labeled device
        dev, ss = next(iter(cls.devices.items()))
        lab_row = cls.base[cls.base.session_id == ss[0]]
        lab_cookie = lab_row.cookie_id.iloc[0]
        from_lab = pd.concat(
            [lab_row.assign(session_id=f"new-lab-{i}", cookie_id=lab_cookie) for i in range(2)]
        )
        new = pd.concat([shared, from_lab], ignore_index=True)
        new.to_parquet(cls.tmp / "new.parquet")
        link.predict(str(cls.tmp / "new.parquet"), str(cls.tmp / "new.csv"))
        link.predict(str(cls.tmp / "new.parquet"), str(cls.tmp / "conf.csv"), confidence=True)
        cls.conf = pd.read_csv(cls.tmp / "conf.csv").set_index("session_id")
        cls.out = pd.read_csv(cls.tmp / "new.csv").set_index("session_id").device_id
        cls.shared_ids = list(shared.session_id)
        cls.picked_originals = list(picks.session_id)
        cls.lab_device = dev

    def test_covers_every_unlabeled_session_once_and_no_labeled(self):
        self.assertTrue(self.plain.index.is_unique)
        self.assertEqual(set(self.plain.index), set(self.unl.session_id))
        self.assertFalse(set(self.plain.index) & self.labeled)
        self.assertTrue(self.plain.notna().all())

    def test_shared_cookie_forces_one_device(self):
        self.assertEqual(
            len(set(self.plain[self.picked_originals])), 3
        )  # premise: different devices
        self.assertEqual(self.out[self.shared_ids].nunique(), 1)

    def test_labeled_cookie_joins_that_labeled_device(self):
        want = "dev_" + hashlib.sha1(("labeled:" + self.lab_device).encode()).hexdigest()[:16]
        self.assertEqual(set(self.out[["new-lab-0", "new-lab-1"]]), {want})

    def test_original_unlabeled_never_on_a_labeled_device(self):
        lab_ids = {
            "dev_" + hashlib.sha1(("labeled:" + d).encode()).hexdigest()[:16] for d in self.devices
        }
        for out in (self.plain, self.out):
            originals = out[out.index.isin(self.unl.session_id)]
            self.assertFalse(set(originals) & lab_ids)

    def test_confidence_is_optional_and_only_for_linked_sessions(self):
        self.assertEqual(
            list(pd.read_csv(self.tmp / "new.csv").columns), ["session_id", "device_id"]
        )
        c = self.conf
        self.assertTrue(
            (c.device_id == self.out.reindex(c.index)).all()
        )  # same grouping with the flag
        size = c.device_id.map(c.device_id.value_counts())
        self.assertTrue(c.link_confidence[size == 1].isna().all())
        linked = c.link_confidence[size > 1]
        self.assertTrue(linked.notna().all() and linked.between(0, 1).all())
        self.assertGreater(linked.nunique(), 5)  # it discriminates, not a constant

    def test_no_new_rows_dropped(self):
        self.assertEqual(len(self.out), len(self.unl) + 5)


if __name__ == "__main__":
    unittest.main()
