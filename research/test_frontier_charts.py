"""Protect the exact-count comparisons used in the public frontier charts."""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier_charts import aggregate, best_at_cap, load_data  # noqa: E402


class FrontierEvidenceTest(unittest.TestCase):
    def test_four_percent_cap_does_not_round_a_failure_into_a_pass(self):
        rows = [
            dict(id="inside", tp=100, true_pairs=200, wrong=789, sessions=19735),
            dict(id="outside", tp=110, true_pairs=200, wrong=790, sessions=19735),
        ]
        self.assertEqual(best_at_cap(rows, Fraction(4, 100))["id"], "inside")

    def test_pooled_cap_does_not_require_every_fold_to_pass(self):
        data = load_data()
        rows = [aggregate(r) for r in data["policies"] if r["family"] == "main_refit"]
        winner = best_at_cap(rows, Fraction(1, 100))
        self.assertEqual(winner["id"], "main:size:0.4")
        self.assertEqual((winner["tp"], winner["wrong"]), (28098, 195))
        self.assertGreater(winner["max_fold_error"], Fraction(1, 100))

    def test_duplicate_fold_cannot_replace_a_missing_fold(self):
        row = load_data()["policies"][0]
        with self.assertRaises(ValueError):
            aggregate(dict(row, folds=[row["folds"][0]] * 3))


if __name__ == "__main__":
    unittest.main()
