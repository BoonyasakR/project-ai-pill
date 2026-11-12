import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bootstrap as b  # noqa: E402

LABELS = [0] * 20 + [1] * 30


def overlapping_scores(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, 20), rng.normal(1.5, 1, 30)])


class BootstrapTest(unittest.TestCase):
    def test_perfect_separation_gives_a_point_interval(self):
        scores = [0.1] * 20 + [0.9] * 30
        result = b.image_ci(LABELS, scores, 0.5, rounds=200)
        for name in b.METRICS:
            self.assertEqual((result[name]["low"], result[name]["high"]), (1.0, 1.0))

    def test_interval_contains_the_point_estimate(self):
        result = b.image_ci(LABELS, overlapping_scores(), 0.75, rounds=500)
        for name in b.METRICS:
            row = result[name]
            self.assertLessEqual(row["low"], row["value"])
            self.assertLessEqual(row["value"], row["high"])
        self.assertLess(result["image_auroc"]["low"], result["image_auroc"]["high"])

    def test_same_seed_gives_same_interval(self):
        scores = overlapping_scores()
        self.assertEqual(
            b.image_ci(LABELS, scores, 0.75, rounds=100, seed=7),
            b.image_ci(LABELS, scores, 0.75, rounds=100, seed=7),
        )

    def test_paired_difference(self):
        scores = overlapping_scores()
        same = b.paired_auroc_diff_ci(LABELS, scores, scores, rounds=100)
        self.assertEqual((same["value"], same["low"], same["high"]), (0.0, 0.0, 0.0))
        noise = np.random.default_rng(1).normal(0, 1, 50)
        better = b.paired_auroc_diff_ci(LABELS, LABELS, noise, rounds=300)
        self.assertGreater(better["low"], 0)

    def test_one_class_only_fails(self):
        with self.assertRaises(ValueError):
            b.image_ci([1, 1, 1], [0.1, 0.2, 0.3], 0.5)


if __name__ == "__main__":
    unittest.main()
