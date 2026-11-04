import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ad_metrics as m  # noqa: E402


def sample_data(perfect=True):
    rng = np.random.default_rng(0)
    masks = np.zeros((6, 32, 32), dtype=bool)
    masks[3, 2:6, 2:6] = True
    masks[4, 10:20, 10:20] = True
    masks[5, 25:28, 25:28] = True
    maps = rng.random(masks.shape).astype(np.float32)
    if perfect:
        maps[masks] += 2
    return masks, maps


class MetricsTest(unittest.TestCase):
    def test_threshold_uses_quantile(self):
        self.assertAlmostEqual(m.pick_threshold(range(101), 0.99), 99.0)

    def test_perfect_map_scores_one(self):
        masks, maps = sample_data()
        self.assertAlmostEqual(m.au_pro(masks, maps), 1.0, places=2)

    def test_random_map_scores_about_half_the_fpr_limit(self):
        masks, maps = sample_data(perfect=False)
        self.assertAlmostEqual(m.au_pro(masks, maps), 0.15, delta=0.06)

    def test_small_region_counts_as_much_as_large_region(self):
        masks = np.zeros((1, 40, 40), dtype=bool)
        masks[0, 1:3, 1:3] = True
        masks[0, 10:30, 10:30] = True
        maps = np.random.default_rng(1).random(masks.shape).astype(np.float32)
        maps[0, 10:30, 10:30] += 2
        self.assertAlmostEqual(m.au_pro(masks, maps), 0.5 + 0.5 * 0.15, delta=0.08)

    def test_shape_mismatch_and_empty_masks_fail(self):
        with self.assertRaises(ValueError):
            m.au_pro(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)))
        with self.assertRaises(ValueError):
            m.au_pro(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)))

    def test_evaluate_counts_and_scores(self):
        masks, maps = sample_data()
        labels = [0, 0, 0, 1, 1, 1]
        result = m.evaluate(labels, [0.1, 0.2, 0.9, 0.8, 0.7, 0.3], masks, maps, 0.5)
        self.assertEqual(
            (result["true_positive"], result["false_positive"]), (2, 1)
        )
        self.assertEqual(
            (result["false_negative"], result["true_negative"]), (1, 2)
        )
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["pixel_auroc"], 1.0)

    def test_evaluate_by_defect_compares_with_good_only(self):
        masks, maps = sample_data()
        groups = ["good", "good", "good", "crack", "crack", "scratch"]
        rows = m.evaluate_by_defect(
            groups, [0.1, 0.2, 0.3, 0.9, 0.2, 0.8], masks, maps, 0.5
        )
        self.assertEqual(set(rows), {"crack", "scratch"})
        self.assertEqual(rows["crack"]["images"], 2)
        self.assertAlmostEqual(rows["crack"]["recall"], 0.5)
        self.assertAlmostEqual(rows["scratch"]["image_auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
