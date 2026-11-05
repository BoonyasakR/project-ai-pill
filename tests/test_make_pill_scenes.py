import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]

from make_pill_scenes import compose_scene, cut_pill  # noqa: E402
from segmentation import find_pills  # noqa: E402


class SceneTest(unittest.TestCase):
    def scene(self, touching):
        rng = np.random.default_rng(0)
        image = np.zeros((800, 800, 3), np.uint8)
        cv2.ellipse(image, (400, 400), (345, 238), 0, 0, 360, (230, 225, 220), -1)
        mask = image[..., 0] > 0
        return compose_scene([cut_pill(image, mask, rng) for _ in range(8)], rng, touching)

    def test_pills_do_not_overlap_and_are_all_found(self):
        for touching in (False, True):
            scene, masks = self.scene(touching)
            self.assertEqual(np.sum(masks, axis=0).max(), 1)
            self.assertEqual(len(find_pills(scene)), 8)


if __name__ == "__main__":
    unittest.main()
