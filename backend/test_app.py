import io
import json
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

import app as service
from colour_check import build_seen, novel_colour
from inference import PILLQC_POSE, Inspector, overlay_heatmap
from segmentation import align_pill, find_pills, in_training_pose


class FakeInspector:
    def inspect(self, image):
        return {"score": 1.0, "threshold": 2.0, "is_anomalous": False, "size": list(image.size)}


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTest(unittest.TestCase):
    def setUp(self):
        service.app.dependency_overrides[service.get_inspector] = FakeInspector
        self.addCleanup(service.app.dependency_overrides.clear)
        self.client = TestClient(service.app)

    def upload(self, content):
        return self.client.post("/api/inspect", files={"file": ("pill.png", content, "image/png")})

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")

    def test_inspect_valid_image(self):
        response = self.upload(png_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["size"], [8, 6])

    def test_rejects_non_image(self):
        response = self.upload(b"not an image")
        self.assertEqual(response.status_code, 400)

    def test_rejects_large_file(self):
        response = self.upload(b"0" * (service.MAX_UPLOAD_BYTES + 1))
        self.assertEqual(response.status_code, 413)

    def test_rejects_unknown_pill_type(self):
        service.app.dependency_overrides.clear()
        response = self.client.post(
            "/api/inspect", files={"file": ("pill.png", png_bytes(), "image/png")}, data={"pill_type": "capsule"}
        )
        self.assertEqual(response.status_code, 400)

    def test_pill_type_picks_its_own_model(self):
        service.app.dependency_overrides.clear()
        picked = []
        original = service.load_inspector
        service.load_inspector = lambda name: picked.append(name) or FakeInspector()
        self.addCleanup(setattr, service, "load_inspector", original)
        for data in ({}, {"pill_type": "pillqc"}):
            response = self.client.post("/api/inspect", files={"file": ("pill.png", png_bytes(), "image/png")}, data=data)
            self.assertEqual(response.status_code, 200)
        self.assertEqual(picked, ["mvtec", "pillqc"])

    def test_results_keep_only_public_fields(self):
        result = self.client.get("/api/results").json()
        self.assertIn("main", result)
        for run in result["main"].values():
            self.assertNotIn("test_scores", run)
            self.assertNotIn("validation_scores", run)
        json.dumps(result)


class HeatmapTest(unittest.TestCase):
    def test_scale_is_fixed_by_threshold(self):
        image = Image.new("RGB", (4, 4), "black")
        cool = np.asarray(overlay_heatmap(image, np.zeros((4, 4)), 10.0))
        hot = np.asarray(overlay_heatmap(image, np.full((4, 4), 100.0), 10.0))
        self.assertGreater(cool[0, 0, 2], cool[0, 0, 0])
        self.assertFalse(np.array_equal(cool, hot))


def ellipse(canvas, centre, axes, angle=0):
    cv2.ellipse(canvas, centre, axes, angle, 0, 360, (230, 225, 220), -1)


class SegmentationTest(unittest.TestCase):
    def test_counts_separate_and_touching_pills(self):
        scene = np.zeros((400, 900, 3), np.uint8)
        ellipse(scene, (120, 200), (90, 60))
        ellipse(scene, (400, 200), (90, 60), 40)
        ellipse(scene, (640, 200), (90, 60))
        ellipse(scene, (818, 200), (90, 60))  # Touches the pill on its left.
        self.assertEqual(len(find_pills(scene)), 4)

    def test_one_pill_stays_one_pill(self):
        scene = np.zeros((800, 800, 3), np.uint8)
        ellipse(scene, (400, 400), (345, 238))
        self.assertEqual(len(find_pills(scene)), 1)

    def test_blank_image_has_no_pills(self):
        self.assertEqual(find_pills(np.zeros((50, 50, 3), np.uint8)), [])

    def test_align_makes_long_axis_horizontal_at_training_size(self):
        scene = np.zeros((500, 500, 3), np.uint8)
        ellipse(scene, (250, 250), (100, 70), 35)
        _, pill = align_pill(scene, find_pills(scene)[0])
        ys, xs = np.nonzero(pill)
        self.assertAlmostEqual(np.ptp(xs), 693, delta=8)
        self.assertLess(np.ptp(ys), np.ptp(xs))
        self.assertAlmostEqual(xs.mean(), 400, delta=3)

    def test_training_pose_allows_small_changes_only(self):
        def mask(centre=(400, 400), axes=(346, 238), angle=0, size=800):
            scene = np.zeros((size, size, 3), np.uint8)
            ellipse(scene, centre, axes, angle)
            return find_pills(scene)[0]

        self.assertTrue(in_training_pose(mask()))
        self.assertTrue(in_training_pose(mask(angle=181)))
        self.assertTrue(in_training_pose(mask(centre=(200, 200), axes=(173, 119), size=400)))
        self.assertFalse(in_training_pose(mask(angle=30)))
        self.assertFalse(in_training_pose(mask(angle=90)))
        self.assertFalse(in_training_pose(mask(axes=(242, 167))))
        self.assertFalse(in_training_pose(mask(centre=(420, 400))))


def specked_pill(stain=None):
    """A white training-pose pill with dull red specks, and optionally a stain of another colour."""
    scene = np.zeros((800, 800, 3), np.uint8)
    ellipse(scene, (400, 400), (346, 238))
    for x in range(150, 650, 60):
        cv2.circle(scene, (x, 380), 6, (130, 60, 70), -1)
    if stain:
        cv2.circle(scene, (400, 500), 12, stain, -1)
    return scene


class ColourTest(unittest.TestCase):
    def setUp(self):
        normal = specked_pill()
        self.seen = build_seen([(normal, find_pills(normal)[0])])

    def area(self, scene):
        return novel_colour(self.seen, scene, find_pills(scene)[0])[0]

    def test_normal_colours_are_seen(self):
        self.assertEqual(self.area(specked_pill()), 0)
        self.assertEqual(self.area(specked_pill(stain=(130, 60, 70))), 0)  # A larger normal speck.

    def test_new_colours_are_found(self):
        for stain in [(40, 40, 200), (230, 220, 60), (240, 40, 60)]:  # Blue, yellow, bright red.
            self.assertGreater(self.area(specked_pill(stain)), 300, stain)


class RoutingTest(unittest.TestCase):
    """A single pill must not be judged by a whole-image model it is not posed for."""

    colour_seen = None

    def inspect(self, scene, scores, scene_threshold=10.0):
        queue = iter(scores)

        def model(batch):
            return SimpleNamespace(pred_score=torch.tensor([next(queue)]), anomaly_map=torch.zeros(1, 1, 8, 8))

        inspector = Inspector.__new__(Inspector)
        inspector.model, inspector.transform = model, lambda tensor: tensor
        inspector.image_size, inspector.threshold, inspector.scene_threshold = 8, 10.0, scene_threshold
        inspector.colour_seen = self.colour_seen
        inspector.colour_threshold = inspector.scene_colour_threshold = 20
        inspector.inspect_scene = lambda image, masks: {"route": "aligned", "pills": [{} for _ in masks]}
        return inspector.inspect(Image.fromarray(scene))

    def scene(self, angle=0):
        scene = np.zeros((800, 800, 3), np.uint8)
        ellipse(scene, (400, 400), (346, 238), angle)
        return scene

    def test_turned_pill_is_aligned(self):
        result = self.inspect(self.scene(45), [])
        self.assertEqual((result["route"], result["pills"]), ("aligned", [{}]))

    def test_posed_normal_pill_keeps_one_model_run(self):
        result = self.inspect(self.scene(), [5.0])
        self.assertEqual(result["score"], 5.0)
        self.assertFalse(result["is_anomalous"])

    def test_posed_defect_keeps_whole_image_score(self):
        result = self.inspect(self.scene(), [20.0, 15.0])  # Lower once turned, but still abnormal.
        self.assertEqual(result["score"], 20.0)
        self.assertTrue(result["is_anomalous"])

    def test_upside_down_pill_is_aligned(self):
        self.assertEqual(self.inspect(self.scene(), [20.0, 5.0])["route"], "aligned")

    def test_without_scene_threshold_uses_whole_image(self):
        result = self.inspect(self.scene(45), [20.0], scene_threshold=None)
        self.assertEqual(result["score"], 20.0)

    def test_ignored_object_warns_on_both_routes(self):
        scene = self.scene()
        scene[:120, :120] = 0
        ellipse(scene, (60, 60), (50, 35))  # Far below the size of a pill.
        result = self.inspect(scene, [5.0])
        self.assertEqual(result["score"], 5.0)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("ไม่ถูกตรวจ", result["warnings"][0])

    def test_small_pill_warns_after_alignment(self):
        scene = np.zeros((800, 800, 3), np.uint8)
        ellipse(scene, (400, 400), (60, 40))
        self.assertIn("สั้นกว่า", self.inspect(scene, [])["warnings"][0])

    def test_speck_gives_no_warning(self):
        scene = self.scene()
        ellipse(scene, (60, 60), (8, 8))
        self.assertEqual(self.inspect(scene, [5.0])["warnings"], [])

    def test_odd_colour_flags_a_low_scoring_pill(self):
        normal = specked_pill()
        self.colour_seen = build_seen([(normal, find_pills(normal)[0])])
        self.assertFalse(self.inspect(normal, [5.0])["is_anomalous"])
        result = self.inspect(specked_pill(stain=(40, 40, 200)), [5.0])
        self.assertEqual(result["score"], 5.0)
        self.assertTrue(result["odd_colour"])
        self.assertTrue(result["is_anomalous"])

    def pillqc(self, scene, scores):
        queue = iter(scores)
        inspector = Inspector.__new__(Inspector)
        inspector.model = lambda batch: SimpleNamespace(pred_score=torch.tensor([next(queue)]), anomaly_map=torch.zeros(1, 1, 8, 8))
        inspector.transform = lambda tensor: tensor
        inspector.image_size, inspector.threshold, inspector.scene_threshold = 8, 10.0, 12.0
        inspector.colour_seen = inspector.colour_threshold = inspector.scene_colour_threshold = None
        inspector.pose = PILLQC_POSE
        inspector.inspect_scene = lambda image, masks: {"route": "pills", "pills": [{} for _ in masks]}
        return inspector.inspect(Image.fromarray(scene))

    def test_pillqc_crop_keeps_whole_image_score(self):
        crop = np.full((225, 225, 3), (70, 120, 120), np.uint8)  # Teal border, like a pillQC crop.
        cv2.circle(crop, (112, 112), 74, (240, 190, 180), -1)
        cv2.circle(crop, (20, 20), 6, (240, 190, 180), -1)  # A speck: no warning on this route.
        result = self.pillqc(crop, [20.0])
        self.assertEqual((result["score"], result["is_anomalous"], result["warnings"]), (20.0, True, []))

    def test_pillqc_pills_on_dark_background_are_cut_out(self):
        scene = np.zeros((600, 900, 3), np.uint8)
        for x in (150, 450, 750):
            cv2.circle(scene, (x, 300), 120, (240, 190, 180), -1)
        result = self.pillqc(scene, [])
        self.assertEqual((result["route"], len(result["pills"]), result["warnings"]), ("pills", 3, []))

    def test_clean_images_have_no_warnings(self):
        self.assertEqual(self.inspect(self.scene(), [5.0])["warnings"], [])
        self.assertEqual(self.inspect(self.scene(45), [])["warnings"], [])


if __name__ == "__main__":
    unittest.main()
