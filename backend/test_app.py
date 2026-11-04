import io
import json
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as service
from inference import overlay_heatmap


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


if __name__ == "__main__":
    unittest.main()
