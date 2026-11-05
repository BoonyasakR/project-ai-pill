"""Run the locked PatchCore model on images with one or several pills."""
import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from matplotlib import colormaps
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

from segmentation import align_pill, find_pills

PILL_EDGE_PIXELS = 5  # Ignore this rim of the pill, where the background bleeds in.
MARK_GOOD = (40, 200, 110)
MARK_BAD = (240, 60, 50)
ROW_HEIGHT = 300  # Pixels. Only affects numbering order.
HEATMAP_LOW = 0.5  # Share of the threshold shown as the coldest colour.
HEATMAP_HIGH = 1.25  # Share of the threshold shown as the hottest colour.


def overlay_heatmap(image, anomaly_map, threshold):
    """Colour an anomaly map on a fixed scale, so normal pills stay cool."""
    scaled = (anomaly_map / threshold - HEATMAP_LOW) / (HEATMAP_HIGH - HEATMAP_LOW)
    colours = colormaps["jet"](np.clip(scaled, 0, 1))[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    blended = 0.55 * base + 0.45 * colours
    return Image.fromarray((blended * 255).astype(np.uint8))


def reading_order(mask):
    """Sort pills top to bottom in rows of about one pill height, then left to right."""
    ys, xs = np.nonzero(mask)
    return int(ys.mean() // ROW_HEIGHT), xs.mean()


def to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class Inspector:
    def __init__(self, model_path, scene_results=None):
        scene_data = json.loads(Path(scene_results).read_text()) if scene_results else {}
        self.scene_threshold = scene_data.get("threshold")
        checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        self.threshold = float(checkpoint["threshold"])
        self.image_size = int(checkpoint["image_size"])
        state = checkpoint["state_dict"]

        self.model = PatchcoreModel(
            layers=checkpoint["layers"], backbone=checkpoint["backbone"], pre_trained=False
        )
        self.model.memory_bank = torch.empty_like(state["memory_bank"])
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = Compose([
            Resize((self.image_size, self.image_size), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def score_pill(self, image, mask, others):
        """Score one pill from a larger image, in both 180-degree orientations.

        Only the pill area counts, because the plain filled background looks unlike
        the training background. The right way up matches training best, so the
        lower score is kept.
        """
        views = [align_pill(image, mask, others, flip) for flip in (False, True)]
        batch = torch.stack([
            self.transform(to_dtype(torch.from_numpy(view).permute(2, 0, 1), torch.float32, scale=True))
            for view, _ in views
        ])
        maps = self.model(batch).anomaly_map.squeeze(1).numpy()
        size = (self.image_size, self.image_size)
        kernel = np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)
        regions = [
            cv2.erode(cv2.resize(pill.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST), kernel) > 0
            for _, pill in views
        ]
        scores = [float(score_map[region].max()) for score_map, region in zip(maps, regions)]
        best = int(np.argmin(scores))
        return scores[best], views[best][0], np.where(regions[best], maps[best], 0)

    def inspect_scene(self, image, masks):
        """Inspect every pill in an RGB array and mark the results on the image."""
        start = time.perf_counter()
        everything = np.any(masks, axis=0)
        marked = image.copy()
        pills = []
        for number, mask in enumerate(sorted(masks, key=reading_order), start=1):
            score, view, anomaly_map = self.score_pill(image, mask, everything & ~mask)
            abnormal = score > self.scene_threshold
            colour = MARK_BAD if abnormal else MARK_GOOD
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(marked, contours, -1, colour, 3)
            x, y = np.nonzero(mask)[::-1]
            cv2.putText(marked, str(number), (int(x.mean()) - 12, int(y.mean()) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
            small = Image.fromarray(view).resize((self.image_size, self.image_size))
            pills.append({
                "number": number,
                "score": score,
                "is_anomalous": abnormal,
                "heatmap": to_data_url(overlay_heatmap(small, anomaly_map, self.scene_threshold)),
            })
        return {
            "score": max(pill["score"] for pill in pills),
            "threshold": self.scene_threshold,
            "is_anomalous": any(pill["is_anomalous"] for pill in pills),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "image": to_data_url(Image.fromarray(marked)),
            "pills": pills,
        }

    @torch.inference_mode()
    def inspect(self, image):
        image = image.convert("RGB")
        array = np.asarray(image)
        masks = find_pills(array)
        if len(masks) > 1 and self.scene_threshold is not None:
            return self.inspect_scene(array, masks)
        tensor = to_dtype(pil_to_tensor(image), torch.float32, scale=True)

        start = time.perf_counter()
        output = self.model(self.transform(tensor).unsqueeze(0))
        latency_ms = (time.perf_counter() - start) * 1000

        score = float(output.pred_score.flatten()[0])
        anomaly_map = output.anomaly_map.squeeze().numpy()
        resized = image.resize((self.image_size, self.image_size))
        return {
            "score": score,
            "threshold": self.threshold,
            "is_anomalous": score > self.threshold,
            "latency_ms": latency_ms,
            "image": to_data_url(resized),
            "heatmap": to_data_url(overlay_heatmap(resized, anomaly_map, self.threshold)),
        }
