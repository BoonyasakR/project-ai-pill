"""Run the locked PatchCore model on one pill image."""
import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from matplotlib import colormaps
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

HEATMAP_LOW = 0.5  # Share of the threshold shown as the coldest colour.
HEATMAP_HIGH = 1.25  # Share of the threshold shown as the hottest colour.


def overlay_heatmap(image, anomaly_map, threshold):
    """Colour an anomaly map on a fixed scale, so normal pills stay cool."""
    scaled = (anomaly_map / threshold - HEATMAP_LOW) / (HEATMAP_HIGH - HEATMAP_LOW)
    colours = colormaps["jet"](np.clip(scaled, 0, 1))[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    blended = 0.55 * base + 0.45 * colours
    return Image.fromarray((blended * 255).astype(np.uint8))


def to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class Inspector:
    def __init__(self, model_path):
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
    def inspect(self, image):
        image = image.convert("RGB")
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
