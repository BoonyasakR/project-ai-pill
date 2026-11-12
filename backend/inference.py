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

from colour_check import BINS, novel_colour
from segmentation import (
    CANVAS, MIN_RELIABLE_LENGTH, MIN_WARNED_AREA, align_pill, find_blobs, in_training_pose, keep_pills, pill_pose,
    pink_pill_mask, round_pill_region,
)

# How each tablet type is cut out of a multi-pill image. MVTec pills are oval, so
# they are turned to the training pose and scored both ways round. pillQC pills
# are round: no turn, scaled to the median pill length of a pillQC crop (148 of 225 px).
MVTEC_POSE = {"align": {}, "flips": (False, True), "min_length": MIN_RELIABLE_LENGTH}
PILLQC_POSE = {"align": {"canvas": 225, "pill_length": 148, "rotate": False}, "flips": (False,), "min_length": 148}
PILLQC_CROP_BACKGROUND = 45  # pillQC crops have a teal border (median 72 or more); scenes are dark (about 18).

PILL_EDGE_PIXELS = 5  # Ignore this rim of the pill, where the background bleeds in.
MARK_GOOD = (40, 200, 110)
MARK_BAD = (240, 60, 50)
ROW_HEIGHT = 300  # Pixels. Only affects numbering order.
HEATMAP_LOW = 0.5  # Share of the threshold shown as the coldest colour.
HEATMAP_HIGH = 1.25  # Share of the threshold shown as the hottest colour.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # The Apple GPU is several times faster.


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


def border_brightness(array):
    gray = array.mean(axis=2)
    return float(np.median(np.concatenate([gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()])))


class Inspector:
    pose = MVTEC_POSE
    score_area = "image"  # "pill": a single crop is scored on its pill only, as its threshold was.

    def __init__(self, model_path, scene_results=None, pose=MVTEC_POSE):
        self.pose = pose
        scene_data = json.loads(Path(scene_results).read_text()) if scene_results else {}
        self.scene_threshold = scene_data.get("threshold")
        self.scene_colour_threshold = scene_data.get("colour_threshold")
        checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        self.threshold = float(checkpoint["threshold"])
        self.image_size = int(checkpoint["image_size"])
        self.score_area = checkpoint.get("score_area", "image")
        state = checkpoint["state_dict"]
        # Older model files have no colour table. They still work, without the colour check.
        self.colour_seen, self.colour_threshold = None, None
        if "colour_seen" in checkpoint:
            bits = np.unpackbits(checkpoint["colour_seen"].numpy())[:BINS ** 3]
            self.colour_seen = bits.reshape((BINS,) * 3).astype(bool)
            self.colour_threshold = float(checkpoint["colour_threshold"])

        self.model = PatchcoreModel(
            layers=checkpoint["layers"], backbone=checkpoint["backbone"], pre_trained=False
        )
        self.model.memory_bank = torch.empty_like(state["memory_bank"])
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
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
        views = [align_pill(image, mask, others, flip, **self.pose["align"]) for flip in self.pose["flips"]]
        batch = torch.stack([
            self.transform(to_dtype(torch.from_numpy(view).permute(2, 0, 1), torch.float32, scale=True))
            for view, _ in views
        ])
        maps = self.model(batch.to(DEVICE)).anomaly_map.squeeze(1).cpu().numpy()
        size = (self.image_size, self.image_size)
        kernel = np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)
        regions = [
            cv2.erode(cv2.resize(pill.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST), kernel) > 0
            for _, pill in views
        ]
        scores = [float(score_map[region].max()) for score_map, region in zip(maps, regions)]
        best = int(np.argmin(scores))
        colour_area, colour_mask = self.check_colour(*views[best])
        return scores[best], views[best][0], np.where(regions[best], maps[best], 0), colour_area, colour_mask

    def check_colour(self, image, mask):
        """Measure unseen colour on a pill posed on the training canvas.

        Returns the largest blob area and the unseen pixels at heatmap size.
        """
        if self.colour_seen is None:
            return 0, None
        area, novel = novel_colour(self.colour_seen, image, mask)
        size = (self.image_size, self.image_size)
        return area, cv2.resize(novel.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0

    def mark_colour(self, anomaly_map, colour_mask, threshold):
        """Show flagged unseen colour as hot on the heatmap."""
        return np.where(colour_mask, np.maximum(anomaly_map, threshold * HEATMAP_HIGH), anomaly_map)

    def inspect_scene(self, image, masks):
        """Inspect every pill in an RGB array and mark the results on the image."""
        start = time.perf_counter()
        everything = np.any(masks, axis=0)
        marked = image.copy()
        pills = []
        for number, mask in enumerate(sorted(masks, key=reading_order), start=1):
            score, view, anomaly_map, colour_area, colour_mask = self.score_pill(image, mask, everything & ~mask)
            odd_colour = self.scene_colour_threshold is not None and colour_area > self.scene_colour_threshold
            if odd_colour:
                anomaly_map = self.mark_colour(anomaly_map, colour_mask, self.scene_threshold)
            abnormal = score > self.scene_threshold or odd_colour
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
                "odd_colour": odd_colour,
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

    def inspect(self, image):
        image = image.convert("RGB")
        array = np.asarray(image)
        blobs = find_blobs(array)
        masks = keep_pills(blobs)
        result = self.judge(image, array, masks)
        result["warnings"] = []
        if self.pose is not MVTEC_POSE and "pills" not in result:
            return result  # A pillQC crop: dirt specks on its teal border are not pills to warn about.
        ignored = sum(blob.sum() >= MIN_WARNED_AREA * blobs[0].sum() for blob in blobs[len(masks):])
        if ignored:
            result["warnings"].append(f"มีวัตถุขนาดเล็ก {ignored} ชิ้นที่ไม่ถูกตรวจ เพราะเล็กกว่าเม็ดใหญ่สุดมาก")
        if "pills" in result:
            shortest = self.pose["min_length"]
            small = sum(pill_pose(mask)[2] < shortest for mask in masks)
            if small:
                result["warnings"].append(
                    f"ยา {small} เม็ดสั้นกว่า {shortest} พิกเซล ผลอาจแจ้งผิดปกติเกินจริง ควรถ่ายให้ใกล้ขึ้น"
                )
        return result

    @torch.inference_mode()
    def judge(self, image, array, masks):
        can_align = self.scene_threshold is not None
        if self.pose is not MVTEC_POSE:
            # A pillQC crop (teal border) keeps the whole-image score it was calibrated on;
            # anything else, such as pills on a dark background, is cut into pills.
            if can_align and masks and border_brightness(array) < PILLQC_CROP_BACKGROUND:
                return self.inspect_scene(array, masks)
        elif can_align and (len(masks) > 1 or (masks and not in_training_pose(masks[0]))):
            return self.inspect_scene(array, masks)
        tensor = to_dtype(pil_to_tensor(image), torch.float32, scale=True).to(DEVICE)

        start = time.perf_counter()
        output = self.model(self.transform(tensor).unsqueeze(0))
        latency_ms = (time.perf_counter() - start) * 1000

        score = float(output.pred_score.flatten()[0])
        if can_align and masks and score > self.threshold and self.pose is MVTEC_POSE:
            # The pose check cannot see an upside-down pill. A normal one passes once turned;
            # a real defect stays above the threshold, so its score is kept.
            turned = self.model(self.transform(tensor.flip(1, 2)).unsqueeze(0))
            if float(turned.pred_score.flatten()[0]) <= self.threshold:
                return self.inspect_scene(array, masks)
        anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
        if self.score_area == "pill":
            # White dirt on the floor must not count, so only the pill is scored and shown.
            region = round_pill_region(pink_pill_mask(array)).astype(np.uint8)
            region = cv2.resize(region, anomaly_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
            region = cv2.erode(region, np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)) > 0
            anomaly_map = np.where(region, anomaly_map, 0)
            score = float(anomaly_map[region].max())
        odd_colour = False
        if masks:
            # The pill is posed like training, so resizing puts it on the training canvas.
            canvas = cv2.resize(array, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
            pill = cv2.resize(masks[0].astype(np.uint8), (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST) > 0
            colour_area, colour_mask = self.check_colour(canvas, pill)
            odd_colour = self.colour_threshold is not None and colour_area > self.colour_threshold
            if odd_colour:
                anomaly_map = self.mark_colour(anomaly_map, colour_mask, self.threshold)
        resized = image.resize((self.image_size, self.image_size))
        return {
            "score": score,
            "threshold": self.threshold,
            "odd_colour": odd_colour,
            "is_anomalous": score > self.threshold or odd_colour,
            "latency_ms": latency_ms,
            "image": to_data_url(resized),
            "heatmap": to_data_url(overlay_heatmap(resized, anomaly_map, self.threshold)),
        }
