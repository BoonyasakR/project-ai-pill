"""Build a PatchCore memory bank from one image of many normal pills.

The backbone stays frozen, as in the main model. Every pill is scored by a bank that has
not seen it (K-fold), because scoring a pill against itself would prove nothing. There are
no real defect labels, so simple painted defects stand in for them.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "src")]

from ad_metrics import pick_threshold  # noqa: E402
from inference import MARK_BAD, MARK_GOOD, Inspector  # noqa: E402
from segmentation import align_pill, find_pills, pill_pose  # noqa: E402
from torchvision.transforms.v2.functional import to_dtype  # noqa: E402

BASE_MODEL = ROOT / "models" / "patchcore_pill_256.pt"  # Source of the frozen backbone.
FOLDS = 5
# Name: shape, opacity, size as a share of the pill length.
DEFECTS = {
    "scratch": ("line", 1.0, 0.18),
    "chip": ("chip", 1.0, 0.06),
    "stain": ("spot", 0.6, 0.07),
    "faint_scratch": ("line", 0.35, 0.10),
    "small_chip": ("chip", 1.0, 0.025),
    "faint_stain": ("spot", 0.25, 0.04),
}
AUGMENTS = ((3, 1.05), (-3, 0.95))  # Rotation in degrees, brightness factor.


def embed(inspector, view):
    """Return the patch features of one aligned pill."""
    tensor = to_dtype(torch.from_numpy(view).permute(2, 0, 1), torch.float32, scale=True)
    inspector.model.training = True  # Only the top module, so the backbone stays in eval mode.
    with torch.no_grad():
        features = inspector.model(inspector.transform(tensor).unsqueeze(0))
    inspector.model.training = False
    inspector.model.embedding_store.clear()
    return features


def augmented(view):
    """Return the view plus slightly rotated and relit copies."""
    centre = (view.shape[1] / 2, view.shape[0] / 2)
    copies = [view]
    for angle, brightness in AUGMENTS:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1)
        turned = cv2.warpAffine(view, matrix, view.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
        copies.append((turned * brightness).clip(0, 255).astype(np.uint8))
    return copies


def build_bank(features, indices, rng, bank_share):
    """The long axis has no direction, so both orientations of each pill go in.

    ponytail: random patches, use KCenterGreedy if rare normal detail gets flagged.
    """
    patches = torch.cat([view for index in indices for view in features[index]])
    keep = rng.permutation(len(patches))[:int(len(patches) * bank_share)]
    return patches[torch.from_numpy(keep)]


def paint_defect(image, mask, kind, rng):
    """Return a copy of the image with one simple artificial defect on this pill."""
    shape, opacity, size = DEFECTS[kind]
    layer = image.copy()
    centre, _, length = pill_pose(mask)
    turn = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(turn), np.sin(turn)])
    spot = centre + direction * length * rng.uniform(0.05, 0.2)
    if shape == "line":
        ends = [tuple(int(v) for v in spot + side * direction[::-1] * length * size) for side in (-1, 1)]
        cv2.line(layer, ends[0], ends[1], (95, 90, 90), 2, cv2.LINE_AA)
    elif shape == "spot":
        cv2.circle(layer, tuple(int(v) for v in spot), int(length * size), (150, 110, 70), -1, cv2.LINE_AA)
    else:
        ys, xs = np.nonzero(mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        edge = int(rng.integers(len(xs)))
        cv2.circle(layer, (int(xs[edge]), int(ys[edge])), int(length * size), (35, 35, 38), -1, cv2.LINE_AA)
    return cv2.addWeighted(layer, opacity, image, 1 - opacity, 0)


def main(image_path, output, bank_share, augment, save_path):
    rng = np.random.default_rng(42)
    inspector = Inspector(BASE_MODEL)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    masks = find_pills(image)
    everything = np.any(masks, axis=0)
    others = [everything & ~mask for mask in masks]
    features = []
    for mask, other in zip(masks, others):
        views = [align_pill(image, mask, other, flip)[0] for flip in (False, True)]
        if augment:
            views = [copy for view in views for copy in augmented(view)]
        features.append([embed(inspector, view) for view in views])
    print(f"{len(masks)} pills embedded", flush=True)

    folds = np.array_split(rng.permutation(len(masks)), FOLDS)
    normal = np.zeros(len(masks))
    painted = {kind: np.zeros(len(masks)) for kind in DEFECTS}
    for fold in folds:
        rest = [index for index in range(len(masks)) if index not in set(fold)]
        inspector.model.memory_bank = build_bank(features, rest, rng, bank_share)
        for index in fold:
            normal[index] = inspector.score_pill(image, masks[index], others[index])[0]
            for number, kind in enumerate(DEFECTS):
                # Seeded per pill and defect, so every training setup sees the same defects.
                damaged = paint_defect(image, masks[index], kind, np.random.default_rng([42, index, number]))
                painted[kind][index] = inspector.score_pill(damaged, masks[index], others[index])[0]
        print("fold done", flush=True)

    # Each fold's threshold comes from the other folds' normal pills.
    thresholds = np.zeros(len(masks))
    for fold in folds:
        thresholds[fold] = pick_threshold(np.delete(normal, fold))
    result = {
        "pills": len(masks),
        "bank_share": bank_share,
        "augment": augment,
        "assumption": "All pills in the image are treated as normal.",
        "threshold": pick_threshold(normal),
        "normal_scores": normal.tolist(),
        "false_alarms": int((normal > thresholds).sum()),
        "painted_defects": {
            kind: {
                "caught": int((scores > thresholds).sum()),
                "auroc": roc_auc_score([0] * len(normal) + [1] * len(scores), [*normal, *scores]),
            }
            for kind, scores in painted.items()
        },
    }

    if save_path:
        checkpoint = torch.load(BASE_MODEL, map_location="cpu", weights_only=True)
        checkpoint["state_dict"]["memory_bank"] = build_bank(features, range(len(masks)), rng, bank_share)
        checkpoint["scene_threshold"] = result["threshold"]
        torch.save(checkpoint, save_path)

    marked = image.copy()
    for number, (mask, score, limit) in enumerate(zip(masks, normal, thresholds), start=1):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, contours, -1, MARK_BAD if score > limit else MARK_GOOD, 3)

    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(marked).save(output / "held_out_result.png")
    (output / "photo_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "normal_scores"}, indent=2))
    print("normal", np.round(np.percentile(normal, [0, 50, 100]), 2))
    for kind, scores in painted.items():
        print(kind, np.round(np.percentile(scores, [0, 50, 100]), 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "photo_model")
    parser.add_argument("--bank-share", type=float, default=0.2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--save", type=Path, help="Also save a model built from all pills")
    arguments = parser.parse_args()
    main(arguments.image, arguments.output, arguments.bank_share, arguments.augment, arguments.save)
