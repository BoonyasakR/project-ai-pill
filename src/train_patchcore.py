"""Train PatchCore on this machine and evaluate it, with the same settings as the Colab notebooks.

    backend/.venv/bin/python src/train_patchcore.py --size 256 --out models/patchcore_pill_256.pt

The threshold comes from normal validation images only. Test results are printed and saved.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from anomalib.models.components.dimensionality_reduction import SparseRandomProjection
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from lightning.pytorch import seed_everything
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]
import ad_metrics  # noqa: E402
from colour_check import MIN_AREA, build_seen, novel_colour  # noqa: E402
from segmentation import find_pills  # noqa: E402

DATA = ROOT / "data" / "mvtec" / "pill"
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 8


def make_transform(size):
    return Compose([
        Resize((size, size), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(path):
    return to_dtype(pil_to_tensor(Image.open(path).convert("RGB")), torch.float32, scale=True)


def batches(paths, size):
    transform = make_transform(size)
    for start in range(0, len(paths), BATCH):
        yield torch.stack([transform(load_image(path)) for path in paths[start:start + BATCH]]).to(DEVICE)


def coreset_indices(embedding, ratio):
    """k-center greedy selection, as in anomalib's KCenterGreedy, picking the same points.

    Distances come from one matrix-vector product per step, which is far faster on MPS than
    anomalib's per-row norm.
    """
    features = SparseRandomProjection(eps=0.9).fit(embedding).transform(embedding).to(DEVICE)
    squared = (features * features).sum(1)
    nearest = torch.full_like(squared, float("inf"))
    index = torch.randint(len(features), (1,), device=features.device).squeeze()
    chosen = [index]
    for _ in range(int(len(features) * ratio) - 1):
        distance = (squared + squared[index] - 2 * (features @ features[index])).clamp_min_(0)
        nearest = torch.minimum(nearest, distance)
        nearest[index] = 0
        index = torch.argmax(nearest)
        chosen.append(index)
    return torch.stack(chosen)


def train(paths, size, coreset):
    seed_everything(42)
    model = PatchcoreModel(layers=LAYERS, backbone=BACKBONE).to(DEVICE)
    model.train()
    # Fill one CPU tensor batch by batch. At 512 pixels the features take 5.7 GB, too much for MPS.
    embedding, row = None, 0
    for batch in batches(paths, size):
        part = model(batch).cpu()
        model.embedding_store.clear()
        if embedding is None:
            embedding = torch.empty(len(paths) * len(part) // len(batch), part.shape[1])
        embedding[row:row + len(part)] = part
        row += len(part)
    model.memory_bank = embedding[coreset_indices(embedding, coreset).cpu()].to(DEVICE)
    return model.eval()


@torch.inference_mode()
def predict(model, paths, size):
    scores, maps = [], []
    for batch in batches(paths, size):
        output = model(batch)
        scores.append(output.pred_score.flatten().float().cpu().numpy())
        maps.append(output.anomaly_map.squeeze(1).float().cpu().numpy())
    return np.concatenate(scores), np.concatenate(maps)


def load_mask(path, size):
    if path.parent.name == "good":
        return np.zeros((size, size), dtype=bool)
    mask_path = DATA / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    return np.asarray(Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)) > 0


def pill(path):
    image = np.asarray(Image.open(path).convert("RGB"))
    return image, find_pills(image)[0]


def colour_areas(seen, paths):
    return np.array([novel_colour(seen, *pill(path))[0] for path in paths])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads((ROOT / "splits" / "pill_seed42.json").read_text())
    train_paths = [DATA / "train" / "good" / name for name in split["train"]]
    val_paths = [DATA / "train" / "good" / name for name in split["validation"]]
    test_paths = sorted((DATA / "test").glob("*/*.png"))
    groups = [path.parent.name for path in test_paths]

    start = time.perf_counter()
    model = train(train_paths, args.size, args.coreset)
    print("fit seconds", round(time.perf_counter() - start, 1), "memory bank", tuple(model.memory_bank.shape))
    val_scores, _ = predict(model, val_paths, args.size)
    threshold = ad_metrics.pick_threshold(val_scores)

    scores, maps = predict(model, test_paths, args.size)
    masks = np.stack([load_mask(path, args.size) for path in test_paths])
    labels = [int(group != "good") for group in groups]
    result = {
        "overall": ad_metrics.evaluate(labels, scores, masks, maps, threshold),
        "by_defect": ad_metrics.evaluate_by_defect(groups, scores, masks, maps, threshold),
        "validation_scores": val_scores.tolist(),
        "test_scores": {f"{g}/{p.name}": float(s) for g, p, s in zip(groups, test_paths, scores)},
    }
    print("patchcore", json.dumps({key: round(value, 4) for key, value in result["overall"].items()}))
    for defect, row in result["by_defect"].items():
        print(defect, json.dumps({key: round(value, 3) for key, value in row.items()}))

    # Colour check: colours from training pills, threshold from validation pills.
    seen = build_seen(pill(path) for path in train_paths)
    val_areas = colour_areas(seen, val_paths)
    colour_threshold = max(float(np.quantile(val_areas, 0.99)), MIN_AREA)
    areas = colour_areas(seen, test_paths)
    # One score for AUROC: each check divided by its own threshold, larger one kept.
    combined = np.maximum(scores / threshold, areas / colour_threshold)
    result["with_colour"] = {
        "colour_threshold": colour_threshold,
        "validation_colour_areas": val_areas.tolist(),
        "test_colour_areas": {f"{g}/{p.name}": int(a) for g, p, a in zip(groups, test_paths, areas)},
        "overall": ad_metrics.evaluate(labels, combined, masks, maps, 1.0),
        "by_defect": ad_metrics.evaluate_by_defect(groups, combined, masks, maps, 1.0),
    }
    print("with colour", json.dumps({key: round(value, 4) for key, value in result["with_colour"]["overall"].items()}))
    for defect, row in result["with_colour"]["by_defect"].items():
        print(defect, json.dumps({key: round(value, 3) for key, value in row.items()}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "threshold": threshold,
        "image_size": args.size,
        "backbone": BACKBONE,
        "layers": LAYERS,
        "colour_seen": torch.from_numpy(np.packbits(seen)),
        "colour_threshold": colour_threshold,
    }, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
