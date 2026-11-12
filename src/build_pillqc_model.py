"""Build the PatchCore model file for pillQC tablets, for the web prototype.

Same settings as the main MVTec model and the round 2 notebook: 256x256 images,
WideResNet50-2, layers 2+3, coreset 0.1, seed 42. The memory bank uses all 46
pool images and the threshold is the 99th percentile of the 54 validation images,
so this is the "all images, seed 42" run of round 2, rebuilt on this machine.

Scores count the pill only, inside `round_pill_region`. Whole-image scores let white
dirt on the floor of some normal validation crops set the threshold (55.21), so most
defects passed. Both are saved; the model file uses the pill-only threshold.

The memory bank also holds each pool pill cut out onto the plain fill used for
multi-pill scenes, so scene pills are compared with pills cut out the same way.

    backend/.venv/bin/python src/build_pillqc_model.py --root data/pillqc
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]
import ad_metrics  # noqa: E402
from inference import PILL_EDGE_PIXELS, PILLQC_POSE  # noqa: E402
from segmentation import align_pill, pink_pill_mask, round_pill_region  # noqa: E402

SIZE = 256
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
SEED = 42


def transform():
    return Compose([
        Resize((SIZE, SIZE), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def pill_region(path):
    """The scored pill area at model size, without the rim where the background bleeds in."""
    region = round_pill_region(pink_pill_mask(np.asarray(Image.open(path).convert("RGB"))))
    small = cv2.resize(region.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return cv2.erode(small, np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)) > 0


@torch.inference_mode()
def predict(model, paths, batch_size=8):
    """Return whole-image scores and pill-only scores."""
    whole, pill = [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = torch.stack([
            transform()(to_dtype(pil_to_tensor(Image.open(p).convert("RGB")), torch.float32, scale=True))
            for p in chunk
        ])
        output = model.model(batch)
        whole += output.pred_score.flatten().float().tolist()
        maps = output.anomaly_map.squeeze(1).float().numpy()
        pill += [float(score_map[pill_region(path)].max()) for score_map, path in zip(maps, chunk)]
    return np.array(whole), np.array(pill)


def train(pool_paths, val_paths, work):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from lightning.pytorch import seed_everything

    seed_everything(SEED)
    for name, paths in (("train_good", pool_paths), ("val_good", val_paths)):
        (work / name).mkdir()
        for path in paths:
            os.symlink(path, work / name / path.name)
    # Scene pills are cut out onto a plain fill, so their rim never touches the teal floor.
    # The memory bank also gets each pool pill cut out that way, or every scene rim looks abnormal.
    for path in pool_paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        mask = pink_pill_mask(image)
        view, _ = align_pill(image, mask & round_pill_region(mask), **PILLQC_POSE["align"])
        Image.fromarray(view).save(work / "train_good" / f"cut_{path.stem}.png")
    datamodule = Folder(
        name="pillqc", root=work, normal_dir="train_good", normal_test_dir="val_good",
        train_batch_size=8, eval_batch_size=8, num_workers=0,
        test_split_mode="from_dir", val_split_mode="same_as_test",
    )
    model = Patchcore(
        backbone=BACKBONE, layers=tuple(LAYERS), coreset_sampling_ratio=0.1,
        pre_processor=PreProcessor(transform=transform()), post_processor=False, evaluator=False, visualizer=False,
    )
    engine = Engine(default_root_dir=work / "runs", logger=False, accelerator="cpu", devices=1,
                    enable_progress_bar=False, num_sanity_val_steps=0)
    engine.fit(model=model, datamodule=datamodule)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="pillQC repository folder")
    parser.add_argument("--out", default=str(ROOT / "models" / "patchcore_pillqc_256.pt"))
    args = parser.parse_args()

    images = Path(args.root).resolve() / "images"
    split = json.loads((ROOT / "splits" / "pillqc_seed42.json").read_text())
    pool = [images / "normal" / name for name in split["pool"]]
    val = [images / "normal" / name for name in split["validation"]]
    test = [images / "normal" / name for name in split["test"]]
    test += sorted((images / "chip").glob("*.jpg")) + sorted((images / "dirt").glob("*.jpg"))
    groups = np.array([p.parent.name for p in test])
    labels = (groups != "normal").astype(int)

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as work:
        model = train(pool, val, Path(work))
    val_whole, val_scores = predict(model, val)
    threshold = ad_metrics.pick_threshold(val_scores)
    test_whole, test_scores = predict(model, test)
    whole_threshold = ad_metrics.pick_threshold(val_whole)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.model.state_dict(),
        "threshold": threshold,
        "image_size": SIZE,
        "backbone": BACKBONE,
        "layers": LAYERS,
        "score_area": "pill",
    }, out)

    summary = {
        "settings": {"size": SIZE, "backbone": BACKBONE, "layers": LAYERS, "coreset": 0.1, "seed": SEED,
                     "pool_images": len(pool), "cut_out_pool_images": len(pool),
                     "validation_images": len(val), "device": "cpu"},
        "threshold": threshold,
        "overall": ad_metrics.image_metrics(labels, test_scores, threshold),
        "recall_by_defect": {g: float(np.mean(test_scores[groups == g] > threshold)) for g in ("chip", "dirt")},
        "validation_scores": val_scores.tolist(),
        "test_scores": {f"{g}/{p.name}": float(s) for g, p, s in zip(groups, test, test_scores)},
        "score_area": "pill",
        "whole_image": {
            "threshold": whole_threshold,
            "overall": ad_metrics.image_metrics(labels, test_whole, whole_threshold),
            "recall_by_defect": {g: float(np.mean(test_whole[groups == g] > whole_threshold)) for g in ("chip", "dirt")},
            "validation_scores": val_whole.tolist(),
        },
        "build_seconds": round(time.perf_counter() - start, 1),
    }
    results = ROOT / "results" / "pillqc_model"
    results.mkdir(parents=True, exist_ok=True)
    (results / "pillqc_model_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("threshold", "overall", "recall_by_defect", "build_seconds")}, indent=1))


if __name__ == "__main__":
    main()
