"""Build the PatchCore model file for pillQC tablets, for the web prototype.

Same settings as the main MVTec model and the round 2 notebook: 256x256 images,
WideResNet50-2, layers 2+3, coreset 0.1, seed 42. The memory bank uses all 46
pool images and the threshold is the 99th percentile of the 54 validation images,
so this is the "all images, seed 42" run of round 2, rebuilt on this machine.

    backend/.venv/bin/python src/build_pillqc_model.py --root data/pillqc
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ad_metrics  # noqa: E402

SIZE = 256
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
SEED = 42


def transform():
    return Compose([
        Resize((SIZE, SIZE), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def predict(model, paths, batch_size=8):
    scores = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([
            transform()(to_dtype(pil_to_tensor(Image.open(p).convert("RGB")), torch.float32, scale=True))
            for p in paths[start:start + batch_size]
        ])
        scores.append(model.model(batch).pred_score.flatten().float().numpy())
    return np.concatenate(scores)


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
    val_scores = predict(model, val)
    threshold = ad_metrics.pick_threshold(val_scores)
    test_scores = predict(model, test)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.model.state_dict(),
        "threshold": threshold,
        "image_size": SIZE,
        "backbone": BACKBONE,
        "layers": LAYERS,
    }, out)

    summary = {
        "settings": {"size": SIZE, "backbone": BACKBONE, "layers": LAYERS, "coreset": 0.1, "seed": SEED,
                     "pool_images": len(pool), "validation_images": len(val), "device": "cpu"},
        "threshold": threshold,
        "overall": ad_metrics.image_metrics(labels, test_scores, threshold),
        "recall_by_defect": {g: float(np.mean(test_scores[groups == g] > threshold)) for g in ("chip", "dirt")},
        "validation_scores": val_scores.tolist(),
        "test_scores": {f"{g}/{p.name}": float(s) for g, p, s in zip(groups, test, test_scores)},
        "build_seconds": round(time.perf_counter() - start, 1),
    }
    results = ROOT / "results" / "pillqc_model"
    results.mkdir(parents=True, exist_ok=True)
    (results / "pillqc_model_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("threshold", "overall", "recall_by_defect", "build_seconds")}, indent=1))


if __name__ == "__main__":
    main()
