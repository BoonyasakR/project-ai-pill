"""Evaluate per-pill inspection on simulated multi-pill scenes.

Run `calibrate` first. It sets the threshold from validation pills only. Run `test` once after that.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "src")]

from ad_metrics import pick_threshold  # noqa: E402
from inference import Inspector  # noqa: E402
from make_pill_scenes import PILLS_PER_SCENE, compose_scene, cut_pill  # noqa: E402
from segmentation import find_pills  # noqa: E402

DATA = ROOT / "data" / "mvtec" / "pill"
OUTPUT = ROOT / "results" / "multi_pill" / "multi_pill_results.json"
MIN_IOU = 0.5
CALIBRATION_ROUNDS = 5
TEST_SEEDS = (1, 2, 3)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def run_scene(inspector, items, rng, touching):
    """Score one scene. Each item is (name, defect, RGB image, pill mask)."""
    cutouts = [cut_pill(image, mask, rng) for _, _, image, mask in items]
    scene, truth = compose_scene(cutouts, rng, touching)
    found = find_pills(scene)
    everything = np.any(found, axis=0) if found else None
    rows = []
    for (name, defect, _, _), mask in zip(items, truth):
        overlaps = [(mask & other).sum() / (mask | other).sum() for other in found]
        best = int(np.argmax(overlaps)) if overlaps else -1
        row = {"name": name, "defect": defect, "touching": touching, "score": None}
        if overlaps and overlaps[best] >= MIN_IOU:
            row["score"] = inspector.score_pill(scene, found[best], everything & ~found[best])[0]
        rows.append(row)
    return rows, len(found)


def run_pass(inspector, pills, seed, touching):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pills))
    rows, found = [], 0
    for start in range(0, len(order), PILLS_PER_SCENE):
        chunk = [pills[index] for index in order[start:start + PILLS_PER_SCENE]]
        scene_rows, scene_found = run_scene(inspector, chunk, rng, touching)
        rows += scene_rows
        found += scene_found
    return rows, found


def prepare(paths):
    """Load pills. Each source image must hold exactly one pill."""
    pills = []
    for path, defect in paths:
        image = load(path)
        masks = find_pills(image)
        if len(masks) != 1:
            raise ValueError(f"{path} gave {len(masks)} pills")
        pills.append((f"{path.parent.name}/{path.name}", defect, image, masks[0]))
    return pills


def calibrate(inspector):
    split = json.loads((ROOT / "splits" / "pill_seed42.json").read_text())
    pills = prepare([(DATA / "train" / "good" / name, "good") for name in split["validation"]])
    rows = []
    for seed in range(CALIBRATION_ROUNDS):
        for touching in (False, True):
            rows += run_pass(inspector, pills, seed, touching)[0]
    scores = [row["score"] for row in rows if row["score"] is not None]
    result = {
        "threshold": float(pick_threshold(scores)),
        "validation_pills": len(rows),
        "validation_missed_by_segmentation": len(rows) - len(scores),
        "validation_score_percentiles": {
            str(q): float(np.percentile(scores, q)) for q in (0, 50, 95, 99, 100)
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def summarize(rows, threshold):
    defective = [row for row in rows if row["defect"] != "good"]
    normal = [row for row in rows if row["defect"] == "good"]

    def flagged(row):
        return row["score"] is not None and row["score"] > threshold

    true_positive = sum(flagged(row) for row in defective)
    false_positive = sum(flagged(row) for row in normal)
    scored = [row for row in rows if row["score"] is not None]
    recall = true_positive / max(len(defective), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    summary = {
        "pills": len(rows),
        "missed_by_segmentation": len(rows) - len(scored),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": len(defective) - true_positive,
        "true_negative": len(normal) - false_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }
    if normal and defective:
        labels = [row["defect"] != "good" for row in scored]
        summary["auroc"] = roc_auc_score(labels, [row["score"] for row in scored])
    return summary


def test(inspector):
    result = json.loads(OUTPUT.read_text())
    threshold = result["threshold"]
    paths = [
        (path, path.parent.name)
        for path in sorted((DATA / "test").glob("*/*.png"))
    ]
    pills = prepare(paths)
    rows, scene_counts = [], {}
    for seed in TEST_SEEDS:
        for touching in (False, True):
            pass_rows, found = run_pass(inspector, pills, seed, touching)
            rows += pass_rows
            scene_counts[f"seed{seed}_touching{touching}"] = {"truth": len(pass_rows), "found": found}
            print(seed, touching, summarize(pass_rows, threshold), flush=True)
    result["segmentation_counts"] = scene_counts
    result["pill_scores"] = rows
    result["overall"] = summarize(rows, threshold)
    result["spaced"] = summarize([row for row in rows if not row["touching"]], threshold)
    result["touching"] = summarize([row for row in rows if row["touching"]], threshold)
    result["by_defect"] = {
        defect: summarize([row for row in rows if row["defect"] == defect], threshold)
        for defect in sorted({row["defect"] for row in rows})
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("overall", "spaced", "touching")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["calibrate", "test"])
    stage = parser.parse_args().stage
    model = Inspector(ROOT / "models" / "patchcore_pill_256.pt")
    (calibrate if stage == "calibrate" else test)(model)
