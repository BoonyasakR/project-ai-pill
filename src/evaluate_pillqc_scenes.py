"""Set and test the multi-pill threshold for pillQC tablets, as was done for MVTec.

pillQC has no multi-pill photos, so scenes are simulated: each crop's pill is cut out,
turned, resized, and pasted on a dark background, 8 pills per scene. `calibrate` sets
the threshold from the 54 validation pills only (5 rounds, spaced and touching). Run
`test` once after that, on the 230 test pills (3 seeds, spaced and touching).

    backend/.venv/bin/python src/evaluate_pillqc_scenes.py calibrate
    backend/.venv/bin/python src/evaluate_pillqc_scenes.py test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "src")]

from ad_metrics import pick_threshold  # noqa: E402
from evaluate_multi_pill import CALIBRATION_ROUNDS, TEST_SEEDS, run_pass, summarize  # noqa: E402
from inference import PILLQC_POSE, Inspector  # noqa: E402
from segmentation import pink_pill_mask, round_pill_region  # noqa: E402
import make_pill_scenes  # noqa: E402

# Round pills take more room than MVTec's ovals, so 8 of them need a larger floor.
make_pill_scenes.SCENE_SIZE = (1300, 1800)


def pill_mask(image):
    """The pink pill without floor dirt that touches it, which would be pasted along."""
    mask = pink_pill_mask(image)
    return mask & round_pill_region(mask)

IMAGES = ROOT / "data" / "pillqc" / "images"
SPLIT = ROOT / "splits" / "pillqc_seed42.json"
OUTPUT = ROOT / "results" / "pillqc_model" / "pillqc_scene_results.json"



def prepare(entries):
    """entries: (class folder, file name). Normal pills are labelled "good" for `summarize`."""
    pills = []
    for folder, name in entries:
        image = np.asarray(Image.open(IMAGES / folder / name).convert("RGB"))
        pills.append((f"{folder}/{name}", "good" if folder == "normal" else folder, image, pill_mask(image)))
    return pills


def calibrate(inspector):
    split = json.loads(SPLIT.read_text())
    pills = prepare([("normal", name) for name in split["validation"]])
    rows = []
    for seed in range(CALIBRATION_ROUNDS):
        for touching in (False, True):
            rows += run_pass(inspector, pills, seed, touching)[0]
    scores = [row["score"] for row in rows if row["score"] is not None]
    result = {
        "threshold": float(pick_threshold(scores)),
        "validation_pills": len(rows),
        "validation_missed_by_segmentation": len(rows) - len(scores),
        "validation_score_percentiles": {str(q): float(np.percentile(scores, q)) for q in (0, 50, 95, 99, 100)},
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def test(inspector):
    result = json.loads(OUTPUT.read_text())
    threshold = result["threshold"]
    split = json.loads(SPLIT.read_text())
    entries = [("normal", name) for name in split["test"]]
    entries += [(folder, path.name) for folder in ("chip", "dirt") for path in sorted((IMAGES / folder).glob("*.jpg"))]
    pills = prepare(entries)
    rows, counts = [], {}
    for seed in TEST_SEEDS:
        for touching in (False, True):
            pass_rows, found = run_pass(inspector, pills, seed, touching)
            rows += pass_rows
            counts[f"seed{seed}_touching{touching}"] = {"truth": len(pass_rows), "found": found}
            print(seed, touching, summarize(pass_rows, threshold), flush=True)
    result["segmentation_counts"] = counts
    result["pill_scores"] = rows
    result["overall"] = summarize(rows, threshold)
    result["spaced"] = summarize([row for row in rows if not row["touching"]], threshold)
    result["touching"] = summarize([row for row in rows if row["touching"]], threshold)
    result["by_defect"] = {
        defect: summarize([row for row in rows if row["defect"] == defect], threshold)
        for defect in sorted({row["defect"] for row in rows})
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("overall", "spaced", "touching", "by_defect")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["calibrate", "test"])
    stage = parser.parse_args().stage
    model = Inspector(ROOT / "models" / "patchcore_pillqc_256.pt", pose=PILLQC_POSE)
    (calibrate if stage == "calibrate" else test)(model)
