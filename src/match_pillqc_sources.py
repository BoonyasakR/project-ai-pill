"""Find the source photo of every pillQC normal crop, then split the crops by photo.

The 149 normal crops were cut from 6 photos. A random split would put crops of one
photo in the pool, validation, and test sets at once, so the split is made by photo.

    python src/match_pillqc_sources.py --root data/pillqc --out splits/pillqc_seed42.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_dataset import make_pillqc_split

EXPECTED = {"normal": 149, "chip": 43, "dirt": 138}
MIN_SCORE = 0.95  # A crop matched to its own photo scores 0.98 or more.
SHRINK = 0.5  # Match at half size for speed.


def find_pills(image, min_area, max_area):
    """Centre and diameter of each pink pill on the teal background."""
    blue, green, red = (image[..., i].astype(np.int16) for i in range(3))
    mask = ((red > 150) & (red > green + 10)).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    return [
        (int(centres[i][0]), int(centres[i][1]), 2 * float(np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi)))
        for i in range(1, count)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]


def best_match(crop, photo, pills, pad):
    """Best template score of the crop around any pill of one photo, and that pill's index."""
    diameter = max(pill[2] for pill in find_pills(crop, 2000, crop.size))
    template = cv2.resize(crop, None, fx=SHRINK, fy=SHRINK, interpolation=cv2.INTER_AREA)
    best = (0.0, -1)
    for index, (x, y, pill_diameter) in enumerate(pills):
        scale = diameter / pill_diameter  # Crops were resized, so bring the photo to the crop's scale.
        half = min(int(0.9 * crop.shape[0] / scale), pad)
        window = photo[y + pad - half:y + pad + half, x + pad - half:x + pad + half]
        window = cv2.resize(window, None, fx=scale * SHRINK, fy=scale * SHRINK, interpolation=cv2.INTER_AREA)
        if min(window.shape[:2]) < template.shape[0]:
            continue
        score = float(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED).max())
        best = max(best, (score, index))
    return best


def match_sources(root, pad=600):
    """Return {photo name: [crop names]} for the normal crops."""
    root = Path(root)
    for name, expected in EXPECTED.items():
        found = len(list((root / "images" / name).glob("*.jpg")))
        if found != expected:
            raise SystemExit(f"{name}: expected {expected} images, found {found}")

    photos = {}
    for path in sorted((root / "sourceImages").glob("normal*.jpg")):
        image = cv2.imread(str(path))
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        photos[path.stem] = (padded, find_pills(image, 4000, 60000))

    groups, used = {name: [] for name in photos}, set()
    for path in sorted((root / "images" / "normal").glob("*.jpg")):
        crop = cv2.imread(str(path))
        (score, pill), name = max((best_match(crop, photo, pills, pad), name) for name, (photo, pills) in photos.items())
        # Every crop must be a different pill of its photo, or the matching is not trustworthy.
        if score < MIN_SCORE or (name, pill) in used:
            raise SystemExit(f"{path.name}: no trustworthy source photo (score {score:.3f}, {name} pill {pill})")
        used.add((name, pill))
        groups[name].append(path.name)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="pillQC repository folder")
    parser.add_argument("--out", default="splits/pillqc_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = match_sources(args.root)
    split = make_pillqc_split(groups, args.seed)
    split["sources"] = groups
    Path(args.out).write_text(json.dumps(split, indent=2) + "\n")
    for name, crops in groups.items():
        print(name, len(crops), crops[0], crops[-1])
    print({part: (split["photos"][part], len(split[part])) for part in ("test", "validation", "pool")})


if __name__ == "__main__":
    main()
