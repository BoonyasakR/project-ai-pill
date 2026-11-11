"""Validate the MVTec AD Pill folder and write a reproducible train/validation split.

Uses only the standard library so it runs anywhere, including Colab.

    python src/prepare_dataset.py --root /path/to/mvtec --out splits/pill_seed42.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

EXPECTED_TRAIN_GOOD = 267
EXPECTED_TEST_GOOD = 26
EXPECTED_TEST_DEFECT = 141
EXPECTED_DEFECTS = {
    "color",
    "combined",
    "contamination",
    "crack",
    "faulty_imprint",
    "pill_type",
    "scratch",
}


class DatasetError(Exception):
    """The dataset folder does not match the expected structure."""


def find_category_root(root, category="pill"):
    """Return the category folder, whether root is that folder or a parent of it."""
    root = Path(root)
    if (root / "train").is_dir() and (root / "test").is_dir():
        return root
    for candidate in sorted(root.rglob(category)):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise DatasetError(f"No '{category}' folder with train/ and test/ under {root}")


def _pngs(folder):
    return sorted(p.name for p in folder.glob("*.png"))


def inspect_dataset(pill_root):
    """Count images and check every defect image has a mask. Raises DatasetError."""
    pill_root = Path(pill_root)
    problems = []

    train_good = _pngs(pill_root / "train" / "good")
    test_good = _pngs(pill_root / "test" / "good")
    test_dirs = {p.name for p in (pill_root / "test").iterdir() if p.is_dir()}
    defects = sorted(test_dirs - {"good"})

    if set(defects) != EXPECTED_DEFECTS:
        problems.append(
            f"defect folders differ: missing {sorted(EXPECTED_DEFECTS - set(defects))}, "
            f"unexpected {sorted(set(defects) - EXPECTED_DEFECTS)}"
        )

    defect_counts = {}
    for defect in defects:
        images = _pngs(pill_root / "test" / defect)
        defect_counts[defect] = len(images)
        mask_dir = pill_root / "ground_truth" / defect
        masks = set(_pngs(mask_dir)) if mask_dir.is_dir() else set()
        expected_masks = {f"{Path(name).stem}_mask.png" for name in images}
        missing = sorted(expected_masks - masks)
        if missing:
            problems.append(f"{defect}: {len(missing)} missing masks, e.g. {missing[0]}")

    total_defect = sum(defect_counts.values())
    for label, actual, expected in (
        ("train/good", len(train_good), EXPECTED_TRAIN_GOOD),
        ("test/good", len(test_good), EXPECTED_TEST_GOOD),
        ("test defects", total_defect, EXPECTED_TEST_DEFECT),
    ):
        if actual != expected:
            problems.append(f"{label}: expected {expected} images, found {actual}")

    if problems:
        raise DatasetError("; ".join(problems))

    return {
        "train_good": len(train_good),
        "test_good": len(test_good),
        "test_defect": total_defect,
        "defect_counts": defect_counts,
    }


def make_split(pill_root, val_ratio=0.15, seed=42):
    """Split train/good only. The test set is never touched."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    names = _pngs(Path(pill_root) / "train" / "good")
    random.Random(seed).shuffle(names)
    n_val = round(len(names) * val_ratio)
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "train": sorted(names[n_val:]),
        "validation": sorted(names[:n_val]),
    }


def make_pillqc_split(groups, seed=42):
    """Split pillQC normal crops by source photo, so one photo never feeds two sets.

    groups: {photo name: [crop file names]}. Two photos go to test, two to
    validation, and the rest form the pool that memory banks are built from.
    """
    photos = sorted(groups)
    if len(photos) < 5:
        raise ValueError("need at least 5 source photos: 2 test, 2 validation, 1 pool")
    random.Random(seed).shuffle(photos)
    parts = {"test": photos[:2], "validation": photos[2:4], "pool": photos[4:]}
    split = {"seed": seed, "method": "by_source_photo", "photos": parts}
    for part, names in parts.items():
        split[part] = sorted(crop for name in names for crop in groups[name])
    return split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="pill folder or a parent of it")
    parser.add_argument("--out", default="splits/pill_seed42.json")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        pill_root = find_category_root(args.root)
        summary = inspect_dataset(pill_root)
    except DatasetError as error:
        print(f"Dataset check failed: {error}", file=sys.stderr)
        return 1

    split = make_split(pill_root, args.val_ratio, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"train={len(split['train'])} validation={len(split['validation'])} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
