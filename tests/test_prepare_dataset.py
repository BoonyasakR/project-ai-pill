import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prepare_dataset as pd  # noqa: E402

# Synthetic per-class counts that add up to the real total of 141.
DEFECT_COUNTS = {
    "color": 25,
    "combined": 17,
    "contamination": 21,
    "crack": 26,
    "faulty_imprint": 19,
    "pill_type": 9,
    "scratch": 24,
}


def build_fake_dataset(root):
    pill = Path(root) / "mvtec" / "pill"
    for folder, count in (("train/good", 267), ("test/good", 26)):
        (pill / folder).mkdir(parents=True)
        for i in range(count):
            (pill / folder / f"{i:03d}.png").touch()
    for defect, count in DEFECT_COUNTS.items():
        (pill / "test" / defect).mkdir(parents=True)
        (pill / "ground_truth" / defect).mkdir(parents=True)
        for i in range(count):
            (pill / "test" / defect / f"{i:03d}.png").touch()
            (pill / "ground_truth" / defect / f"{i:03d}_mask.png").touch()
    return pill


class PrepareDatasetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)
        self.pill = build_fake_dataset(self.tmp)

    def test_finds_pill_from_parent_or_directly(self):
        self.assertEqual(pd.find_category_root(self.tmp), self.pill)
        self.assertEqual(pd.find_category_root(self.pill), self.pill)

    def test_missing_pill_folder_fails(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(pd.DatasetError):
                pd.find_category_root(empty)

    def test_valid_dataset_summary(self):
        summary = pd.inspect_dataset(self.pill)
        self.assertEqual(summary["train_good"], 267)
        self.assertEqual(summary["test_good"], 26)
        self.assertEqual(summary["test_defect"], 141)
        self.assertEqual(summary["defect_counts"], DEFECT_COUNTS)

    def test_missing_mask_fails(self):
        (self.pill / "ground_truth" / "crack" / "000_mask.png").unlink()
        with self.assertRaisesRegex(pd.DatasetError, "crack: 1 missing masks"):
            pd.inspect_dataset(self.pill)

    def test_wrong_count_fails(self):
        (self.pill / "train" / "good" / "000.png").unlink()
        with self.assertRaisesRegex(pd.DatasetError, "expected 267 images, found 266"):
            pd.inspect_dataset(self.pill)

    def test_missing_defect_folder_fails(self):
        for path in (self.pill / "test" / "scratch").iterdir():
            path.unlink()
        (self.pill / "test" / "scratch").rmdir()
        with self.assertRaisesRegex(pd.DatasetError, "missing \\['scratch'\\]"):
            pd.inspect_dataset(self.pill)

    def test_split_is_disjoint_complete_and_reproducible(self):
        first = pd.make_split(self.pill)
        second = pd.make_split(self.pill)
        self.assertEqual(first, second)
        self.assertEqual(len(first["validation"]), 40)
        self.assertEqual(len(first["train"]), 227)
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertEqual(len(set(first["train"]) | set(first["validation"])), 267)

    def test_different_seed_changes_split(self):
        self.assertNotEqual(
            pd.make_split(self.pill, seed=1)["validation"],
            pd.make_split(self.pill, seed=2)["validation"],
        )

    def test_invalid_ratio_fails(self):
        with self.assertRaises(ValueError):
            pd.make_split(self.pill, val_ratio=1)

    def test_cli_writes_split(self):
        out = self.tmp / "out" / "split.json"
        code = pd.main(["--root", str(self.tmp), "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out.read_text())["train"]), 227)

    def test_cli_reports_bad_dataset(self):
        (self.pill / "train" / "good" / "000.png").unlink()
        out = self.tmp / "split.json"
        self.assertEqual(pd.main(["--root", str(self.tmp), "--out", str(out)]), 1)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
