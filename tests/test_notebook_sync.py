import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookSyncTest(unittest.TestCase):
    """Notebook 04 runs on Colab from embedded copies, which must match the repository."""

    def setUp(self):
        notebook = json.loads((ROOT / "notebooks" / "04_reliability_fewshot.ipynb").read_text())
        self.sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    def test_embedded_code_matches_src(self):
        embedded = {}
        for source in self.sources:
            if source.startswith("%%writefile "):
                first, _, body = source.partition("\n")
                embedded[first.split()[1]] = body
        self.assertEqual(set(embedded), {"prepare_dataset.py", "ad_metrics.py", "bootstrap.py"})
        for name, body in embedded.items():
            self.assertEqual(body.rstrip("\n"), (ROOT / "src" / name).read_text().rstrip("\n"), name)

    def test_embedded_splits_match_split_files(self):
        found = dict(re.findall(r"^(\w+) = json\.loads\('''(.+)'''\)$", "\n".join(self.sources), re.MULTILINE))
        for variable, name in (("SPLIT", "pill_seed42.json"), ("PILLQC_SPLIT", "pillqc_seed42.json")):
            self.assertEqual(json.loads(found[variable]), json.loads((ROOT / "splits" / name).read_text()), name)


if __name__ == "__main__":
    unittest.main()
