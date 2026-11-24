# นำเข้า json: อ่านไฟล์ notebook และไฟล์แบ่งชุดข้อมูล | input: - | output: โมดูล json
import json
# นำเข้า re: ค้นหาข้อความด้วย regular expression | input: - | output: โมดูล re
import re
# นำเข้า unittest: เฟรมเวิร์กทดสอบมาตรฐาน | input: - | output: โมดูล unittest
import unittest
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# โฟลเดอร์รากของโปรเจกต์ | input: path ของไฟล์นี้ | output: ROOT = Path
ROOT = Path(__file__).resolve().parents[1]


# ชุดทดสอบว่า notebook ตรงกับ repository
class NotebookSyncTest(unittest.TestCase):
    """Notebook 04 runs on Colab from embedded copies, which must match the repository."""

    # เตรียมก่อนทุกเทสต์: อ่านโค้ดทุก cell ของ notebook | input: ไฟล์ notebooks/04_reliability_fewshot.ipynb | output: self.sources = list ของข้อความโค้ดต่อ cell
    def setUp(self):
        # อ่านไฟล์ notebook เป็น dict | input: path ของ notebook | output: notebook
        notebook = json.loads((ROOT / "notebooks" / "04_reliability_fewshot.ipynb").read_text())
        # ต่อบรรทัดของแต่ละ cell เป็นข้อความเดียว | input: notebook["cells"] | output: self.sources
        self.sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    # เทสต์ว่าโค้ดที่ฝังใน notebook ตรงกับไฟล์ใน src | input: cell ที่ขึ้นต้นด้วย %%writefile | output: เนื้อหาตรงกับ src/<ชื่อไฟล์>
    def test_embedded_code_matches_src(self):
        # dict เก็บโค้ดที่ฝัง {ชื่อไฟล์: เนื้อหา} | input: - | output: embedded = {}
        embedded = {}
        # วนทุก cell
        for source in self.sources:
            # ถ้าเป็น cell ที่เขียนไฟล์ (%%writefile)
            if source.startswith("%%writefile "):
                # แยกบรรทัดแรก (คำสั่ง) ออกจากเนื้อหา | input: source | output: first, body
                first, _, body = source.partition("\n")
                # เก็บเนื้อหาโดยใช้ชื่อไฟล์ (คำที่สองของบรรทัดแรก) เป็นคีย์ | input: first, body | output: embedded ถูกเพิ่ม
                embedded[first.split()[1]] = body
        # ต้องมีไฟล์ฝังครบ 3 ไฟล์ที่คาด | input: embedded | output: assertion
        self.assertEqual(set(embedded), {"prepare_dataset.py", "ad_metrics.py", "bootstrap.py"})
        # วนตรวจทีละไฟล์
        for name, body in embedded.items():
            # เนื้อหาที่ฝัง (ตัดบรรทัดใหม่ท้าย) ต้องเท่ากับไฟล์ใน src; ชื่อไฟล์ใช้เป็นข้อความเมื่อไม่ตรง | input: body, src/name | output: assertion
            self.assertEqual(body.rstrip("\n"), (ROOT / "src" / name).read_text().rstrip("\n"), name)

    # เทสต์ว่าการแบ่งชุดข้อมูลที่ฝังใน notebook ตรงกับไฟล์ใน splits | input: ตัวแปร SPLIT และ PILLQC_SPLIT ใน notebook | output: JSON ตรงกับไฟล์
    def test_embedded_splits_match_split_files(self):
        # ค้นหาบรรทัดรูปแบบ `ชื่อตัวแปร = json.loads('''...''')` ในทุก cell | input: self.sources | output: found = dict {ชื่อตัวแปร: ข้อความ JSON}
        found = dict(re.findall(r"^(\w+) = json\.loads\('''(.+)'''\)$", "\n".join(self.sources), re.MULTILINE))
        # วนตรวจสองคู่ (ตัวแปร, ชื่อไฟล์)
        for variable, name in (("SPLIT", "pill_seed42.json"), ("PILLQC_SPLIT", "pillqc_seed42.json")):
            # JSON ที่ฝังต้องเท่ากับ JSON ในไฟล์ splits | input: found[variable], splits/name | output: assertion
            self.assertEqual(json.loads(found[variable]), json.loads((ROOT / "splits" / name).read_text()), name)


# รันเทสต์ทั้งหมดเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เริ่มตัวรันเทสต์ | input: - | output: ผลผ่าน/ไม่ผ่าน
    unittest.main()
