# นำเข้า json: อ่านไฟล์ผลลัพธ์ | input: - | output: โมดูล json
import json
# นำเข้า sys: แก้ sys.path | input: - | output: โมดูล sys
import sys
# นำเข้า tempfile: สร้างโฟลเดอร์ชั่วคราว | input: - | output: โมดูล tempfile
import tempfile
# นำเข้า unittest: เฟรมเวิร์กทดสอบมาตรฐาน | input: - | output: โมดูล unittest
import unittest
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# เพิ่มโฟลเดอร์ src ไว้หน้าสุดของ sys.path | input: path ของไฟล์นี้ | output: sys.path ถูกแก้ไข
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# นำเข้าโมดูล prepare_dataset โดยตั้งชื่อย่อ pd | input: - | output: โมดูล pd
import prepare_dataset as pd  # noqa: E402

# Synthetic per-class counts that add up to the real total of 141.
# จำนวนรูปจำลองต่อชนิดความบกพร่อง รวมได้ 141 เท่าของจริง | input: - | output: DEFECT_COUNTS = dict
DEFECT_COUNTS = {
    # สี 25 รูป
    "color": 25,
    # รวมหลายชนิด 17 รูป
    "combined": 17,
    # สิ่งปนเปื้อน 21 รูป
    "contamination": 21,
    # รอยแตก 26 รูป
    "crack": 26,
    # ตัวพิมพ์ผิด 19 รูป
    "faulty_imprint": 19,
    # ชนิดยาผิด 9 รูป
    "pill_type": 9,
    # รอยขีดข่วน 24 รูป
    "scratch": 24,
}


# ฟังก์ชันสร้างโฟลเดอร์ชุดข้อมูลปลอม (ไฟล์ว่าง) ตามโครงสร้างจริง | input: root = โฟลเดอร์ราก | output: Path ของโฟลเดอร์ pill ที่สร้าง
def build_fake_dataset(root):
    # path โฟลเดอร์ pill ปลอม | input: root | output: pill = Path
    pill = Path(root) / "mvtec" / "pill"
    # วนสร้างรูปปกติของชุดฝึก 267 รูป และชุดทดสอบ 26 รูป
    for folder, count in (("train/good", 267), ("test/good", 26)):
        # สร้างโฟลเดอร์ (รวมโฟลเดอร์แม่) | input: folder | output: โฟลเดอร์ใหม่
        (pill / folder).mkdir(parents=True)
        # วนสร้างไฟล์ตามจำนวน
        for i in range(count):
            # สร้างไฟล์ว่างชื่อเลขสามหลัก เช่น 000.png | input: i | output: ไฟล์ใหม่
            (pill / folder / f"{i:03d}.png").touch()
    # วนสร้างรูปบกพร่องและ mask ตามชนิด
    for defect, count in DEFECT_COUNTS.items():
        # สร้างโฟลเดอร์รูปทดสอบของชนิดนี้ | input: defect | output: โฟลเดอร์ใหม่
        (pill / "test" / defect).mkdir(parents=True)
        # สร้างโฟลเดอร์ mask ของชนิดนี้ | input: defect | output: โฟลเดอร์ใหม่
        (pill / "ground_truth" / defect).mkdir(parents=True)
        # วนสร้างไฟล์ตามจำนวน
        for i in range(count):
            # สร้างไฟล์รูปว่าง | input: i | output: ไฟล์ใหม่
            (pill / "test" / defect / f"{i:03d}.png").touch()
            # สร้างไฟล์ mask ว่างชื่อ <เลข>_mask.png | input: i | output: ไฟล์ใหม่
            (pill / "ground_truth" / defect / f"{i:03d}_mask.png").touch()
    # คืน path โฟลเดอร์ pill | input: pill | output: Path
    return pill


# ชุดทดสอบสคริปต์เตรียมชุดข้อมูล
class PrepareDatasetTest(unittest.TestCase):
    # เตรียมก่อนทุกเทสต์: สร้างชุดข้อมูลปลอมในโฟลเดอร์ชั่วคราว | input: - | output: self.tmp, self.pill
    def setUp(self):
        # สร้างโฟลเดอร์ชั่วคราว | input: - | output: temp
        temp = tempfile.TemporaryDirectory()
        # ลงทะเบียนให้ลบโฟลเดอร์หลังเทสต์จบ | input: temp.cleanup | output: cleanup ถูกจัดไว้
        self.addCleanup(temp.cleanup)
        # เก็บ path โฟลเดอร์ชั่วคราว | input: temp.name | output: self.tmp = Path
        self.tmp = Path(temp.name)
        # สร้างชุดข้อมูลปลอม | input: self.tmp | output: self.pill = Path
        self.pill = build_fake_dataset(self.tmp)

    # เทสต์ว่าหาโฟลเดอร์ pill ได้ทั้งจากโฟลเดอร์แม่และโดยตรง | input: self.tmp และ self.pill | output: ได้ self.pill ทั้งคู่
    def test_finds_pill_from_parent_or_directly(self):
        # ส่งโฟลเดอร์แม่ ต้องหา pill เจอ | input: self.tmp | output: assertion
        self.assertEqual(pd.find_category_root(self.tmp), self.pill)
        # ส่งโฟลเดอร์ pill เอง ต้องคืนตัวมันเอง | input: self.pill | output: assertion
        self.assertEqual(pd.find_category_root(self.pill), self.pill)

    # เทสต์ว่าไม่พบโฟลเดอร์ pill ต้องเกิด error | input: โฟลเดอร์ว่าง | output: DatasetError
    def test_missing_pill_folder_fails(self):
        # สร้างโฟลเดอร์ว่างชั่วคราว
        with tempfile.TemporaryDirectory() as empty:
            # ต้องเกิด DatasetError
            with self.assertRaises(pd.DatasetError):
                # ค้นหา pill ในโฟลเดอร์ว่าง | input: empty | output: ยกข้อยกเว้น
                pd.find_category_root(empty)

    # เทสต์ว่าสรุปชุดข้อมูลที่ถูกต้องได้จำนวนตามที่คาด | input: ชุดข้อมูลปลอม | output: 267, 26, 141 และจำนวนต่อชนิด
    def test_valid_dataset_summary(self):
        # ตรวจชุดข้อมูล | input: self.pill | output: summary = dict
        summary = pd.inspect_dataset(self.pill)
        # รูปปกติชุดฝึก = 267 | input: summary | output: assertion
        self.assertEqual(summary["train_good"], 267)
        # รูปปกติชุดทดสอบ = 26 | input: summary | output: assertion
        self.assertEqual(summary["test_good"], 26)
        # รูปบกพร่อง = 141 | input: summary | output: assertion
        self.assertEqual(summary["test_defect"], 141)
        # จำนวนต่อชนิดตรงกับ DEFECT_COUNTS | input: summary | output: assertion
        self.assertEqual(summary["defect_counts"], DEFECT_COUNTS)

    # เทสต์ว่า mask ขาดต้องเกิด error | input: ลบ mask ของ crack 1 ไฟล์ | output: DatasetError ที่มีข้อความ "crack: 1 missing masks"
    def test_missing_mask_fails(self):
        # ลบไฟล์ mask 1 ไฟล์ | input: path | output: ไฟล์ถูกลบ
        (self.pill / "ground_truth" / "crack" / "000_mask.png").unlink()
        # ต้องเกิด DatasetError ที่ข้อความตรงรูปแบบ
        with self.assertRaisesRegex(pd.DatasetError, "crack: 1 missing masks"):
            # ตรวจชุดข้อมูล | input: self.pill | output: ยกข้อยกเว้น
            pd.inspect_dataset(self.pill)

    # เทสต์ว่าจำนวนรูปผิดต้องเกิด error | input: ลบรูปฝึก 1 ไฟล์ | output: DatasetError ที่มีข้อความ "expected 267 images, found 266"
    def test_wrong_count_fails(self):
        # ลบรูปฝึก 1 ไฟล์ | input: path | output: ไฟล์ถูกลบ
        (self.pill / "train" / "good" / "000.png").unlink()
        # ต้องเกิด DatasetError ที่ข้อความตรงรูปแบบ
        with self.assertRaisesRegex(pd.DatasetError, "expected 267 images, found 266"):
            # ตรวจชุดข้อมูล | input: self.pill | output: ยกข้อยกเว้น
            pd.inspect_dataset(self.pill)

    # เทสต์ว่าโฟลเดอร์ความบกพร่องหายต้องเกิด error | input: ลบโฟลเดอร์ scratch ทั้งโฟลเดอร์ | output: DatasetError ที่มีข้อความ "missing ['scratch']"
    def test_missing_defect_folder_fails(self):
        # ลบทุกไฟล์ในโฟลเดอร์ scratch
        for path in (self.pill / "test" / "scratch").iterdir():
            # ลบไฟล์ | input: path | output: ไฟล์ถูกลบ
            path.unlink()
        # ลบโฟลเดอร์ที่ว่างแล้ว | input: path | output: โฟลเดอร์ถูกลบ
        (self.pill / "test" / "scratch").rmdir()
        # ต้องเกิด DatasetError ที่ข้อความตรงรูปแบบ (ใช้ \\[ เพื่อ escape วงเล็บเหลี่ยมของ regex)
        with self.assertRaisesRegex(pd.DatasetError, "missing \\['scratch'\\]"):
            # ตรวจชุดข้อมูล | input: self.pill | output: ยกข้อยกเว้น
            pd.inspect_dataset(self.pill)

    # เทสต์ว่าการแบ่งชุดไม่ทับกัน ครบถ้วน และซ้ำได้ | input: make_split สองครั้ง | output: ผลเท่ากัน validation 40, train 227, ไม่ซ้อนกัน, รวม 267
    def test_split_is_disjoint_complete_and_reproducible(self):
        # แบ่งชุดครั้งที่ 1 | input: self.pill | output: first = dict
        first = pd.make_split(self.pill)
        # แบ่งชุดครั้งที่ 2 | input: self.pill | output: second = dict
        second = pd.make_split(self.pill)
        # สองครั้งต้องได้ผลเท่ากัน | input: first, second | output: assertion
        self.assertEqual(first, second)
        # validation = 15% ของ 267 ปัดเป็น 40 | input: first | output: assertion
        self.assertEqual(len(first["validation"]), 40)
        # train = 267 - 40 = 227 | input: first | output: assertion
        self.assertEqual(len(first["train"]), 227)
        # train กับ validation ต้องไม่มีไฟล์ซ้ำกัน | input: first | output: assertion
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        # รวมสองชุดต้องได้ครบ 267 ไฟล์ | input: first | output: assertion
        self.assertEqual(len(set(first["train"]) | set(first["validation"])), 267)

    # เทสต์ว่า seed ต่างกันได้การแบ่งต่างกัน | input: seed 1 และ 2 | output: validation ต่างกัน
    def test_different_seed_changes_split(self):
        # ยืนยันว่า validation ของสอง seed ต้องไม่เท่ากัน | input: ผลของ make_split สอง seed | output: assertion
        self.assertNotEqual(
            # validation ของ seed 1 | input: self.pill, seed=1 | output: list
            pd.make_split(self.pill, seed=1)["validation"],
            # validation ของ seed 2 | input: self.pill, seed=2 | output: list
            pd.make_split(self.pill, seed=2)["validation"],
        )

    # เทสต์ว่าสัดส่วนที่ไม่ถูกต้องต้องเกิด error | input: val_ratio=1 | output: ValueError
    def test_invalid_ratio_fails(self):
        # ต้องเกิด ValueError
        with self.assertRaises(ValueError):
            # แบ่งชุดด้วยสัดส่วน 1 (ไม่อยู่ระหว่าง 0 และ 1) | input: self.pill, val_ratio=1 | output: ยกข้อยกเว้น
            pd.make_split(self.pill, val_ratio=1)

    # เทสต์ว่าการแบ่งชุด pillQC เก็บแต่ละรูปต้นฉบับไว้ชุดเดียว | input: 6 รูปต้นฉบับ รูปละ 3-8 ครอป | output: แบ่ง 2/2/2 รูป, ครอปครบ ไม่ซ้ำ, และน้อยกว่า 5 รูปต้นฉบับต้องเกิด error
    def test_pillqc_split_keeps_each_photo_in_one_set(self):
        # สร้างกลุ่มรูปต้นฉบับ 6 รูป จำนวนครอปต่างกัน | input: - | output: groups = dict
        groups = {f"photo{p}": [f"p{p}_{i}.jpg" for i in range(3 + p)] for p in range(6)}
        # แบ่งชุด | input: groups | output: split = dict
        split = pd.make_pillqc_split(groups)
        # แบ่งซ้ำต้องได้ผลเหมือนเดิม | input: groups | output: assertion
        self.assertEqual(split, pd.make_pillqc_split(groups))
        # จำนวนรูปต้นฉบับของ test/validation/pool ต้องเป็น 2, 2, 2 | input: split | output: assertion
        self.assertEqual([len(split["photos"][part]) for part in ("test", "validation", "pool")], [2, 2, 2])
        # วนตรวจแต่ละชุด
        for part, photos in split["photos"].items():
            # ครอปของชุดนั้นต้องเท่ากับครอปของรูปต้นฉบับในชุด (เรียงลำดับ) | input: split[part], photos | output: assertion
            self.assertEqual(split[part], sorted(crop for photo in photos for crop in groups[photo]))
        # รวมครอปของทั้งสามชุด | input: split | output: everything = list
        everything = split["test"] + split["validation"] + split["pool"]
        # ต้องได้ครอปครบทุกไฟล์ ไม่ขาดไม่ซ้ำ | input: everything, groups | output: assertion
        self.assertEqual(sorted(everything), sorted(crop for crops in groups.values() for crop in crops))
        # มีรูปต้นฉบับ 4 รูป (< 5) ต้องเกิด ValueError
        with self.assertRaises(ValueError):
            # แบ่งชุดจาก 4 รูปต้นฉบับ | input: dict 4 รูป | output: ยกข้อยกเว้น
            pd.make_pillqc_split({f"photo{p}": ["a.jpg"] for p in range(4)})

    # เทสต์ว่าบรรทัดคำสั่งเขียนไฟล์แบ่งชุดสำเร็จ | input: --root, --out | output: exit code 0 และไฟล์ที่มี train 227 รูป
    def test_cli_writes_split(self):
        # path ไฟล์ผลลัพธ์ในโฟลเดอร์ย่อยที่ยังไม่มี | input: self.tmp | output: out = Path
        out = self.tmp / "out" / "split.json"
        # รัน main ด้วยอาร์กิวเมนต์ | input: ["--root", ..., "--out", ...] | output: code = int
        code = pd.main(["--root", str(self.tmp), "--out", str(out)])
        # exit code ต้องเป็น 0 | input: code | output: assertion
        self.assertEqual(code, 0)
        # ไฟล์ที่เขียนต้องมี train 227 รูป | input: out | output: assertion
        self.assertEqual(len(json.loads(out.read_text())["train"]), 227)

    # เทสต์ว่าบรรทัดคำสั่งรายงานชุดข้อมูลที่ไม่ถูกต้อง | input: ลบรูปฝึก 1 ไฟล์ | output: exit code 1 และไม่เขียนไฟล์
    def test_cli_reports_bad_dataset(self):
        # ลบรูปฝึก 1 ไฟล์ให้ชุดข้อมูลผิด | input: path | output: ไฟล์ถูกลบ
        (self.pill / "train" / "good" / "000.png").unlink()
        # path ไฟล์ผลลัพธ์ | input: self.tmp | output: out = Path
        out = self.tmp / "split.json"
        # exit code ต้องเป็น 1 | input: ["--root", ..., "--out", ...] | output: assertion
        self.assertEqual(pd.main(["--root", str(self.tmp), "--out", str(out)]), 1)
        # ต้องไม่มีไฟล์ผลลัพธ์ถูกสร้าง | input: out | output: assertion
        self.assertFalse(out.exists())


# รันเทสต์ทั้งหมดเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เริ่มตัวรันเทสต์ | input: - | output: ผลผ่าน/ไม่ผ่าน
    unittest.main()
