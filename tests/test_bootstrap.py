# นำเข้า sys: แก้ sys.path เพื่อให้นำเข้าโมดูลใน src ได้ | input: - | output: โมดูล sys
import sys
# นำเข้า unittest: เฟรมเวิร์กทดสอบมาตรฐาน | input: - | output: โมดูล unittest
import unittest
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า NumPy: สร้างข้อมูลสุ่มสำหรับทดสอบ | input: - | output: โมดูล np
import numpy as np

# เพิ่มโฟลเดอร์ src ไว้หน้าสุดของ sys.path | input: path ของไฟล์นี้ | output: sys.path ถูกแก้ไข
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# นำเข้าโมดูล bootstrap โดยตั้งชื่อย่อ b | input: - | output: โมดูล b
import bootstrap as b  # noqa: E402

# ป้ายกำกับทดสอบ: 20 รูปปกติ (0) ตามด้วย 30 รูปบกพร่อง (1) | input: - | output: LABELS = list ยาว 50
LABELS = [0] * 20 + [1] * 30


# ฟังก์ชันสร้างคะแนนที่ปกติกับบกพร่องซ้อนทับกันบางส่วน | input: seed = เมล็ดสุ่ม | output: อาร์เรย์ float ยาว 50 (20 ค่าแรกเฉลี่ย 0, 30 ค่าหลังเฉลี่ย 1.5)
def overlapping_scores(seed=0):
    # สร้างตัวสุ่มตาม seed | input: seed | output: rng
    rng = np.random.default_rng(seed)
    # สุ่มคะแนนปกติ 20 ค่า (เฉลี่ย 0) ต่อกับคะแนนบกพร่อง 30 ค่า (เฉลี่ย 1.5) | input: rng | output: อาร์เรย์ยาว 50
    return np.concatenate([rng.normal(0, 1, 20), rng.normal(1.5, 1, 30)])


# ชุดทดสอบ bootstrap
class BootstrapTest(unittest.TestCase):
    # เทสต์ว่าแยกได้สมบูรณ์ให้ช่วงเป็นจุดเดียว | input: คะแนนปกติ 0.1 บกพร่อง 0.9 threshold 0.5 | output: low = high = 1.0 ทุกเมตริก
    def test_perfect_separation_gives_a_point_interval(self):
        # สร้างคะแนนแยกกันชัดเจน | input: - | output: scores
        scores = [0.1] * 20 + [0.9] * 30
        # หาช่วงความเชื่อมั่น 200 รอบ | input: LABELS, scores, 0.5 | output: result = dict
        result = b.image_ci(LABELS, scores, 0.5, rounds=200)
        # วนทุกเมตริก
        for name in b.METRICS:
            # ขอบล่างและขอบบนต้องเป็น 1.0 ทั้งคู่ | input: result[name] | output: assertion
            self.assertEqual((result[name]["low"], result[name]["high"]), (1.0, 1.0))

    # เทสต์ว่าช่วงครอบคลุมค่าจริง | input: คะแนนซ้อนทับ threshold 0.75 | output: low <= value <= high ทุกเมตริก และ AUROC มีความกว้าง
    def test_interval_contains_the_point_estimate(self):
        # หาช่วงความเชื่อมั่น 500 รอบ | input: LABELS, overlapping_scores(), 0.75 | output: result
        result = b.image_ci(LABELS, overlapping_scores(), 0.75, rounds=500)
        # วนทุกเมตริก
        for name in b.METRICS:
            # แถวผลของเมตริกนี้ | input: result, name | output: row = dict
            row = result[name]
            # ขอบล่างต้องไม่เกินค่าจริง | input: row | output: assertion
            self.assertLessEqual(row["low"], row["value"])
            # ค่าจริงต้องไม่เกินขอบบน | input: row | output: assertion
            self.assertLessEqual(row["value"], row["high"])
        # ช่วงของ AUROC ต้องมีความกว้าง (low < high) | input: result | output: assertion
        self.assertLess(result["image_auroc"]["low"], result["image_auroc"]["high"])

    # เทสต์ว่า seed เดียวกันให้ผลเหมือนกัน | input: seed 7 สองครั้ง | output: ผลเท่ากัน
    def test_same_seed_gives_same_interval(self):
        # สร้างคะแนนซ้อนทับ | input: - | output: scores
        scores = overlapping_scores()
        # เรียกสองครั้งด้วย seed เดียวกัน ต้องได้ผลเท่ากัน
        self.assertEqual(
            # ครั้งที่ 1 | input: LABELS, scores, 0.75, rounds=100, seed=7 | output: dict
            b.image_ci(LABELS, scores, 0.75, rounds=100, seed=7),
            # ครั้งที่ 2 | input: เหมือนกัน | output: dict
            b.image_ci(LABELS, scores, 0.75, rounds=100, seed=7),
        )

    # เทสต์ช่วงความต่างของ AUROC แบบจับคู่ | input: โมเดลเหมือนกัน และโมเดลสมบูรณ์เทียบกับสัญญาณรบกวน | output: กรณีแรกต่าง 0 ตลอด กรณีหลัง low > 0
    def test_paired_difference(self):
        # สร้างคะแนนซ้อนทับ | input: - | output: scores
        scores = overlapping_scores()
        # เทียบโมเดลกับตัวเอง | input: LABELS, scores, scores, rounds=100 | output: same = dict
        same = b.paired_auroc_diff_ci(LABELS, scores, scores, rounds=100)
        # ผลต่างต้องเป็น 0 ทั้งค่าจริง ขอบล่าง และขอบบน | input: same | output: assertion
        self.assertEqual((same["value"], same["low"], same["high"]), (0.0, 0.0, 0.0))
        # สร้างสัญญาณรบกวน 50 ค่า | input: seed 1 | output: noise
        noise = np.random.default_rng(1).normal(0, 1, 50)
        # เทียบโมเดลสมบูรณ์ (ใช้ LABELS เป็นคะแนน) กับสัญญาณรบกวน | input: LABELS, LABELS, noise, rounds=300 | output: better = dict
        better = b.paired_auroc_diff_ci(LABELS, LABELS, noise, rounds=300)
        # ขอบล่างต้องมากกว่า 0 (โมเดลสมบูรณ์ดีกว่าแน่นอน) | input: better | output: assertion
        self.assertGreater(better["low"], 0)

    # เทสต์ว่ามีคลาสเดียวต้องเกิด error | input: ป้ายกำกับเป็น 1 ทั้งหมด | output: ValueError
    def test_one_class_only_fails(self):
        # ต้องเกิด ValueError
        with self.assertRaises(ValueError):
            # เรียก image_ci ด้วยป้ายกำกับคลาสเดียว | input: [1,1,1], [0.1,0.2,0.3], 0.5 | output: ยกข้อยกเว้น
            b.image_ci([1, 1, 1], [0.1, 0.2, 0.3], 0.5)


# รันเทสต์ทั้งหมดเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เริ่มตัวรันเทสต์ | input: - | output: ผลผ่าน/ไม่ผ่าน
    unittest.main()
