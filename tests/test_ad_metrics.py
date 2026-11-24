# นำเข้า sys: แก้ sys.path เพื่อให้นำเข้าโมดูลใน src ได้ | input: - | output: โมดูล sys
import sys
# นำเข้า unittest: เฟรมเวิร์กทดสอบมาตรฐาน | input: - | output: โมดูล unittest
import unittest
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า NumPy: สร้างข้อมูลทดสอบ | input: - | output: โมดูล np
import numpy as np

# เพิ่มโฟลเดอร์ src ไว้หน้าสุดของ sys.path | input: path ของไฟล์นี้ | output: sys.path ถูกแก้ไข
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# นำเข้าโมดูลเมตริกโดยตั้งชื่อย่อ m | input: - | output: โมดูล m
import ad_metrics as m  # noqa: E402


# ฟังก์ชันสร้างข้อมูลทดสอบ mask และ map | input: perfect = True ให้ map สูงตรงบริเวณบกพร่อง (ทำนายสมบูรณ์), False = map สุ่มล้วน | output: tuple (masks bool (6, 32, 32), maps float32 (6, 32, 32))
def sample_data(perfect=True):
    # สร้างตัวสุ่มที่ seed 0 (ผลซ้ำได้) | input: 0 | output: rng
    rng = np.random.default_rng(0)
    # สร้าง mask ว่าง 6 รูป ขนาด 32x32 | input: - | output: masks = bool ค่า False
    masks = np.zeros((6, 32, 32), dtype=bool)
    # รูปที่ 3: บริเวณบกพร่องขนาด 4x4 | input: - | output: masks แก้แล้ว
    masks[3, 2:6, 2:6] = True
    # รูปที่ 4: บริเวณบกพร่องขนาด 10x10 | input: - | output: masks แก้แล้ว
    masks[4, 10:20, 10:20] = True
    # รูปที่ 5: บริเวณบกพร่องขนาด 3x3 | input: - | output: masks แก้แล้ว
    masks[5, 25:28, 25:28] = True
    # สร้าง map ค่าสุ่ม 0-1 ชนิด float32 | input: rng, masks.shape | output: maps
    maps = rng.random(masks.shape).astype(np.float32)
    # ถ้าต้องการทำนายสมบูรณ์
    if perfect:
        # บวก 2 ให้พิกเซลบกพร่อง ทำให้สูงกว่าพิกเซลอื่นแน่นอน | input: maps, masks | output: maps แก้แล้ว
        maps[masks] += 2
    # คืน masks และ maps | input: masks, maps | output: tuple
    return masks, maps


# ชุดทดสอบเมตริก
class MetricsTest(unittest.TestCase):
    # เทสต์ว่า threshold ใช้ควอนไทล์ | input: ตัวเลข 0-100 ควอนไทล์ 0.99 | output: 99.0
    def test_threshold_uses_quantile(self):
        # ควอนไทล์ที่ 0.99 ของ 0..100 ต้องเป็น 99.0 | input: range(101), 0.99 | output: assertion
        self.assertAlmostEqual(m.pick_threshold(range(101), 0.99), 99.0)

    # เทสต์ว่า map สมบูรณ์ได้ AU-PRO ประมาณ 1 | input: sample_data() | output: AU-PRO ~ 1.0
    def test_perfect_map_scores_one(self):
        # สร้างข้อมูลทำนายสมบูรณ์ | input: - | output: masks, maps
        masks, maps = sample_data()
        # AU-PRO ต้องเป็น 1.0 (คลาดได้ 2 ตำแหน่งทศนิยม) | input: masks, maps | output: assertion
        self.assertAlmostEqual(m.au_pro(masks, maps), 1.0, places=2)

    # เทสต์ว่า map สุ่มได้ AU-PRO ประมาณครึ่งหนึ่งของ fpr_limit | input: sample_data(perfect=False) | output: AU-PRO ~ 0.15
    def test_random_map_scores_about_half_the_fpr_limit(self):
        # สร้างข้อมูลที่ map เป็นสุ่มล้วน | input: perfect=False | output: masks, maps
        masks, maps = sample_data(perfect=False)
        # AU-PRO ต้องใกล้ 0.15 (คลาดได้ 0.06) | input: masks, maps | output: assertion
        self.assertAlmostEqual(m.au_pro(masks, maps), 0.15, delta=0.06)

    # เทสต์ว่าบริเวณเล็กนับเท่ากับบริเวณใหญ่ | input: บริเวณ 2x2 (สุ่ม) กับ 20x20 (ทำนายถูก) | output: AU-PRO ~ 0.575
    def test_small_region_counts_as_much_as_large_region(self):
        # สร้าง mask 1 รูป ขนาด 40x40 | input: - | output: masks
        masks = np.zeros((1, 40, 40), dtype=bool)
        # บริเวณเล็ก 2x2 | input: - | output: masks แก้แล้ว
        masks[0, 1:3, 1:3] = True
        # บริเวณใหญ่ 20x20 | input: - | output: masks แก้แล้ว
        masks[0, 10:30, 10:30] = True
        # map สุ่ม (seed 1) | input: masks.shape | output: maps
        maps = np.random.default_rng(1).random(masks.shape).astype(np.float32)
        # ทำให้บริเวณใหญ่ทำนายถูกสมบูรณ์ (บวก 2) | input: maps | output: maps แก้แล้ว
        maps[0, 10:30, 10:30] += 2
        # บริเวณใหญ่ได้ ~1 และเล็กได้ ~0.15 เฉลี่ยเท่ากัน จึงคาดว่า 0.5 + 0.5*0.15 | input: masks, maps | output: assertion
        self.assertAlmostEqual(m.au_pro(masks, maps), 0.5 + 0.5 * 0.15, delta=0.08)

    # เทสต์ว่ารูปร่างไม่ตรงกันหรือไม่มีพิกเซลบกพร่องต้องเกิด error | input: อาร์เรย์ผิดรูปร่าง และ mask ศูนย์ | output: ValueError ทั้งสองกรณี
    def test_shape_mismatch_and_empty_masks_fail(self):
        # กรณีรูปร่าง masks กับ maps ต่างกัน ต้องเกิด ValueError
        with self.assertRaises(ValueError):
            # เรียก au_pro ด้วย (1,2,2) กับ (1,2,3) | input: อาร์เรย์ศูนย์ | output: ยกข้อยกเว้น
            m.au_pro(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)))
        # กรณี mask ไม่มีพิกเซลบกพร่องเลย ต้องเกิด ValueError
        with self.assertRaises(ValueError):
            # เรียก au_pro ด้วย mask ศูนย์ | input: อาร์เรย์ศูนย์ | output: ยกข้อยกเว้น
            m.au_pro(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)))

    # เทสต์ว่า evaluate นับ TP/FP/FN/TN และคำนวณเมตริกถูกต้อง | input: labels, scores, threshold 0.5 | output: TP=2, FP=1, FN=1, TN=2, precision=recall=2/3
    def test_evaluate_counts_and_scores(self):
        # สร้างข้อมูลทำนายสมบูรณ์ | input: - | output: masks, maps
        masks, maps = sample_data()
        # ป้ายกำกับ 3 ปกติ 3 บกพร่อง | input: - | output: labels
        labels = [0, 0, 0, 1, 1, 1]
        # คำนวณเมตริกด้วยคะแนนและ threshold 0.5 | input: labels, scores, masks, maps, 0.5 | output: result = dict
        result = m.evaluate(labels, [0.1, 0.2, 0.9, 0.8, 0.7, 0.3], masks, maps, 0.5)
        # ตรวจ TP กับ FP
        self.assertEqual(
            # ต้องได้ TP=2 และ FP=1 | input: result | output: assertion
            (result["true_positive"], result["false_positive"]), (2, 1)
        )
        # ตรวจ FN กับ TN
        self.assertEqual(
            # ต้องได้ FN=1 และ TN=2 | input: result | output: assertion
            (result["false_negative"], result["true_negative"]), (1, 2)
        )
        # precision = 2/3 | input: result | output: assertion
        self.assertAlmostEqual(result["precision"], 2 / 3)
        # recall = 2/3 | input: result | output: assertion
        self.assertAlmostEqual(result["recall"], 2 / 3)
        # pixel AUROC = 1.0 เพราะ map ทำนายสมบูรณ์ | input: result | output: assertion
        self.assertAlmostEqual(result["pixel_auroc"], 1.0)

    # เทสต์ว่าผลแยกตามความบกพร่องเทียบกับรูปปกติเท่านั้น | input: กลุ่ม good/crack/scratch | output: มีแถว crack กับ scratch พร้อมค่าที่คาด
    def test_evaluate_by_defect_compares_with_good_only(self):
        # สร้างข้อมูลทำนายสมบูรณ์ | input: - | output: masks, maps
        masks, maps = sample_data()
        # ชื่อกลุ่มของแต่ละรูป | input: - | output: groups
        groups = ["good", "good", "good", "crack", "crack", "scratch"]
        # คำนวณผลแยกตามความบกพร่อง | input: groups, scores, masks, maps, 0.5 | output: rows = dict
        rows = m.evaluate_by_defect(
            groups, [0.1, 0.2, 0.3, 0.9, 0.2, 0.8], masks, maps, 0.5
        )
        # ต้องมีเฉพาะ crack กับ scratch (ไม่มี good) | input: rows | output: assertion
        self.assertEqual(set(rows), {"crack", "scratch"})
        # crack มี 2 รูป | input: rows | output: assertion
        self.assertEqual(rows["crack"]["images"], 2)
        # recall ของ crack = 0.5 (จับได้ 1 จาก 2) | input: rows | output: assertion
        self.assertAlmostEqual(rows["crack"]["recall"], 0.5)
        # image AUROC ของ scratch = 1.0 | input: rows | output: assertion
        self.assertAlmostEqual(rows["scratch"]["image_auroc"], 1.0)


# รันเทสต์ทั้งหมดเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เริ่มตัวรันเทสต์ | input: - | output: ผลผ่าน/ไม่ผ่าน
    unittest.main()
