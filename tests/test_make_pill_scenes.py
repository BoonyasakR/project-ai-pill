# นำเข้า sys: แก้ sys.path | input: - | output: โมดูล sys
import sys
# นำเข้า unittest: เฟรมเวิร์กทดสอบมาตรฐาน | input: - | output: โมดูล unittest
import unittest
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า OpenCV: วาดวงรีเป็นเม็ดยาจำลอง | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: สร้างและรวมอาร์เรย์ | input: - | output: โมดูล np
import numpy as np

# โฟลเดอร์รากของโปรเจกต์ | input: path ของไฟล์นี้ | output: ROOT = Path
ROOT = Path(__file__).resolve().parents[1]
# เพิ่มโฟลเดอร์ src และ backend ไว้หน้าสุดของ sys.path | input: ROOT | output: sys.path ถูกแก้ไข
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]

# นำเข้าฟังก์ชันประกอบฉากและตัดเม็ดยา | input: - | output: compose_scene, cut_pill
from make_pill_scenes import compose_scene, cut_pill  # noqa: E402
# นำเข้าฟังก์ชันหาเม็ดยาในรูป | input: - | output: find_pills
from segmentation import find_pills  # noqa: E402


# ชุดทดสอบการประกอบฉาก
class SceneTest(unittest.TestCase):
    # ตัวช่วยสร้างฉากที่มีเม็ดยา 8 เม็ด | input: touching = ให้เม็ดแตะกันหรือไม่ | output: tuple (รูปฉาก, list ของ mask เม็ด)
    def scene(self, touching):
        # สร้างตัวสุ่ม seed 0 (ผลซ้ำได้) | input: 0 | output: rng
        rng = np.random.default_rng(0)
        # สร้างรูปดำ 800x800 | input: - | output: image
        image = np.zeros((800, 800, 3), np.uint8)
        # วาดเม็ดยารูปวงรีสีขาวอมครีม | input: image | output: image มีวงรี
        cv2.ellipse(image, (400, 400), (345, 238), 0, 0, 360, (230, 225, 220), -1)
        # mask = พิกเซลที่ช่องแดงมากกว่า 0 (คือตัววงรี) | input: image | output: mask = bool (800, 800)
        mask = image[..., 0] > 0
        # ตัดเม็ด 8 ชิ้น (หมุน/ปรับขนาดสุ่มต่างกัน) แล้วประกอบเป็นฉาก | input: image, mask, rng, touching | output: (ฉาก, masks)
        return compose_scene([cut_pill(image, mask, rng) for _ in range(8)], rng, touching)

    # เทสต์ว่าเม็ดไม่ทับกันและหาเจอครบ | input: ฉากแบบแยกกันและแบบแตะกัน | output: ไม่มีพิกเซลที่ถูกใช้ซ้ำ และ find_pills พบ 8 เม็ด
    def test_pills_do_not_overlap_and_are_all_found(self):
        # ทดสอบทั้งสองแบบ
        for touching in (False, True):
            # สร้างฉาก | input: touching | output: scene, masks
            scene, masks = self.scene(touching)
            # ผลรวม mask ทุกเม็ดในแต่ละพิกเซลต้องไม่เกิน 1 (ไม่ทับกัน) | input: masks | output: assertion
            self.assertEqual(np.sum(masks, axis=0).max(), 1)
            # ระบบหาเม็ดต้องพบ 8 เม็ด | input: scene | output: assertion
            self.assertEqual(len(find_pills(scene)), 8)


# รันเทสต์ทั้งหมดเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เริ่มตัวรันเทสต์ | input: - | output: ผลผ่าน/ไม่ผ่าน
    unittest.main()
