# นำเข้า io: สร้างไฟล์จำลองในหน่วยความจำ | input: - | output: โมดูล io
import io
# นำเข้า json: ใช้ตรวจว่าผลลัพธ์แปลงเป็น JSON ได้ | input: - | output: โมดูล json
import json
# นำเข้า unittest: เฟรมเวิร์กทดสอบมาตรฐาน | input: - | output: โมดูล unittest
import unittest
# นำเข้า SimpleNamespace: สร้างวัตถุจำลองที่มี attribute ตามต้องการ | input: - | output: คลาส SimpleNamespace
from types import SimpleNamespace

# นำเข้า OpenCV: วาดรูปทรงสำหรับสร้างรูปทดสอบ | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: สร้างและเปรียบเทียบอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า PyTorch: สร้าง tensor จำลองเป็นผลของโมเดล | input: - | output: โมดูล torch
import torch
# นำเข้า TestClient: ไคลเอนต์จำลองสำหรับเรียก API ของ FastAPI โดยไม่ต้องเปิดเซิร์ฟเวอร์จริง | input: - | output: คลาส TestClient
from fastapi.testclient import TestClient
# นำเข้า Image ของ Pillow: สร้าง/แปลงรูป | input: - | output: คลาส Image
from PIL import Image

# นำเข้าโมดูล app โดยตั้งชื่อเรียกว่า service | input: - | output: โมดูล service
import app as service
# นำเข้าฟังก์ชันสร้างตารางสีและวัดสีแปลก | input: - | output: build_seen, novel_colour
from colour_check import build_seen, novel_colour
# นำเข้าท่าวางของ pillQC, คลาส Inspector และฟังก์ชันซ้อน heatmap | input: - | output: PILLQC_POSE, Inspector, overlay_heatmap
from inference import PILLQC_POSE, Inspector, overlay_heatmap
# นำเข้าฟังก์ชันตัดแบ่งเม็ดยาที่จะทดสอบ | input: - | output: ฟังก์ชันตามชื่อ
from segmentation import align_pill, find_pills, in_training_pose, pink_pill_mask, round_pill_region


# คลาสตัวตรวจปลอม ใช้แทนตัวจริงเพื่อให้ทดสอบ API ได้โดยไม่ต้องโหลดโมเดล
class FakeInspector:
    # เมธอดตรวจปลอม | input: image = PIL.Image | output: dict ผลตายตัว พร้อมขนาดรูป [กว้าง, สูง]
    def inspect(self, image):
        # คืนผลตายตัวและขนาดรูปที่ได้รับ เพื่อให้เทสต์ตรวจได้ว่ารูปไปถึงตัวตรวจ | input: image | output: dict
        return {"score": 1.0, "threshold": 2.0, "is_anomalous": False, "size": list(image.size)}


# ฟังก์ชันสร้างไบต์ของรูป PNG ขนาด 8x6 สีขาว | input: - | output: bytes ของไฟล์ PNG
def png_bytes():
    # สร้างไฟล์จำลองในหน่วยความจำ | input: - | output: buffer = BytesIO
    buffer = io.BytesIO()
    # สร้างรูปขาว 8x6 แล้วบันทึกลง buffer เป็น PNG | input: ขนาดและสี | output: buffer ที่มีไบต์ PNG
    Image.new("RGB", (8, 6), "white").save(buffer, format="PNG")
    # คืนไบต์ทั้งหมดใน buffer | input: buffer | output: bytes
    return buffer.getvalue()


# ชุดทดสอบ API
class ApiTest(unittest.TestCase):
    # เตรียมก่อนทุกเทสต์ | input: - | output: client พร้อมตัวตรวจปลอม
    def setUp(self):
        # แทนที่ dependency get_inspector ด้วยตัวปลอม | input: FakeInspector | output: app ใช้ตัวปลอม
        service.app.dependency_overrides[service.get_inspector] = FakeInspector
        # ลงทะเบียนให้ล้างการแทนที่หลังเทสต์จบ | input: - | output: cleanup ถูกจัดไว้
        self.addCleanup(service.app.dependency_overrides.clear)
        # สร้างไคลเอนต์ทดสอบ | input: service.app | output: self.client
        self.client = TestClient(service.app)

    # ตัวช่วยอัปโหลดไฟล์ไปที่ /api/inspect | input: content = bytes | output: response ของเซิร์ฟเวอร์
    def upload(self, content):
        # ส่ง POST แนบไฟล์ชื่อ pill.png | input: content | output: response (status_code, json)
        return self.client.post("/api/inspect", files={"file": ("pill.png", content, "image/png")})

    # เทสต์ endpoint สุขภาพ | input: - | output: ผ่านเมื่อ status เป็น "ok"
    def test_health(self):
        # เรียก GET /api/health แล้วยืนยันว่า status == "ok" | input: - | output: assertion
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")

    # เทสต์ส่งรูปที่ถูกต้อง | input: PNG 8x6 | output: status 200 และ size = [8, 6]
    def test_inspect_valid_image(self):
        # อัปโหลดรูปจริง | input: png_bytes() | output: response
        response = self.upload(png_bytes())
        # ยืนยันว่าตอบ 200 | input: response | output: assertion
        self.assertEqual(response.status_code, 200)
        # ยืนยันว่าตัวตรวจได้รับรูปขนาด 8x6 | input: response.json() | output: assertion
        self.assertEqual(response.json()["size"], [8, 6])

    # เทสต์ส่งไฟล์ที่ไม่ใช่รูป | input: ข้อความธรรมดา | output: status 400
    def test_rejects_non_image(self):
        # อัปโหลดไบต์ที่ไม่ใช่รูป | input: b"not an image" | output: response
        response = self.upload(b"not an image")
        # ยืนยันว่าถูกปฏิเสธด้วย 400 | input: response | output: assertion
        self.assertEqual(response.status_code, 400)

    # เทสต์ส่งไฟล์ใหญ่เกิน 10 MB | input: ไบต์ยาว MAX+1 | output: status 413
    def test_rejects_large_file(self):
        # อัปโหลดไฟล์ที่ใหญ่เกินขีดจำกัด 1 ไบต์ | input: b"0" * (MAX+1) | output: response
        response = self.upload(b"0" * (service.MAX_UPLOAD_BYTES + 1))
        # ยืนยันว่าถูกปฏิเสธด้วย 413 | input: response | output: assertion
        self.assertEqual(response.status_code, 413)

    # เทสต์ส่งชนิดยาที่ไม่รู้จัก | input: pill_type="capsule" | output: status 400
    def test_rejects_unknown_pill_type(self):
        # เลิกใช้ตัวปลอม เพื่อให้ get_inspector ตัวจริงทำงาน | input: - | output: dependency_overrides ว่าง
        service.app.dependency_overrides.clear()
        # ส่งรูปพร้อมชนิดยาที่ไม่มีในระบบ | input: png_bytes(), pill_type="capsule" | output: response
        response = self.client.post(
            "/api/inspect", files={"file": ("pill.png", png_bytes(), "image/png")}, data={"pill_type": "capsule"}
        )
        # ยืนยันว่าถูกปฏิเสธด้วย 400 | input: response | output: assertion
        self.assertEqual(response.status_code, 400)

    # เทสต์ว่าชนิดยาแต่ละชนิดเลือกโมเดลของตัวเอง | input: ไม่ระบุชนิด และ "pillqc" | output: โหลดชนิด "mvtec" แล้ว "pillqc" ตามลำดับ
    def test_pill_type_picks_its_own_model(self):
        # เลิกใช้ตัวปลอม | input: - | output: dependency_overrides ว่าง
        service.app.dependency_overrides.clear()
        # list สำหรับบันทึกชนิดที่ถูกขอโหลด | input: - | output: picked = []
        picked = []
        # เก็บฟังก์ชันโหลดจริงไว้คืนภายหลัง | input: service.load_inspector | output: original
        original = service.load_inspector
        # แทนที่ด้วยฟังก์ชันที่จดชื่อชนิดแล้วคืนตัวปลอม (append คืน None จึงไปใช้ FakeInspector()) | input: name | output: FakeInspector
        service.load_inspector = lambda name: picked.append(name) or FakeInspector()
        # ลงทะเบียนให้คืนฟังก์ชันจริงหลังเทสต์จบ | input: original | output: cleanup ถูกจัดไว้
        self.addCleanup(setattr, service, "load_inspector", original)
        # วนส่งสองแบบ: ไม่ระบุชนิด (ค่าเริ่มต้น) และระบุ pillqc
        for data in ({}, {"pill_type": "pillqc"}):
            # ส่งรูปพร้อมข้อมูลฟอร์ม | input: png_bytes(), data | output: response
            response = self.client.post("/api/inspect", files={"file": ("pill.png", png_bytes(), "image/png")}, data=data)
            # ยืนยันว่าตอบ 200 | input: response | output: assertion
            self.assertEqual(response.status_code, 200)
        # ยืนยันว่าชนิดที่ถูกโหลดคือ mvtec แล้ว pillqc | input: picked | output: assertion
        self.assertEqual(picked, ["mvtec", "pillqc"])

    # เทสต์ว่า /api/results ส่งเฉพาะ field สาธารณะ | input: - | output: ไม่มี test_scores/validation_scores
    def test_results_keep_only_public_fields(self):
        # เรียก GET /api/results | input: - | output: result = dict
        result = self.client.get("/api/results").json()
        # ยืนยันว่ามี key "main" | input: result | output: assertion
        self.assertIn("main", result)
        # วนตรวจทุกการทดลองใน main
        for run in result["main"].values():
            # ต้องไม่มีคะแนนชุดทดสอบ (ข้อมูลภายใน) | input: run | output: assertion
            self.assertNotIn("test_scores", run)
            # ต้องไม่มีคะแนนชุดตรวจสอบ | input: run | output: assertion
            self.assertNotIn("validation_scores", run)
        # ยืนยันว่าแปลงเป็น JSON ได้ (ถ้าไม่ได้จะเกิด error) | input: result | output: str JSON (ไม่ใช้)
        json.dumps(result)


# ชุดทดสอบ heatmap
class HeatmapTest(unittest.TestCase):
    # เทสต์ว่าสเกลสีถูกกำหนดโดย threshold | input: map ศูนย์ เทียบ map ค่า 100 ที่ threshold 10 | output: สีต่างกัน และกรณีเย็นเป็นสีน้ำเงิน
    def test_scale_is_fixed_by_threshold(self):
        # สร้างรูปดำ 4x4 | input: - | output: image
        image = Image.new("RGB", (4, 4), "black")
        # ซ้อน heatmap กรณีคะแนน 0 (เย็น) แล้วแปลงเป็นอาร์เรย์ | input: image, zeros, 10.0 | output: cool = uint8 (4, 4, 3)
        cool = np.asarray(overlay_heatmap(image, np.zeros((4, 4)), 10.0))
        # ซ้อน heatmap กรณีคะแนน 100 (ร้อน) | input: image, 100s, 10.0 | output: hot = uint8 (4, 4, 3)
        hot = np.asarray(overlay_heatmap(image, np.full((4, 4), 100.0), 10.0))
        # กรณีเย็น ช่องน้ำเงิน (index 2) ต้องมากกว่าช่องแดง (index 0) | input: cool | output: assertion
        self.assertGreater(cool[0, 0, 2], cool[0, 0, 0])
        # รูปเย็นกับรูปร้อนต้องไม่เหมือนกัน | input: cool, hot | output: assertion
        self.assertFalse(np.array_equal(cool, hot))


# ฟังก์ชันตัวช่วยวาดวงรีสีขาวอมครีมลงบนผืนผ้าใบ | input: canvas = อาร์เรย์ RGB, centre = (x, y), axes = (ครึ่งแกนยาว, ครึ่งแกนสั้น), angle = องศา | output: ไม่คืนค่า (แก้ canvas ในที่เดิม)
def ellipse(canvas, centre, axes, angle=0):
    # วาดวงรีทึบสี (230, 225, 220) ครบ 0-360 องศา | input: canvas, centre, axes, angle | output: canvas ที่มีวงรี
    cv2.ellipse(canvas, centre, axes, angle, 0, 360, (230, 225, 220), -1)


# ชุดทดสอบการตัดแบ่งเม็ดยา
class SegmentationTest(unittest.TestCase):
    # เทสต์นับเม็ดทั้งที่แยกกันและที่ติดกัน | input: ฉากวงรี 4 อัน | output: find_pills ได้ 4 เม็ด
    def test_counts_separate_and_touching_pills(self):
        # สร้างฉากดำขนาด 400x900 | input: - | output: scene
        scene = np.zeros((400, 900, 3), np.uint8)
        # วาดเม็ดที่ 1 | input: scene | output: scene มีวงรี
        ellipse(scene, (120, 200), (90, 60))
        # วาดเม็ดที่ 2 เอียง 40 องศา | input: scene | output: scene มีวงรี
        ellipse(scene, (400, 200), (90, 60), 40)
        # วาดเม็ดที่ 3 | input: scene | output: scene มีวงรี
        ellipse(scene, (640, 200), (90, 60))
        # วาดเม็ดที่ 4 โดยให้ติดกับเม็ดที่ 3 ทางซ้าย
        ellipse(scene, (818, 200), (90, 60))  # Touches the pill on its left.
        # ยืนยันว่านับได้ 4 เม็ด | input: scene | output: assertion
        self.assertEqual(len(find_pills(scene)), 4)

    # เทสต์ว่าเม็ดเดียวต้องไม่ถูกแยกเป็นหลายเม็ด | input: วงรีใหญ่ 1 อัน | output: 1 เม็ด
    def test_one_pill_stays_one_pill(self):
        # สร้างฉากดำ 800x800 | input: - | output: scene
        scene = np.zeros((800, 800, 3), np.uint8)
        # วาดวงรีใหญ่กลางภาพ | input: scene | output: scene มีวงรี
        ellipse(scene, (400, 400), (345, 238))
        # ยืนยันว่านับได้ 1 เม็ด | input: scene | output: assertion
        self.assertEqual(len(find_pills(scene)), 1)

    # เทสต์ภาพดำล้วนต้องไม่พบเม็ดยา | input: ภาพดำ 50x50 | output: list ว่าง
    def test_blank_image_has_no_pills(self):
        # ยืนยันว่า find_pills คืน [] | input: ภาพดำ | output: assertion
        self.assertEqual(find_pills(np.zeros((50, 50, 3), np.uint8)), [])

    # เทสต์ว่าการจัดท่าทำให้แกนยาวเป็นแนวนอนและขนาดเท่ารูปฝึก | input: วงรีเอียง 35 องศา | output: ความยาว ~693, แนวนอน, อยู่กลาง
    def test_align_makes_long_axis_horizontal_at_training_size(self):
        # สร้างฉากดำ 500x500 | input: - | output: scene
        scene = np.zeros((500, 500, 3), np.uint8)
        # วาดวงรีเอียง 35 องศา | input: scene | output: scene มีวงรี
        ellipse(scene, (250, 250), (100, 70), 35)
        # จัดท่าเม็ดแรกที่พบ แล้วรับเฉพาะ mask (ตัวแรกคือรูปไม่ใช้) | input: scene, mask | output: pill = bool (800, 800)
        _, pill = align_pill(scene, find_pills(scene)[0])
        # หาพิกัดของพิกเซลในเม็ด | input: pill | output: ys, xs
        ys, xs = np.nonzero(pill)
        # ความกว้างของเม็ดต้องใกล้ 693 (คลาดได้ 8) | input: xs | output: assertion
        self.assertAlmostEqual(np.ptp(xs), 693, delta=8)
        # ความสูงต้องน้อยกว่าความกว้าง (แกนยาวเป็นแนวนอน) | input: ys, xs | output: assertion
        self.assertLess(np.ptp(ys), np.ptp(xs))
        # จุดกึ่งกลางแนวนอนต้องใกล้ 400 (คลาดได้ 3) | input: xs | output: assertion
        self.assertAlmostEqual(xs.mean(), 400, delta=3)

    # เทสต์ว่า in_training_pose ยอมให้เปลี่ยนแปลงเล็กน้อยเท่านั้น | input: วงรีหลายแบบ | output: True/False ตามที่คาด
    def test_training_pose_allows_small_changes_only(self):
        # ฟังก์ชันซ้อนในสร้าง mask ของวงรีเม็ดเดียว | input: centre, axes, angle, size | output: mask bool ของเม็ดแรกที่พบ
        def mask(centre=(400, 400), axes=(346, 238), angle=0, size=800):
            # สร้างฉากดำสี่เหลี่ยมจัตุรัส | input: size | output: scene
            scene = np.zeros((size, size, 3), np.uint8)
            # วาดวงรีตามพารามิเตอร์ | input: scene, centre, axes, angle | output: scene มีวงรี
            ellipse(scene, centre, axes, angle)
            # คืน mask ของเม็ดแรก | input: scene | output: bool (size, size)
            return find_pills(scene)[0]

        # ท่าตรงรูปฝึก ต้องผ่าน | input: mask() | output: assertion
        self.assertTrue(in_training_pose(mask()))
        # หมุน 181 องศา (เกือบกลับหัว) ยังผ่าน เพราะแกนไม่มีทิศ | input: mask(angle=181) | output: assertion
        self.assertTrue(in_training_pose(mask(angle=181)))
        # ภาพขนาด 400 ที่ย่อสัดส่วนเท่ากัน ต้องผ่าน | input: mask(...) | output: assertion
        self.assertTrue(in_training_pose(mask(centre=(200, 200), axes=(173, 119), size=400)))
        # เอียง 30 องศา ต้องไม่ผ่าน | input: mask(angle=30) | output: assertion
        self.assertFalse(in_training_pose(mask(angle=30)))
        # เอียง 90 องศา ต้องไม่ผ่าน | input: mask(angle=90) | output: assertion
        self.assertFalse(in_training_pose(mask(angle=90)))
        # ขนาดเล็กลงมาก ต้องไม่ผ่าน | input: mask(axes=(242, 167)) | output: assertion
        self.assertFalse(in_training_pose(mask(axes=(242, 167))))
        # เลื่อนจากกลาง 20 พิกเซล ต้องไม่ผ่าน | input: mask(centre=(420, 400)) | output: assertion
        self.assertFalse(in_training_pose(mask(centre=(420, 400))))


# ฟังก์ชันสร้างรูปเม็ดยาขาวมีจุดแดงหม่นแบบรูปฝึก และอาจมีรอยเปื้อนสีอื่น | input: stain = สี RGB ของรอยเปื้อน (หรือ None) | output: อาร์เรย์ uint8 (800, 800, 3)
def specked_pill(stain=None):
    """A white training-pose pill with dull red specks, and optionally a stain of another colour."""
    # สร้างฉากดำ 800x800 | input: - | output: scene
    scene = np.zeros((800, 800, 3), np.uint8)
    # วาดตัวเม็ดยาวงรีกลางภาพ | input: scene | output: scene มีวงรี
    ellipse(scene, (400, 400), (346, 238))
    # วนวางจุดแดงหม่นทุก 60 พิกเซลจาก x=150 ถึง 650
    for x in range(150, 650, 60):
        # วาดจุดวงกลมรัศมี 6 สีแดงหม่น | input: scene, x | output: scene มีจุด
        cv2.circle(scene, (x, 380), 6, (130, 60, 70), -1)
    # ถ้าระบุสีรอยเปื้อน
    if stain:
        # วาดรอยเปื้อนวงกลมรัศมี 12 ตามสีที่กำหนด | input: scene, stain | output: scene มีรอยเปื้อน
        cv2.circle(scene, (400, 500), 12, stain, -1)
    # คืนรูปที่สร้างเสร็จ | input: scene | output: อาร์เรย์ (800, 800, 3)
    return scene


# ชุดทดสอบการตรวจสีแปลก
class ColourTest(unittest.TestCase):
    # เตรียมตารางสีที่เคยเห็นจากเม็ดปกติหนึ่งเม็ด | input: - | output: self.seen
    def setUp(self):
        # สร้างเม็ดปกติ | input: - | output: normal
        normal = specked_pill()
        # สร้างตารางสีจากเม็ดปกติและ mask ของมัน | input: [(normal, mask)] | output: self.seen = bool (64, 64, 64)
        self.seen = build_seen([(normal, find_pills(normal)[0])])

    # ตัวช่วยวัดพื้นที่สีแปลกในฉาก | input: scene = รูป RGB | output: int พื้นที่กลุ่มสีแปลกใหญ่สุด
    def area(self, scene):
        # หาเม็ดแรก แล้วเอาค่าที่ 0 (พื้นที่) จากผลของ novel_colour | input: self.seen, scene, mask | output: int
        return novel_colour(self.seen, scene, find_pills(scene)[0])[0]

    # เทสต์ว่าสีปกติไม่ถูกแจ้ง | input: เม็ดปกติ และเม็ดที่มีจุดแดงหม่นใหญ่ขึ้น | output: พื้นที่ = 0 ทั้งคู่
    def test_normal_colours_are_seen(self):
        # เม็ดปกติต้องไม่มีสีแปลก | input: specked_pill() | output: assertion
        self.assertEqual(self.area(specked_pill()), 0)
        # จุดแดงหม่นขนาดใหญ่กว่าปกติ แต่สีเดิม ต้องไม่ถูกแจ้ง
        self.assertEqual(self.area(specked_pill(stain=(130, 60, 70))), 0)  # A larger normal speck.

    # เทสต์ว่าสีใหม่ถูกตรวจพบ | input: รอยเปื้อนน้ำเงิน เหลือง แดงสด | output: พื้นที่ > 300 ทุกสี
    def test_new_colours_are_found(self):
        # วนทดสอบทีละสี (น้ำเงิน เหลือง แดงสด)
        for stain in [(40, 40, 200), (230, 220, 60), (240, 40, 60)]:  # Blue, yellow, bright red.
            # พื้นที่สีแปลกต้องมากกว่า 300 พิกเซล (ข้อความ error จะแสดงสีที่พลาด) | input: specked_pill(stain) | output: assertion
            self.assertGreater(self.area(specked_pill(stain)), 300, stain)


# ชุดทดสอบเม็ดยากลม
class RoundPillTest(unittest.TestCase):
    # เทสต์ว่าสิ่งสกปรกบนพื้นที่ติดเม็ดถูกตัดออกจากพื้นที่ให้คะแนน | input: crop ที่มีวงกลมชมพูกับแท่งขาวติดขอบ | output: จุดบนแท่งขาวอยู่ใน mask ชมพูแต่ไม่อยู่ใน region
    def test_floor_dirt_touching_the_pill_is_left_out(self):
        # สร้างรูป crop 225x225 สีพื้นเขียวน้ำเงิน | input: - | output: crop
        crop = np.full((225, 225, 3), (70, 120, 120), np.uint8)
        # วาดเม็ดยากลมสีชมพูกลางรูป รัศมี 74 | input: crop | output: crop มีวงกลม
        cv2.circle(crop, (112, 112), 74, (240, 190, 180), -1)
        # วาดแท่งสิ่งสกปรกสีขาวอมชมพูที่ติดขอบเม็ดลงไปที่พื้น
        cv2.rectangle(crop, (100, 180), (125, 222), (235, 200, 195), -1)  # White dirt joined to the pill.
        # หา mask เม็ดชมพู (รวมสิ่งสกปรกที่ติดกัน) | input: crop | output: mask = bool (225, 225)
        mask = pink_pill_mask(crop)
        # จุดบนสิ่งสกปรกต้องอยู่ใน mask (เพราะติดกับเม็ด) | input: mask | output: assertion
        self.assertTrue(mask[215, 112])
        # สร้างพื้นที่วงกลมครอบเม็ด | input: mask | output: region = bool (225, 225)
        region = round_pill_region(mask)
        # จุดบนสิ่งสกปรกต้องไม่อยู่ใน region | input: region | output: assertion
        self.assertFalse(region[215, 112])
        # กึ่งกลางเม็ดและขอบซ้าย/ขวาต้องอยู่ใน region
        self.assertTrue(region[112, 112] and region[112, 40] and region[112, 184])  # Centre and rim.


# ชุดทดสอบการเลือกเส้นทางตรวจ
class RoutingTest(unittest.TestCase):
    """A single pill must not be judged by a whole-image model it is not posed for."""

    # ตารางสีที่ใช้ในเทสต์ (เริ่มต้นไม่มี = ไม่ตรวจสี) | input: - | output: colour_seen = None
    colour_seen = None

    # ตัวช่วยสร้าง Inspector จำลอง แล้วตรวจรูป | input: scene = RGB, scores = list คะแนนที่โมเดลปลอมจะคืนตามลำดับ, scene_threshold = เกณฑ์ฉาก | output: dict ผลตรวจ
    def inspect(self, scene, scores, scene_threshold=10.0):
        # ทำให้ list คะแนนเป็นตัวไล่ค่าทีละตัว | input: scores | output: queue
        queue = iter(scores)

        # โมเดลปลอม: คืนคะแนนถัดไปในคิว และ anomaly map ศูนย์ | input: batch | output: SimpleNamespace(pred_score, anomaly_map)
        def model(batch):
            # หยิบคะแนนถัดไปมาห่อเป็นผลของโมเดล | input: queue | output: SimpleNamespace
            return SimpleNamespace(pred_score=torch.tensor([next(queue)]), anomaly_map=torch.zeros(1, 1, 8, 8))

        # สร้าง Inspector เปล่า ๆ โดยไม่เรียก __init__ (ไม่ต้องโหลดไฟล์โมเดล) | input: - | output: inspector
        inspector = Inspector.__new__(Inspector)
        # ใส่โมเดลปลอมและ transform ที่ไม่ทำอะไร | input: model | output: inspector.model, inspector.transform
        inspector.model, inspector.transform = model, lambda tensor: tensor
        # ตั้งขนาดรูป 8, threshold ทั้งรูป 10 และ threshold ฉาก | input: scene_threshold | output: attribute ของ inspector
        inspector.image_size, inspector.threshold, inspector.scene_threshold = 8, 10.0, scene_threshold
        # ใส่ตารางสีของเทสต์นี้ | input: self.colour_seen | output: inspector.colour_seen
        inspector.colour_seen = self.colour_seen
        # ตั้ง threshold สีทั้งสองแบบเป็น 20 | input: - | output: inspector.colour_threshold, inspector.scene_colour_threshold
        inspector.colour_threshold = inspector.scene_colour_threshold = 20
        # แทน inspect_scene ด้วยตัวปลอมที่ระบุว่าไปทางเส้นทาง "aligned" | input: image, masks | output: dict {route, pills}
        inspector.inspect_scene = lambda image, masks: {"route": "aligned", "pills": [{} for _ in masks]}
        # ตรวจรูปโดยแปลงอาร์เรย์เป็นรูป PIL | input: scene | output: dict ผลตรวจ
        return inspector.inspect(Image.fromarray(scene))

    # ตัวช่วยสร้างฉากเม็ดยาเม็ดเดียวในท่าฝึก (หรือเอียง) | input: angle = องศา | output: อาร์เรย์ (800, 800, 3)
    def scene(self, angle=0):
        # สร้างฉากดำ 800x800 | input: - | output: scene
        scene = np.zeros((800, 800, 3), np.uint8)
        # วาดวงรีขนาดเท่าเม็ดฝึกตามมุมที่กำหนด | input: scene, angle | output: scene มีวงรี
        ellipse(scene, (400, 400), (346, 238), angle)
        # คืนฉาก | input: scene | output: อาร์เรย์
        return scene

    # เทสต์ว่าเม็ดที่เอียงต้องถูกส่งไปจัดท่า | input: เม็ดเอียง 45 องศา ไม่มีคะแนน | output: route = "aligned" และมี 1 เม็ด
    def test_turned_pill_is_aligned(self):
        # ตรวจเม็ดเอียง (โมเดลปลอมไม่ควรถูกเรียก) | input: scene(45), [] | output: result
        result = self.inspect(self.scene(45), [])
        # ยืนยันว่าไปเส้นทาง aligned กับ 1 เม็ด | input: result | output: assertion
        self.assertEqual((result["route"], result["pills"]), ("aligned", [{}]))

    # เทสต์เม็ดปกติในท่าฝึกใช้โมเดลครั้งเดียว | input: คะแนน 5.0 | output: score = 5.0 และไม่ผิดปกติ
    def test_posed_normal_pill_keeps_one_model_run(self):
        # ตรวจเม็ดในท่าฝึก | input: scene(), [5.0] | output: result
        result = self.inspect(self.scene(), [5.0])
        # คะแนนต้องเป็น 5.0 | input: result | output: assertion
        self.assertEqual(result["score"], 5.0)
        # ต้องไม่ถูกแจ้งผิดปกติ | input: result | output: assertion
        self.assertFalse(result["is_anomalous"])

    # เทสต์เม็ดบกพร่องจริงต้องคงคะแนนทั้งรูป | input: คะแนน 20.0 แล้วหลังพลิก 15.0 | output: score = 20.0 และผิดปกติ
    def test_posed_defect_keeps_whole_image_score(self):
        # ตรวจเม็ดที่คะแนนสูง พลิกแล้วลดลงแต่ยังเกินเกณฑ์ | input: scene(), [20.0, 15.0] | output: result
        result = self.inspect(self.scene(), [20.0, 15.0])  # Lower once turned, but still abnormal.
        # คะแนนต้องคงเป็น 20.0 | input: result | output: assertion
        self.assertEqual(result["score"], 20.0)
        # ต้องถูกแจ้งผิดปกติ | input: result | output: assertion
        self.assertTrue(result["is_anomalous"])

    # เทสต์เม็ดกลับหัวต้องถูกส่งไปจัดท่า | input: คะแนน 20.0 แล้วหลังพลิก 5.0 | output: route = "aligned"
    def test_upside_down_pill_is_aligned(self):
        # ตรวจเม็ดที่พลิกแล้วคะแนนต่ำกว่าเกณฑ์ ต้องไปเส้นทาง aligned | input: scene(), [20.0, 5.0] | output: assertion
        self.assertEqual(self.inspect(self.scene(), [20.0, 5.0])["route"], "aligned")

    # เทสต์ว่าไม่มี threshold ฉากให้ใช้คะแนนทั้งรูป | input: เม็ดเอียง, scene_threshold=None | output: score = 20.0
    def test_without_scene_threshold_uses_whole_image(self):
        # ตรวจเม็ดเอียงโดยไม่มี threshold ฉาก | input: scene(45), [20.0], None | output: result
        result = self.inspect(self.scene(45), [20.0], scene_threshold=None)
        # คะแนนต้องมาจากโมเดลทั้งรูป = 20.0 | input: result | output: assertion
        self.assertEqual(result["score"], 20.0)

    # เทสต์ว่าวัตถุที่ถูกข้ามทำให้มีคำเตือนหนึ่งข้อ | input: ฉากที่มีวัตถุเล็กเพิ่ม | output: score = 5.0 และคำเตือน 1 ข้อที่มี "ไม่ถูกตรวจ"
    def test_ignored_object_warns_on_both_routes(self):
        # สร้างฉากเม็ดเดียว | input: - | output: scene
        scene = self.scene()
        # ล้างมุมซ้ายบนเป็นสีดำเพื่อวางวัตถุเล็ก | input: scene | output: scene แก้แล้ว
        scene[:120, :120] = 0
        # วาดวัตถุเล็กกว่าเม็ดยามากที่มุม
        ellipse(scene, (60, 60), (50, 35))  # Far below the size of a pill.
        # ตรวจฉากนี้ | input: scene, [5.0] | output: result
        result = self.inspect(scene, [5.0])
        # คะแนนต้องเป็น 5.0 | input: result | output: assertion
        self.assertEqual(result["score"], 5.0)
        # ต้องมีคำเตือนข้อเดียว | input: result["warnings"] | output: assertion
        self.assertEqual(len(result["warnings"]), 1)
        # ข้อความเตือนต้องมีคำว่า "ไม่ถูกตรวจ" | input: warnings[0] | output: assertion
        self.assertIn("ไม่ถูกตรวจ", result["warnings"][0])

    # เทสต์ว่าเม็ดเล็กเกินไปมีคำเตือนหลังจัดท่า | input: เม็ดเล็ก | output: คำเตือนข้อแรกมีคำว่า "สั้นกว่า"
    def test_small_pill_warns_after_alignment(self):
        # สร้างฉากดำ 800x800 | input: - | output: scene
        scene = np.zeros((800, 800, 3), np.uint8)
        # วาดเม็ดเล็ก | input: scene | output: scene มีวงรีเล็ก
        ellipse(scene, (400, 400), (60, 40))
        # คำเตือนข้อแรกต้องมีคำว่า "สั้นกว่า" | input: inspect(scene, []) | output: assertion
        self.assertIn("สั้นกว่า", self.inspect(scene, [])["warnings"][0])

    # เทสต์ว่าเศษผงเล็กมากไม่ทำให้เตือน | input: จุดเล็กที่มุม | output: warnings = []
    def test_speck_gives_no_warning(self):
        # สร้างฉากเม็ดเดียว | input: - | output: scene
        scene = self.scene()
        # วาดเศษผงรัศมี 8 | input: scene | output: scene มีจุดเล็ก
        ellipse(scene, (60, 60), (8, 8))
        # ต้องไม่มีคำเตือน | input: inspect(scene, [5.0]) | output: assertion
        self.assertEqual(self.inspect(scene, [5.0])["warnings"], [])

    # เทสต์ว่าสีแปลกทำให้เม็ดคะแนนต่ำถูกแจ้งผิดปกติ | input: เม็ดปกติ และเม็ดมีรอยเปื้อนน้ำเงิน | output: ปกติผ่าน ส่วนรอยเปื้อนถูกแจ้ง
    def test_odd_colour_flags_a_low_scoring_pill(self):
        # สร้างเม็ดปกติ | input: - | output: normal
        normal = specked_pill()
        # สร้างตารางสีที่เคยเห็นจากเม็ดปกติ แล้วเก็บไว้ให้ตัวช่วย inspect ใช้ | input: normal | output: self.colour_seen
        self.colour_seen = build_seen([(normal, find_pills(normal)[0])])
        # เม็ดปกติต้องไม่ถูกแจ้งผิดปกติ | input: normal, [5.0] | output: assertion
        self.assertFalse(self.inspect(normal, [5.0])["is_anomalous"])
        # ตรวจเม็ดที่มีรอยเปื้อนสีน้ำเงิน (คะแนนโมเดลยังต่ำ 5.0) | input: specked_pill(stain), [5.0] | output: result
        result = self.inspect(specked_pill(stain=(40, 40, 200)), [5.0])
        # คะแนนโมเดลคงเป็น 5.0 | input: result | output: assertion
        self.assertEqual(result["score"], 5.0)
        # ต้องพบสีแปลก | input: result | output: assertion
        self.assertTrue(result["odd_colour"])
        # ต้องถูกแจ้งผิดปกติเพราะสีแปลก | input: result | output: assertion
        self.assertTrue(result["is_anomalous"])

    # ตัวช่วยสร้าง Inspector จำลองแบบ pillQC แล้วตรวจรูป | input: scene = RGB, scores = list คะแนนของโมเดลปลอม | output: dict ผลตรวจ
    def pillqc(self, scene, scores):
        # ทำให้ list คะแนนเป็นตัวไล่ค่า | input: scores | output: queue
        queue = iter(scores)
        # สร้าง Inspector เปล่า ๆ | input: - | output: inspector
        inspector = Inspector.__new__(Inspector)
        # โมเดลปลอมคืนคะแนนถัดไปและ anomaly map ศูนย์ | input: batch | output: SimpleNamespace
        inspector.model = lambda batch: SimpleNamespace(pred_score=torch.tensor([next(queue)]), anomaly_map=torch.zeros(1, 1, 8, 8))
        # transform ที่ไม่ทำอะไร | input: tensor | output: tensor เดิม
        inspector.transform = lambda tensor: tensor
        # ตั้งขนาดรูป 8, threshold 10 และ threshold ฉาก 12 | input: - | output: attribute ของ inspector
        inspector.image_size, inspector.threshold, inspector.scene_threshold = 8, 10.0, 12.0
        # ไม่มีตารางสีและ threshold สี | input: - | output: attribute เป็น None
        inspector.colour_seen = inspector.colour_threshold = inspector.scene_colour_threshold = None
        # ใช้ท่าวางของ pillQC | input: PILLQC_POSE | output: inspector.pose
        inspector.pose = PILLQC_POSE
        # แทน inspect_scene ด้วยตัวปลอมที่ระบุเส้นทาง "pills" | input: image, masks | output: dict {route, pills}
        inspector.inspect_scene = lambda image, masks: {"route": "pills", "pills": [{} for _ in masks]}
        # ตรวจรูป | input: scene | output: dict ผลตรวจ
        return inspector.inspect(Image.fromarray(scene))

    # เทสต์ว่ารูป crop ของ pillQC คงคะแนนทั้งรูป | input: crop ขอบสว่าง คะแนน 20.0 | output: score 20.0, ผิดปกติ, ไม่มีคำเตือน
    def test_pillqc_crop_keeps_whole_image_score(self):
        # สร้างรูป crop ขอบเขียวน้ำเงินแบบ pillQC | input: - | output: crop
        crop = np.full((225, 225, 3), (70, 120, 120), np.uint8)  # Teal border, like a pillQC crop.
        # วาดเม็ดกลมกลางรูป | input: crop | output: crop มีเม็ด
        cv2.circle(crop, (112, 112), 74, (240, 190, 180), -1)
        # วาดเศษผงที่มุม ซึ่งเส้นทางนี้ต้องไม่เตือน
        cv2.circle(crop, (20, 20), 6, (240, 190, 180), -1)  # A speck: no warning on this route.
        # ตรวจ crop นี้ | input: crop, [20.0] | output: result
        result = self.pillqc(crop, [20.0])
        # ต้องได้คะแนน 20.0, ผิดปกติ และไม่มีคำเตือน | input: result | output: assertion
        self.assertEqual((result["score"], result["is_anomalous"], result["warnings"]), (20.0, True, []))

    # เทสต์ว่ายา pillQC บนพื้นมืดถูกตัดเป็นรายเม็ด | input: เม็ดกลม 3 เม็ดบนพื้นดำ | output: route "pills", 3 เม็ด, ไม่มีคำเตือน
    def test_pillqc_pills_on_dark_background_are_cut_out(self):
        # สร้างฉากดำ 600x900 | input: - | output: scene
        scene = np.zeros((600, 900, 3), np.uint8)
        # วนวางเม็ดกลม 3 เม็ดที่ x ต่างกัน
        for x in (150, 450, 750):
            # วาดเม็ดกลมรัศมี 120 | input: scene, x | output: scene มีเม็ด
            cv2.circle(scene, (x, 300), 120, (240, 190, 180), -1)
        # ตรวจฉาก (ไม่ต้องใช้คะแนนโมเดล) | input: scene, [] | output: result
        result = self.pillqc(scene, [])
        # ต้องไปเส้นทาง "pills" มี 3 เม็ด และไม่มีคำเตือน | input: result | output: assertion
        self.assertEqual((result["route"], len(result["pills"]), result["warnings"]), ("pills", 3, []))

    # เทสต์ว่ารูป crop ของ pillQC ให้คะแนนเฉพาะเม็ดยา | input: crop ที่ผิดปกติสูงเฉพาะมุมพื้น | output: score 0.0 และไม่ผิดปกติ
    def test_pillqc_crop_scores_the_pill_only(self):
        # สร้างรูป crop ขอบเขียวน้ำเงิน | input: - | output: crop
        crop = np.full((225, 225, 3), (70, 120, 120), np.uint8)
        # วาดเม็ดกลมกลางรูป | input: crop | output: crop มีเม็ด
        cv2.circle(crop, (112, 112), 74, (240, 190, 180), -1)
        # สร้าง anomaly map ศูนย์ขนาด 32x32 | input: - | output: hot_floor = tensor (1, 1, 32, 32)
        hot_floor = torch.zeros(1, 1, 32, 32)
        # ตั้งค่าร้อน 50 ที่มุมซ้ายบน ซึ่งอยู่ไกลเม็ด
        hot_floor[..., 0, 0] = 50.0  # Dirt in the corner, far from the pill.
        # สร้าง Inspector เปล่า ๆ | input: - | output: inspector
        inspector = Inspector.__new__(Inspector)
        # โมเดลปลอมคืนคะแนน 50 และ map ที่ร้อนเฉพาะมุม | input: batch | output: SimpleNamespace
        inspector.model = lambda batch: SimpleNamespace(pred_score=torch.tensor([50.0]), anomaly_map=hot_floor)
        # transform ที่ไม่ทำอะไร | input: tensor | output: tensor เดิม
        inspector.transform = lambda tensor: tensor
        # ตั้งขนาดรูป 32, threshold 10 และ threshold ฉาก 12 | input: - | output: attribute ของ inspector
        inspector.image_size, inspector.threshold, inspector.scene_threshold = 32, 10.0, 12.0
        # ไม่มีตารางสีและ threshold สี | input: - | output: attribute เป็น None
        inspector.colour_seen = inspector.colour_threshold = inspector.scene_colour_threshold = None
        # ใช้ท่า pillQC และให้คะแนนเฉพาะเม็ดยา | input: PILLQC_POSE, "pill" | output: inspector.pose, inspector.score_area
        inspector.pose, inspector.score_area = PILLQC_POSE, "pill"
        # ตรวจ crop | input: crop | output: result
        result = inspector.inspect(Image.fromarray(crop))
        # คะแนนต้องเป็น 0.0 (ไม่นับมุมพื้น) และไม่ผิดปกติ | input: result | output: assertion
        self.assertEqual((result["score"], result["is_anomalous"]), (0.0, False))

    # เทสต์ว่ารูปปกติไม่มีคำเตือน | input: เม็ดท่าฝึก และเม็ดเอียง | output: warnings = [] ทั้งคู่
    def test_clean_images_have_no_warnings(self):
        # เม็ดท่าฝึกต้องไม่มีคำเตือน | input: scene(), [5.0] | output: assertion
        self.assertEqual(self.inspect(self.scene(), [5.0])["warnings"], [])
        # เม็ดเอียงที่ถูกจัดท่าต้องไม่มีคำเตือน | input: scene(45), [] | output: assertion
        self.assertEqual(self.inspect(self.scene(45), [])["warnings"], [])


# รันเทสต์ทั้งหมดเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เริ่มตัวรันเทสต์ของ unittest | input: - | output: ผลผ่าน/ไม่ผ่านของเทสต์
    unittest.main()
