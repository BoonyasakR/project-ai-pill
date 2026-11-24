"""Run the locked PatchCore model on images with one or several pills."""
# นำเข้า base64: แปลงไบต์รูปเป็นข้อความเพื่อฝังใน data URL | input: - | output: โมดูล base64
import base64
# นำเข้า io: ใช้ BytesIO เป็นไฟล์จำลองในหน่วยความจำ | input: - | output: โมดูล io
import io
# นำเข้า json: อ่านไฟล์ผลลัพธ์ฉาก | input: - | output: โมดูล json
import json
# นำเข้า time: จับเวลาประมวลผล | input: - | output: โมดูล time
import time
# นำเข้า Path: จัดการ path ไฟล์ | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า OpenCV: ประมวลผล/วาดบนรูป | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า PyTorch: รันโมเดลและจัดการ tensor | input: - | output: โมดูล torch
import torch
# นำเข้าโมเดล PatchCore จาก anomalib | input: - | output: คลาส PatchcoreModel
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
# นำเข้า colormaps ของ matplotlib: ใช้แปลงค่าความผิดปกติเป็นสี | input: - | output: ตารางสีที่ชื่อ colormaps
from matplotlib import colormaps
# นำเข้า Image ของ Pillow: เปิด/สร้างรูป | input: - | output: คลาส Image
from PIL import Image
# นำเข้า Compose (ต่อขั้นแปลงรูป), Normalize (ปรับค่าเฉลี่ย/ส่วนเบี่ยงเบน), Resize (ปรับขนาด) | input: - | output: คลาสดังกล่าว
from torchvision.transforms.v2 import Compose, Normalize, Resize
# นำเข้า pil_to_tensor (PIL -> tensor) และ to_dtype (แปลงชนิดข้อมูล) | input: - | output: ฟังก์ชันดังกล่าว
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

# นำเข้าจำนวนช่องสี (BINS) และฟังก์ชันวัดสีแปลกปลอม (novel_colour) | input: - | output: BINS, novel_colour
from colour_check import BINS, novel_colour
# นำเข้าค่าคงที่และฟังก์ชันตัดแบ่งเม็ดยาจาก segmentation.py | input: - | output: ชื่อต่าง ๆ ตามที่ระบุ
from segmentation import (
    CANVAS, MIN_RELIABLE_LENGTH, MIN_WARNED_AREA, align_pill, find_blobs, in_training_pose, keep_pills, pill_pose,
    pink_pill_mask, round_pill_region,
)

# How each tablet type is cut out of a multi-pill image. MVTec pills are oval, so
# they are turned to the training pose and scored both ways round. pillQC pills
# are round: no turn, scaled to the median pill length of a pillQC crop (148 of 225 px).
# ท่าวางของ MVTec (เม็ดรูปไข่): หมุนตามแกน, ให้คะแนนทั้งสองทิศ (ปกติ/กลับ 180 องศา), ความยาวขั้นต่ำเชื่อถือได้ | input: - | output: MVTEC_POSE = dict {align, flips, min_length}
MVTEC_POSE = {"align": {}, "flips": (False, True), "min_length": MIN_RELIABLE_LENGTH}
# ท่าวางของ pillQC (เม็ดกลม): ผืนผ้าใบ 225, ความยาวเม็ด 148, ไม่หมุน, ไม่กลับทิศ | input: - | output: PILLQC_POSE = dict {align, flips, min_length}
PILLQC_POSE = {"align": {"canvas": 225, "pill_length": 148, "rotate": False}, "flips": (False,), "min_length": 148}
# ความสว่างขอบรูปที่แยกรูป pillQC crop (ขอบสีเขียวน้ำเงิน สว่าง >=72) ออกจากฉากพื้นมืด (~18) | input: - | output: PILLQC_CROP_BACKGROUND = 45
PILLQC_CROP_BACKGROUND = 45  # pillQC crops have a teal border (median 72 or more); scenes are dark (about 18).

# กันขอบเม็ดยา 5 พิกเซลที่พื้นหลังซึมเข้ามา ไม่นำมาคิดคะแนน | input: - | output: PILL_EDGE_PIXELS = 5
PILL_EDGE_PIXELS = 5  # Ignore this rim of the pill, where the background bleeds in.
# สีเขียว (RGB) ใช้วาดกรอบเม็ดยาปกติ | input: - | output: MARK_GOOD = (40, 200, 110)
MARK_GOOD = (40, 200, 110)
# สีแดง (RGB) ใช้วาดกรอบเม็ดยาผิดปกติ | input: - | output: MARK_BAD = (240, 60, 50)
MARK_BAD = (240, 60, 50)
# ความสูงของแถวเมื่อเรียงเม็ดยา (พิกเซล) มีผลแค่ลำดับหมายเลข | input: - | output: ROW_HEIGHT = 300
ROW_HEIGHT = 300  # Pixels. Only affects numbering order.
# สัดส่วนของ threshold ที่แสดงเป็นสีเย็นสุด | input: - | output: HEATMAP_LOW = 0.5
HEATMAP_LOW = 0.5  # Share of the threshold shown as the coldest colour.
# สัดส่วนของ threshold ที่แสดงเป็นสีร้อนสุด | input: - | output: HEATMAP_HIGH = 1.25
HEATMAP_HIGH = 1.25  # Share of the threshold shown as the hottest colour.
# เลือกอุปกรณ์ประมวลผล: GPU ของ Apple (mps) ถ้ามี ไม่งั้นใช้ CPU | input: - | output: DEVICE = "mps" หรือ "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # The Apple GPU is several times faster.


# ฟังก์ชันซ้อนสี heatmap ทับรูป | input: image = PIL.Image, anomaly_map = อาร์เรย์ float (H, W) คะแนนผิดปกติรายพิกเซล, threshold = ค่าเกณฑ์ float | output: PIL.Image RGB รูปที่ซ้อน heatmap แล้ว
def overlay_heatmap(image, anomaly_map, threshold):
    """Colour an anomaly map on a fixed scale, so normal pills stay cool."""
    # ปรับคะแนนให้เป็นสเกล 0-1 ตามเกณฑ์คงที่ (LOW -> 0, HIGH -> 1) | input: anomaly_map, threshold | output: scaled = float (H, W)
    scaled = (anomaly_map / threshold - HEATMAP_LOW) / (HEATMAP_HIGH - HEATMAP_LOW)
    # จำกัดค่า 0-1 แล้วแปลงเป็นสีด้วย colormap "jet" เก็บเฉพาะ R, G, B | input: scaled | output: colours = float (H, W, 3) ค่า 0-1
    colours = colormaps["jet"](np.clip(scaled, 0, 1))[..., :3]
    # แปลงรูปเป็น RGB อาร์เรย์ float32 และหาร 255 ให้เป็นช่วง 0-1 | input: image | output: base = float32 (H, W, 3)
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    # ผสมรูปเดิม 55% กับสี heatmap 45% | input: base, colours | output: blended = float (H, W, 3)
    blended = 0.55 * base + 0.45 * colours
    # แปลงกลับเป็น 0-255 uint8 แล้วสร้างเป็นรูป PIL | input: blended | output: PIL.Image
    return Image.fromarray((blended * 255).astype(np.uint8))


# ฟังก์ชันคีย์สำหรับเรียงลำดับเม็ดยา | input: mask = bool (H, W) ของเม็ดยา | output: tuple (หมายเลขแถว int, ตำแหน่ง x เฉลี่ย float)
def reading_order(mask):
    """Sort pills top to bottom in rows of about one pill height, then left to right."""
    # หาพิกัดแถว (ys) และคอลัมน์ (xs) ของพิกเซลในเม็ดยา | input: mask | output: ys, xs = อาร์เรย์ 1 มิติ
    ys, xs = np.nonzero(mask)
    # คืนหมายเลขแถว (ตำแหน่ง y เฉลี่ยหารปัดลงด้วยความสูงแถว) และ x เฉลี่ย เพื่อเรียงบนลงล่างแล้วซ้ายไปขวา | input: ys, xs | output: (แถว, x เฉลี่ย)
    return int(ys.mean() // ROW_HEIGHT), xs.mean()


# ฟังก์ชันแปลงรูปเป็น data URL แบบ PNG เพื่อส่งให้เว็บแสดงได้ทันที | input: image = PIL.Image | output: str ขึ้นต้นด้วย "data:image/png;base64,"
def to_data_url(image):
    # สร้างไฟล์จำลองในหน่วยความจำ | input: - | output: buffer = BytesIO ว่าง
    buffer = io.BytesIO()
    # บันทึกรูปลง buffer เป็น PNG | input: image | output: buffer ที่มีไบต์ PNG
    image.save(buffer, format="PNG")
    # เข้ารหัสไบต์เป็น base64 แปลงเป็นข้อความ แล้วต่อท้ายส่วนหัว data URL | input: buffer | output: str data URL
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ฟังก์ชันวัดความสว่างของขอบรูป (ใช้แยกรูป crop ที่มีขอบสว่างออกจากฉากพื้นมืด) | input: array = รูป RGB (H, W, 3) | output: float ค่ามัธยฐานความสว่างของขอบ 8 พิกเซลทั้งสี่ด้าน
def border_brightness(array):
    # หาความสว่างเฉลี่ยของ 3 ช่องสีเป็นรูปขาวดำ | input: array | output: gray = float (H, W)
    gray = array.mean(axis=2)
    # รวมแถบขอบบน ล่าง ซ้าย ขวา (หนา 8 พิกเซล) เป็นแถวเดียว แล้วหามัธยฐาน | input: gray | output: float
    return float(np.median(np.concatenate([gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()])))


# คลาสตัวตรวจเม็ดยา ห่อโมเดล PatchCore และขั้นตอนทั้งหมด
class Inspector:
    # ท่าวางเริ่มต้น (ค่าของคลาส) ใช้ของ MVTec
    pose = MVTEC_POSE
    # พื้นที่ที่ใช้ให้คะแนนเริ่มต้นคือทั้งรูป; "pill" = ให้คะแนนเฉพาะเม็ดยา ตามที่ threshold ถูกตั้ง
    score_area = "image"  # "pill": a single crop is scored on its pill only, as its threshold was.

    # ตัวสร้างวัตถุ: โหลดไฟล์โมเดลและตั้งค่าทุกอย่าง | input: model_path = path ไฟล์ .pt, scene_results = path ไฟล์ผลลัพธ์ฉาก (หรือ None), pose = ท่าวาง | output: วัตถุ Inspector ที่พร้อมใช้
    def __init__(self, model_path, scene_results=None, pose=MVTEC_POSE):
        # เก็บท่าวางที่ใช้กับวัตถุนี้ | input: pose | output: self.pose
        self.pose = pose
        # อ่านไฟล์ผลลัพธ์ฉากเป็น dict ถ้ามี path ไม่งั้นใช้ dict ว่าง | input: scene_results | output: scene_data = dict
        scene_data = json.loads(Path(scene_results).read_text()) if scene_results else {}
        # threshold ของคะแนนต่อเม็ดในรูปหลายเม็ด (None ถ้าไม่มี) | input: scene_data | output: self.scene_threshold
        self.scene_threshold = scene_data.get("threshold")
        # threshold ของพื้นที่สีแปลกในรูปหลายเม็ด (None ถ้าไม่มี) | input: scene_data | output: self.scene_colour_threshold
        self.scene_colour_threshold = scene_data.get("colour_threshold")
        # โหลดไฟล์ checkpoint ของโมเดลลง CPU (อนุญาตเฉพาะน้ำหนัก เพื่อความปลอดภัย) | input: model_path | output: checkpoint = dict
        checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        # threshold ของคะแนนทั้งรูปที่บันทึกไว้ตอนฝึก | input: checkpoint["threshold"] | output: self.threshold (float)
        self.threshold = float(checkpoint["threshold"])
        # ขนาดรูปที่โมเดลรับเข้า (ด้านของสี่เหลี่ยมจัตุรัส) | input: checkpoint["image_size"] | output: self.image_size (int)
        self.image_size = int(checkpoint["image_size"])
        # พื้นที่ให้คะแนน ("image" ถ้า checkpoint ไม่ระบุ) | input: checkpoint | output: self.score_area (str)
        self.score_area = checkpoint.get("score_area", "image")
        # น้ำหนักของโมเดล (รวม memory bank) | input: checkpoint["state_dict"] | output: state = dict ของ tensor
        state = checkpoint["state_dict"]
        # Older model files have no colour table. They still work, without the colour check.
        # ตั้งค่าเริ่มต้นว่าไม่มีตารางสี (ไฟล์โมเดลรุ่นเก่ายังใช้ได้ แต่ไม่ตรวจสี) | input: - | output: self.colour_seen, self.colour_threshold = None, None
        self.colour_seen, self.colour_threshold = None, None
        # ถ้า checkpoint มีตารางสีที่เคยเห็น
        if "colour_seen" in checkpoint:
            # แกะบิตที่บีบอัดไว้เป็นอาร์เรย์ 0/1 แล้วตัดให้เหลือ 64^3 ค่า | input: checkpoint["colour_seen"] (tensor uint8) | output: bits = อาร์เรย์ 1 มิติ
            bits = np.unpackbits(checkpoint["colour_seen"].numpy())[:BINS ** 3]
            # จัดรูปเป็นลูกบาศก์ 64x64x64 ชนิด bool | input: bits | output: self.colour_seen
            self.colour_seen = bits.reshape((BINS,) * 3).astype(bool)
            # threshold ของพื้นที่สีแปลก | input: checkpoint["colour_threshold"] | output: self.colour_threshold (float)
            self.colour_threshold = float(checkpoint["colour_threshold"])

        # สร้างโครงโมเดล PatchCore ตาม layer และ backbone ที่บันทึกไว้ (ไม่โหลดน้ำหนักที่ฝึกมาก่อน) | input: checkpoint["layers"], checkpoint["backbone"] | output: self.model
        self.model = PatchcoreModel(
            layers=checkpoint["layers"], backbone=checkpoint["backbone"], pre_trained=False
        )
        # เตรียม memory bank เปล่าให้ขนาดเท่าของที่บันทึกไว้ เพื่อให้โหลดน้ำหนักได้ | input: state["memory_bank"] | output: self.model.memory_bank
        self.model.memory_bank = torch.empty_like(state["memory_bank"])
        # โหลดน้ำหนักทั้งหมดเข้าโมเดล | input: state | output: โมเดลมีน้ำหนักที่ฝึกแล้ว
        self.model.load_state_dict(state)
        # ย้ายโมเดลไปอุปกรณ์ที่เลือก และสลับเป็นโหมดประเมินผล (ไม่ฝึก) | input: DEVICE | output: self.model พร้อมใช้
        self.model.to(DEVICE).eval()
        # ประกอบขั้นแปลงรูป: ปรับขนาดเป็น image_size และปรับค่าสีตามค่า ImageNet
        self.transform = Compose([
            # ปรับขนาดรูปเป็นสี่เหลี่ยมจัตุรัสแบบลดรอยหยัก | input: tensor รูป | output: tensor ขนาด (image_size, image_size)
            Resize((self.image_size, self.image_size), antialias=True),
            # ลบค่าเฉลี่ยและหารส่วนเบี่ยงเบนของ ImageNet ต่อช่องสี | input: tensor 0-1 | output: tensor ที่ปรับแล้ว
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ปิดการคำนวณ gradient เพื่อประหยัดเวลา/หน่วยความจำตอนอนุมาน
    @torch.inference_mode()
    # ฟังก์ชันให้คะแนนเม็ดยาหนึ่งเม็ดจากรูปที่ใหญ่กว่า | input: image = RGB (H, W, 3), mask = mask เม็ดนี้, others = mask เม็ดข้างเคียง | output: tuple (คะแนน float, รูปที่จัดแล้ว, anomaly map ที่ตัดนอกเม็ดเป็น 0, พื้นที่สีแปลก int, mask สีแปลกหรือ None)
    def score_pill(self, image, mask, others):
        """Score one pill from a larger image, in both 180-degree orientations.

        Only the pill area counts, because the plain filled background looks unlike
        the training background. The right way up matches training best, so the
        lower score is kept.
        """
        # จัดเม็ดยาให้เป็นท่าฝึกตามแต่ละทิศ (flips) ที่ท่าวางกำหนด | input: image, mask, others, flip, ค่า align | output: views = list ของ (รูปที่จัดแล้ว, mask เม็ดบนรูปนั้น)
        views = [align_pill(image, mask, others, flip, **self.pose["align"]) for flip in self.pose["flips"]]
        # แปลงรูปแต่ละทิศเป็น tensor แล้วซ้อนเป็น batch
        batch = torch.stack([
            # รูป HxWx3 -> tensor 3xHxW ชนิด float 0-1 -> ปรับขนาดและ normalize | input: view | output: tensor (3, image_size, image_size)
            self.transform(to_dtype(torch.from_numpy(view).permute(2, 0, 1), torch.float32, scale=True))
            # วนทุกทิศ (ตัว mask ไม่ใช้ในขั้นนี้)
            for view, _ in views
        ])
        # รันโมเดลบน batch ลด dimension ช่องออก แล้วย้ายกลับ CPU เป็นอาร์เรย์ | input: batch | output: maps = float (จำนวนทิศ, image_size, image_size)
        maps = self.model(batch.to(DEVICE)).anomaly_map.squeeze(1).cpu().numpy()
        # ขนาดเป้าหมายของ anomaly map | input: self.image_size | output: size = (w, h)
        size = (self.image_size, self.image_size)
        # เคอร์เนลสำหรับกัดขอบเม็ดยาเข้า | input: PILL_EDGE_PIXELS | output: kernel = uint8 (5, 5)
        kernel = np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)
        # สร้างพื้นที่เม็ดยาแต่ละทิศในขนาดของ anomaly map
        regions = [
            # ย่อ mask ให้เท่า map (จุดใกล้สุด) แล้วกัดขอบเข้า 5 พิกเซล แปลงเป็น bool | input: pill mask | output: bool (image_size, image_size)
            cv2.erode(cv2.resize(pill.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST), kernel) > 0
            # วนทุกทิศ (รูปไม่ใช้ในขั้นนี้)
            for _, pill in views
        ]
        # คะแนนของแต่ละทิศ = ค่าสูงสุดของ map ภายในพื้นที่เม็ดยา | input: maps, regions | output: scores = list ของ float
        scores = [float(score_map[region].max()) for score_map, region in zip(maps, regions)]
        # เลือกทิศที่คะแนนต่ำสุด (ท่าที่ตรงกับรูปฝึกที่สุด) | input: scores | output: best = ดัชนี int
        best = int(np.argmin(scores))
        # ตรวจสีแปลกปลอมบนรูปทิศที่ดีที่สุด (กระจาย (รูป, mask) เป็นอาร์กิวเมนต์) | input: views[best] | output: colour_area (int), colour_mask (bool หรือ None)
        colour_area, colour_mask = self.check_colour(*views[best])
        # คืนคะแนน, รูป, map (นอกเม็ดเป็น 0), พื้นที่สีแปลก และ mask สีแปลก | input: ค่าข้างต้น | output: tuple 5 ค่า
        return scores[best], views[best][0], np.where(regions[best], maps[best], 0), colour_area, colour_mask

    # ฟังก์ชันวัดสีแปลกปลอมบนเม็ดที่จัดท่าบนผืนผ้าใบฝึกแล้ว | input: image = RGB บนผืนผ้าใบฝึก, mask = mask เม็ดยา | output: tuple (พื้นที่กลุ่มใหญ่สุด int, mask สีแปลกขนาดเท่า heatmap หรือ None)
    def check_colour(self, image, mask):
        """Measure unseen colour on a pill posed on the training canvas.

        Returns the largest blob area and the unseen pixels at heatmap size.
        """
        # ถ้าไม่มีตารางสี (โมเดลรุ่นเก่า) ให้ข้ามการตรวจ
        if self.colour_seen is None:
            # คืนพื้นที่ 0 และไม่มี mask | input: - | output: (0, None)
            return 0, None
        # หาพื้นที่กลุ่มสีแปลกใหญ่สุดและ mask พิกเซลสีแปลก | input: self.colour_seen, image, mask | output: area (int), novel (bool)
        area, novel = novel_colour(self.colour_seen, image, mask)
        # ขนาดเป้าหมายเท่า heatmap | input: self.image_size | output: size = (w, h)
        size = (self.image_size, self.image_size)
        # ย่อ mask สีแปลกให้เท่า heatmap (เฉลี่ยพื้นที่) แล้วเก็บพิกเซลที่มีค่า >0 เป็น bool | input: novel, size | output: (area, bool (image_size, image_size))
        return area, cv2.resize(novel.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0

    # ฟังก์ชันทำให้บริเวณสีแปลกร้อนบน heatmap | input: anomaly_map = float (H, W), colour_mask = bool (H, W), threshold = float | output: อาร์เรย์ float (H, W) ที่บริเวณสีแปลกถูกยกให้ถึงอย่างน้อยค่าร้อนสุด
    def mark_colour(self, anomaly_map, colour_mask, threshold):
        """Show flagged unseen colour as hot on the heatmap."""
        # ที่พิกเซลสีแปลก ใช้ค่ามากกว่าระหว่างคะแนนเดิมกับค่าร้อนสุด ที่อื่นคงค่าเดิม | input: anomaly_map, colour_mask, threshold | output: อาร์เรย์ float (H, W)
        return np.where(colour_mask, np.maximum(anomaly_map, threshold * HEATMAP_HIGH), anomaly_map)

    # ฟังก์ชันตรวจเม็ดยาทุกเม็ดในรูปหลายเม็ดและทำเครื่องหมายบนรูป | input: image = RGB (H, W, 3), masks = list ของ mask เม็ดยา | output: dict {score, threshold, is_anomalous, latency_ms, image, pills}
    def inspect_scene(self, image, masks):
        """Inspect every pill in an RGB array and mark the results on the image."""
        # เริ่มจับเวลา | input: - | output: start = float (วินาที)
        start = time.perf_counter()
        # รวมทุกเม็ดเป็น mask เดียว | input: masks | output: everything = bool (H, W)
        everything = np.any(masks, axis=0)
        # สำเนารูปไว้วาดกรอบ (ไม่แก้รูปเดิม) | input: image | output: marked = RGB (H, W, 3)
        marked = image.copy()
        # เตรียม list เก็บผลของแต่ละเม็ด | input: - | output: pills = []
        pills = []
        # วนทีละเม็ดตามลำดับการอ่าน (บนลงล่าง ซ้ายไปขวา) เริ่มหมายเลข 1
        for number, mask in enumerate(sorted(masks, key=reading_order), start=1):
            # ให้คะแนนเม็ดนี้ โดยส่ง mask เม็ดอื่น ๆ (everything & ~mask) เพื่อลบออกจากรูป | input: image, mask, everything & ~mask | output: score, view, anomaly_map, colour_area, colour_mask
            score, view, anomaly_map, colour_area, colour_mask = self.score_pill(image, mask, everything & ~mask)
            # สีแปลกเกินเกณฑ์ของฉากหรือไม่ (ต้องมีเกณฑ์ก่อน) | input: self.scene_colour_threshold, colour_area | output: odd_colour = bool
            odd_colour = self.scene_colour_threshold is not None and colour_area > self.scene_colour_threshold
            # ถ้าพบสีแปลก
            if odd_colour:
                # ยกบริเวณสีแปลกให้ร้อนบน heatmap | input: anomaly_map, colour_mask, self.scene_threshold | output: anomaly_map ใหม่
                anomaly_map = self.mark_colour(anomaly_map, colour_mask, self.scene_threshold)
            # ผิดปกติถ้าคะแนนเกินเกณฑ์ หรือมีสีแปลก | input: score, self.scene_threshold, odd_colour | output: abnormal = bool
            abnormal = score > self.scene_threshold or odd_colour
            # เลือกสีกรอบ: แดงถ้าผิดปกติ เขียวถ้าปกติ | input: abnormal | output: colour = tuple RGB
            colour = MARK_BAD if abnormal else MARK_GOOD
            # หาเส้นขอบรอบนอกของเม็ดยา | input: mask | output: contours = list ของเส้นขอบ (ตัวที่สองไม่ใช้)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # วาดเส้นขอบหนา 3 พิกเซลบนรูป | input: marked, contours, colour | output: marked ถูกแก้ในที่เดิม
            cv2.drawContours(marked, contours, -1, colour, 3)
            # หาพิกัดพิกเซลของเม็ด แล้วกลับลำดับเป็น (x, y) | input: mask | output: x, y = อาร์เรย์ 1 มิติ
            x, y = np.nonzero(mask)[::-1]
            # เขียนหมายเลขเม็ดตรงกึ่งกลางเม็ด (เลื่อนเล็กน้อยให้ข้อความอยู่กลาง) | input: marked, number, ตำแหน่งกลาง, colour | output: marked มีตัวเลข
            cv2.putText(marked, str(number), (int(x.mean()) - 12, int(y.mean()) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
            # ย่อรูปเม็ดที่จัดแล้วให้เท่าขนาดของโมเดล | input: view | output: small = PIL.Image
            small = Image.fromarray(view).resize((self.image_size, self.image_size))
            # เก็บผลของเม็ดนี้ลง list
            pills.append({
                # หมายเลขเม็ด
                "number": number,
                # คะแนนความผิดปกติ
                "score": score,
                # มีสีแปลกหรือไม่
                "odd_colour": odd_colour,
                # ผิดปกติโดยรวมหรือไม่
                "is_anomalous": abnormal,
                # heatmap ของเม็ดนี้เป็น data URL | input: small, anomaly_map, threshold | output: str
                "heatmap": to_data_url(overlay_heatmap(small, anomaly_map, self.scene_threshold)),
            })
        # คืนผลรวมของทั้งรูป
        return {
            # คะแนนของทั้งรูป = คะแนนสูงสุดในบรรดาเม็ดยา
            "score": max(pill["score"] for pill in pills),
            # threshold ที่ใช้ตัดสิน
            "threshold": self.scene_threshold,
            # ผิดปกติถ้ามีเม็ดใดผิดปกติ
            "is_anomalous": any(pill["is_anomalous"] for pill in pills),
            # เวลาที่ใช้ทั้งหมดเป็นมิลลิวินาที
            "latency_ms": (time.perf_counter() - start) * 1000,
            # รูปที่วาดกรอบและหมายเลขแล้ว เป็น data URL
            "image": to_data_url(Image.fromarray(marked)),
            # ผลรายเม็ด
            "pills": pills,
        }

    # ฟังก์ชันหลักตรวจรูปที่ผู้ใช้อัปโหลด | input: image = PIL.Image | output: dict ผลตรวจ พร้อม key "warnings" (list ข้อความเตือนภาษาไทย)
    def inspect(self, image):
        # แปลงรูปเป็นโหมด RGB (ตัดช่องโปร่งใส/ขาวดำ) | input: image | output: image = PIL.Image RGB
        image = image.convert("RGB")
        # แปลงเป็นอาร์เรย์ (H, W, 3) | input: image | output: array = uint8
        array = np.asarray(image)
        # หากลุ่มวัตถุสว่างทั้งหมด | input: array | output: blobs = list ของ mask เรียงใหญ่ไปเล็ก
        blobs = find_blobs(array)
        # เก็บเฉพาะกลุ่มที่ใหญ่พอเป็นเม็ดยา | input: blobs | output: masks = list ของ mask
        masks = keep_pills(blobs)
        # ตัดสินและให้คะแนน | input: image, array, masks | output: result = dict
        result = self.judge(image, array, masks)
        # เตรียม list คำเตือน | input: - | output: result["warnings"] = []
        result["warnings"] = []
        # ถ้าเป็นรูป pillQC เม็ดเดียว (ไม่มีผลรายเม็ด)
        if self.pose is not MVTEC_POSE and "pills" not in result:
            # คืนผลเลย เพราะเศษสกปรกบนขอบรูปไม่ใช่เม็ดยาที่ต้องเตือน | input: result | output: result
            return result  # A pillQC crop: dirt specks on its teal border are not pills to warn about.
        # นับวัตถุเล็กที่ไม่ถูกตรวจ แต่ใหญ่พอควรเตือน (>= 1% ของวัตถุใหญ่สุด) | input: blobs, masks | output: ignored = int
        ignored = sum(blob.sum() >= MIN_WARNED_AREA * blobs[0].sum() for blob in blobs[len(masks):])
        # ถ้ามีวัตถุที่ถูกข้าม
        if ignored:
            # เพิ่มคำเตือนภาษาไทยว่ามีวัตถุเล็กที่ไม่ถูกตรวจ | input: ignored | output: result["warnings"] ยาวขึ้น 1 ข้อความ
            result["warnings"].append(f"มีวัตถุขนาดเล็ก {ignored} ชิ้นที่ไม่ถูกตรวจ เพราะเล็กกว่าเม็ดใหญ่สุดมาก")
        # ถ้าเป็นผลหลายเม็ด
        if "pills" in result:
            # ความยาวขั้นต่ำที่เชื่อถือได้ของท่าวางนี้ | input: self.pose | output: shortest = int
            shortest = self.pose["min_length"]
            # นับเม็ดที่แกนยาวสั้นกว่าขั้นต่ำ (pill_pose(...)[2] คือความยาว) | input: masks, shortest | output: small = int
            small = sum(pill_pose(mask)[2] < shortest for mask in masks)
            # ถ้ามีเม็ดที่สั้นเกินไป
            if small:
                # เพิ่มคำเตือนให้ถ่ายใกล้ขึ้น เพราะผลอาจแจ้งผิดปกติเกินจริง | input: small, shortest | output: result["warnings"] ยาวขึ้น 1 ข้อความ
                result["warnings"].append(
                    f"ยา {small} เม็ดสั้นกว่า {shortest} พิกเซล ผลอาจแจ้งผิดปกติเกินจริง ควรถ่ายให้ใกล้ขึ้น"
                )
        # คืนผลพร้อมคำเตือน | input: result | output: dict
        return result

    # ปิดการคำนวณ gradient ตอนอนุมาน
    @torch.inference_mode()
    # ฟังก์ชันตัดสินว่าจะตรวจแบบทั้งรูปหรือแยกรายเม็ด แล้วให้คะแนน | input: image = PIL.Image, array = RGB (H, W, 3), masks = list ของ mask เม็ดยา | output: dict ผลตรวจ (ทั้งรูป หรือรายเม็ดจาก inspect_scene)
    def judge(self, image, array, masks):
        # ตรวจแบบแยกเม็ดได้ก็ต่อเมื่อมี threshold ของฉาก | input: self.scene_threshold | output: can_align = bool
        can_align = self.scene_threshold is not None
        # ถ้าเป็นท่าของ pillQC
        if self.pose is not MVTEC_POSE:
            # A pillQC crop (teal border) keeps the whole-image score it was calibrated on;
            # anything else, such as pills on a dark background, is cut into pills.
            # ถ้ามีเม็ด และขอบรูปมืด (ไม่ใช่ crop ที่ขอบสว่าง) ให้แยกตรวจรายเม็ด
            if can_align and masks and border_brightness(array) < PILLQC_CROP_BACKGROUND:
                # ตรวจแบบฉากหลายเม็ด | input: array, masks | output: dict ผลตรวจ
                return self.inspect_scene(array, masks)
        # ท่า MVTec: ถ้ามีเม็ดเกิน 1 หรือเม็ดเดียวแต่ท่าไม่ตรงกับรูปฝึก ให้แยกตรวจรายเม็ด
        elif can_align and (len(masks) > 1 or (masks and not in_training_pose(masks[0]))):
            # ตรวจแบบฉากหลายเม็ด | input: array, masks | output: dict ผลตรวจ
            return self.inspect_scene(array, masks)
        # แปลงรูปเป็น tensor float 0-1 แล้วย้ายไปอุปกรณ์ | input: image | output: tensor (3, H, W)
        tensor = to_dtype(pil_to_tensor(image), torch.float32, scale=True).to(DEVICE)

        # เริ่มจับเวลาอนุมาน | input: - | output: start = float
        start = time.perf_counter()
        # รันโมเดลกับรูปทั้งรูป (เพิ่มมิติ batch) | input: tensor | output: output = ผลที่มี pred_score และ anomaly_map
        output = self.model(self.transform(tensor).unsqueeze(0))
        # คำนวณเวลาที่ใช้เป็นมิลลิวินาที | input: start | output: latency_ms = float
        latency_ms = (time.perf_counter() - start) * 1000

        # ดึงคะแนนทั้งรูปเป็นตัวเลข | input: output.pred_score | output: score = float
        score = float(output.pred_score.flatten()[0])
        # ถ้าแยกเม็ดได้ มีเม็ด คะแนนเกินเกณฑ์ และเป็นท่า MVTec
        if can_align and masks and score > self.threshold and self.pose is MVTEC_POSE:
            # The pose check cannot see an upside-down pill. A normal one passes once turned;
            # a real defect stays above the threshold, so its score is kept.
            # ลองพลิกรูปกลับ 180 องศา (flip ทั้งสองแกน) แล้วให้คะแนนอีกครั้ง | input: tensor | output: turned = ผลของรูปที่พลิก
            turned = self.model(self.transform(tensor.flip(1, 2)).unsqueeze(0))
            # ถ้าพลิกแล้วคะแนนไม่เกินเกณฑ์ แปลว่าเม็ดปกติแต่วางกลับหัว
            if float(turned.pred_score.flatten()[0]) <= self.threshold:
                # ตรวจใหม่แบบแยกเม็ดซึ่งจะลองทั้งสองทิศ | input: array, masks | output: dict ผลตรวจ
                return self.inspect_scene(array, masks)
        # แปลง anomaly map เป็นอาร์เรย์ (H, W) บน CPU | input: output.anomaly_map | output: anomaly_map = float (image_size, image_size)
        anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
        # ถ้าให้คะแนนเฉพาะเม็ดยา (pillQC)
        if self.score_area == "pill":
            # White dirt on the floor must not count, so only the pill is scored and shown.
            # หาบริเวณวงกลมของเม็ดชมพู เพื่อไม่นับสิ่งสกปรกบนพื้น | input: array | output: region = uint8 (H, W)
            region = round_pill_region(pink_pill_mask(array)).astype(np.uint8)
            # ย่อให้เท่าขนาด anomaly map (สลับ shape เป็น (กว้าง, สูง)) | input: region, anomaly_map.shape | output: region = uint8
            region = cv2.resize(region, anomaly_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
            # กัดขอบเข้า 5 พิกเซล แล้วแปลงเป็น bool | input: region | output: region = bool
            region = cv2.erode(region, np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)) > 0
            # ตั้งค่านอกเม็ดยาเป็น 0 | input: anomaly_map, region | output: anomaly_map ใหม่
            anomaly_map = np.where(region, anomaly_map, 0)
            # คะแนนใหม่ = ค่าสูงสุดภายในเม็ดยา | input: anomaly_map, region | output: score = float
            score = float(anomaly_map[region].max())
        # เริ่มต้นว่าไม่พบสีแปลก | input: - | output: odd_colour = False
        odd_colour = False
        # ถ้าพบเม็ดยาในรูป
        if masks:
            # The pill is posed like training, so resizing puts it on the training canvas.
            # ย่อ/ขยายรูปเป็นผืนผ้าใบฝึก 800x800 | input: array | output: canvas = RGB (800, 800, 3)
            canvas = cv2.resize(array, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
            # ปรับ mask ของเม็ดแรกให้เป็นขนาด 800x800 แล้วแปลงเป็น bool | input: masks[0] | output: pill = bool (800, 800)
            pill = cv2.resize(masks[0].astype(np.uint8), (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST) > 0
            # ตรวจสีแปลก | input: canvas, pill | output: colour_area (int), colour_mask (bool หรือ None)
            colour_area, colour_mask = self.check_colour(canvas, pill)
            # สีแปลกเกินเกณฑ์หรือไม่ (ต้องมีเกณฑ์ก่อน) | input: self.colour_threshold, colour_area | output: odd_colour = bool
            odd_colour = self.colour_threshold is not None and colour_area > self.colour_threshold
            # ถ้าพบสีแปลก
            if odd_colour:
                # ยกบริเวณสีแปลกให้ร้อนบน heatmap | input: anomaly_map, colour_mask, self.threshold | output: anomaly_map ใหม่
                anomaly_map = self.mark_colour(anomaly_map, colour_mask, self.threshold)
        # ย่อรูปต้นฉบับให้เท่าขนาดที่โมเดลใช้ เพื่อแสดงผล | input: image | output: resized = PIL.Image
        resized = image.resize((self.image_size, self.image_size))
        # คืนผลตรวจทั้งรูป
        return {
            # คะแนนความผิดปกติ
            "score": score,
            # threshold ที่ใช้ตัดสิน
            "threshold": self.threshold,
            # พบสีแปลกหรือไม่
            "odd_colour": odd_colour,
            # ผิดปกติถ้าคะแนนเกินเกณฑ์ หรือมีสีแปลก
            "is_anomalous": score > self.threshold or odd_colour,
            # เวลาที่ใช้ของโมเดล (มิลลิวินาที)
            "latency_ms": latency_ms,
            # รูปต้นฉบับที่ย่อแล้วเป็น data URL
            "image": to_data_url(resized),
            # รูปที่ซ้อน heatmap เป็น data URL
            "heatmap": to_data_url(overlay_heatmap(resized, anomaly_map, self.threshold)),
        }
