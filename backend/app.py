"""FastAPI service for the pill inspection prototype."""
# นำเข้าโมดูล io: ใช้ห่อไบต์ของไฟล์ให้เป็นสตรีมที่ PIL เปิดอ่านได้ | input: - | output: โมดูล io
import io
# นำเข้าโมดูล json: ใช้อ่านไฟล์ผลลัพธ์ .json | input: - | output: โมดูล json
import json
# นำเข้าโมดูล os: ใช้อ่านตัวแปรสภาพแวดล้อม (environment variable) | input: - | output: โมดูล os
import os
# นำเข้า lru_cache: ตัวช่วยจำผลลัพธ์ของฟังก์ชัน (cache) ไม่ให้โหลดโมเดลซ้ำ | input: - | output: ตัว decorator lru_cache
from functools import lru_cache
# นำเข้า Path: ใช้จัดการ path ของไฟล์แบบ object | input: - | output: คลาส Path
from pathlib import Path

# นำเข้าส่วนประกอบของ FastAPI: Depends (ฉีด dependency), FastAPI (ตัวแอป), File/Form (อ่านข้อมูลจากฟอร์ม), HTTPException (ส่ง error), UploadFile (ไฟล์ที่อัปโหลด) | input: - | output: คลาส/ฟังก์ชันดังกล่าว
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
# นำเข้า Image (เปิด/ประมวลผลรูป) และ UnidentifiedImageError (error เมื่อไฟล์ไม่ใช่รูป) จาก Pillow | input: - | output: คลาส/ข้อยกเว้นดังกล่าว
from PIL import Image, UnidentifiedImageError

# หาโฟลเดอร์รากของโปรเจกต์ (ขึ้นไป 1 ระดับจากโฟลเดอร์ backend) | input: path ของไฟล์นี้ (__file__) | output: ROOT = Path ของโฟลเดอร์รากโปรเจกต์
ROOT = Path(__file__).resolve().parents[1]
# กำหนด path ไฟล์โมเดลหลัก: ใช้ค่าจากตัวแปรสภาพแวดล้อม MODEL_PATH ถ้ามี ไม่งั้นใช้ models/patchcore_pill_512.pt | input: env MODEL_PATH (ไม่บังคับ) | output: MODEL_PATH = Path ของไฟล์โมเดล
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "patchcore_pill_512.pt"))
# One memory bank and threshold per tablet type, on the same backbone. Each type has its
# own multi-pill threshold and its own way of cutting pills out of a larger image.
# ตารางตั้งค่าของยาแต่ละชนิด | input: - | output: PILL_TYPES = dict {ชื่อชนิด: {model, scenes, pose}}
PILL_TYPES = {
    # ชนิด mvtec (เม็ดยารูปไข่ จากชุดข้อมูล MVTec)
    "mvtec": {
        # path ไฟล์โมเดลของชนิด mvtec
        "model": MODEL_PATH,
        # path ไฟล์ผลลัพธ์ที่ใช้อ่านค่า threshold ของรูปที่มีหลายเม็ด
        "scenes": ROOT / "results" / "multi_pill" / "multi_pill_results.json",
        # ชื่อค่าคงที่ท่าวางเม็ดยาใน inference.py ที่ใช้กับชนิดนี้
        "pose": "MVTEC_POSE",
    },
    # ชนิด pillqc (เม็ดยากลม จากชุดข้อมูล pillQC)
    "pillqc": {
        # path ไฟล์โมเดลของชนิด pillqc
        "model": ROOT / "models" / "patchcore_pillqc_256.pt",
        # path ไฟล์ผลลัพธ์ฉากหลายเม็ดของ pillqc
        "scenes": ROOT / "results" / "pillqc_model" / "pillqc_scene_results.json",
        # ชื่อค่าคงที่ท่าวางเม็ดยาของ pillqc
        "pose": "PILLQC_POSE",
    },
}
# ขนาดไฟล์อัปโหลดสูงสุด = 10 MB (หน่วยไบต์) | input: - | output: MAX_UPLOAD_BYTES = 10485760
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# สร้างตัวแอป FastAPI พร้อมตั้งชื่อ | input: title | output: app = ตัวแอปเว็บเซิร์ฟเวอร์
app = FastAPI(title="Pill Anomaly Inspection")


# จำผลลัพธ์ไว้ไม่เกินจำนวนชนิดยา เพื่อให้โหลดโมเดลแต่ละชนิดครั้งเดียว
@lru_cache(maxsize=len(PILL_TYPES))
# ฟังก์ชันโหลดตัวตรวจ (Inspector) | input: pill_type (str) ชื่อชนิดยา | output: วัตถุ Inspector ที่พร้อมใช้งาน
def load_inspector(pill_type):
    # นำเข้าโมดูล inference ตอนเรียกใช้ (ไม่นำเข้าตอนเริ่มโปรแกรม เพื่อให้เริ่มเร็วและทดสอบได้) | input: - | output: โมดูล inference
    import inference

    # ดึงค่าตั้งค่าของชนิดยานี้จากตาราง | input: pill_type | output: setup = dict {model, scenes, pose}
    setup = PILL_TYPES[pill_type]
    # ถ้าไม่พบไฟล์โมเดล
    if not setup["model"].exists():
        # ตอบ HTTP 503 (บริการยังไม่พร้อม) พร้อมชื่อไฟล์ที่หาย | input: ชื่อไฟล์ | output: ยกข้อยกเว้น HTTPException
        raise HTTPException(503, f"Model file not found: {setup['model'].name}")
    # ใช้ไฟล์ผลลัพธ์ฉากหลายเม็ดถ้ามีอยู่จริง ไม่งั้นใช้ None | input: setup["scenes"] | output: scenes = path หรือ None
    scenes = setup["scenes"] if setup["scenes"].exists() else None
    # สร้าง Inspector จากโมเดล, ผลลัพธ์ฉาก และท่าวางเม็ดยา แล้วคืนค่า | input: path โมเดล, path ฉาก, ค่าคงที่ pose | output: วัตถุ Inspector
    return inference.Inspector(setup["model"], scenes, pose=getattr(inference, setup["pose"]))


# ฟังก์ชัน dependency: เลือก Inspector ตามชนิดยาที่ผู้ใช้ส่งมาในฟอร์ม | input: pill_type (str) จากฟอร์ม ค่าเริ่มต้น "mvtec" | output: วัตถุ Inspector
def get_inspector(pill_type: str = Form("mvtec")):
    # ถ้าชื่อชนิดยาไม่อยู่ในตาราง
    if pill_type not in PILL_TYPES:
        # ตอบ HTTP 400 (คำขอไม่ถูกต้อง) | input: ชื่อชนิดที่ผิด | output: ยกข้อยกเว้น HTTPException
        raise HTTPException(400, f"Unknown pill type: {pill_type}")
    # โหลด (หรือดึงจาก cache) Inspector ของชนิดนี้แล้วคืนค่า | input: pill_type | output: วัตถุ Inspector
    return load_inspector(pill_type)


# ฟังก์ชันสรุปไฟล์ผลลัพธ์ .json ให้เหลือเฉพาะ field ที่หน้าเว็บใช้ | input: path (Path) ของไฟล์ json | output: dict {ชื่อการทดลอง: {config, overall, by_defect, error}} หรือ {} ถ้าไม่มีไฟล์
def summarize(path):
    # ถ้าไฟล์ไม่มีอยู่
    if not path.exists():
        # คืน dict ว่าง | input: - | output: {}
        return {}
    # อ่านไฟล์เป็นข้อความแล้วแปลง JSON เป็น dict | input: path | output: data = dict ผลลัพธ์ทั้งหมด
    data = json.loads(path.read_text())
    # คืนค่าที่สร้างจาก dict comprehension
    return {
        # แต่ละการทดลอง เก็บเฉพาะ key ที่อนุญาต 4 ตัว | input: run (dict ของการทดลอง) | output: dict ที่กรองแล้ว
        name: {key: value for key, value in run.items() if key in ("config", "overall", "by_defect", "error")}
        # วนอ่านทุกการทดลองใน data
        for name, run in data.items()
    }


# ลงทะเบียน endpoint GET /api/health
@app.get("/api/health")
# ฟังก์ชันตรวจสุขภาพระบบ | input: - | output: dict {status, model_available, pill_types}
def health():
    # คืนสถานะของเซิร์ฟเวอร์
    return {
        # บอกว่าเซิร์ฟเวอร์ทำงานปกติ
        "status": "ok",
        # บอกว่ามีไฟล์โมเดลหลักหรือไม่ (True/False)
        "model_available": MODEL_PATH.exists(),
        # บอกว่าไฟล์โมเดลของยาแต่ละชนิดมีอยู่หรือไม่ | output: dict {ชื่อชนิด: True/False}
        "pill_types": {name: setup["model"].exists() for name, setup in PILL_TYPES.items()},
    }


# ลงทะเบียน endpoint GET /api/results
@app.get("/api/results")
# ฟังก์ชันส่งผลการทดลองให้หน้าเว็บแสดง | input: - | output: dict {main, exploratory}
def results():
    # คืนผลการทดลองที่สรุปแล้ว
    return {
        # ผลหลักจาก phase3 | output: dict ที่สรุปแล้ว
        "main": summarize(ROOT / "results" / "phase3" / "phase3_results.json"),
        # ผลการทดลองเสริม
        "exploratory": {
            # รวมผล phase4 เข้ามา (กระจาย dict)
            **summarize(ROOT / "results" / "phase4" / "phase4_results.json"),
            # รวมผล phase5 เข้ามา (กระจาย dict)
            **summarize(ROOT / "results" / "phase5" / "phase5_results.json"),
        },
    }


# ลงทะเบียน endpoint POST /api/inspect
@app.post("/api/inspect")
# ฟังก์ชันตรวจเม็ดยาในรูปที่อัปโหลด (async เพราะอ่านไฟล์แบบไม่บล็อก) | input: file (UploadFile รูปภาพ), inspector (ได้จาก get_inspector) | output: dict ผลตรวจ {score, threshold, is_anomalous, image, heatmap, ...}
async def inspect(file: UploadFile = File(...), inspector=Depends(get_inspector)):
    # อ่านไบต์ของไฟล์ แต่อ่านเกินขีดจำกัดไป 1 ไบต์เพื่อรู้ว่าไฟล์ใหญ่เกิน | input: file | output: content = bytes
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    # ถ้าไฟล์ใหญ่เกิน 10 MB
    if len(content) > MAX_UPLOAD_BYTES:
        # ตอบ HTTP 413 (ไฟล์ใหญ่เกิน) | input: ข้อความ | output: ยกข้อยกเว้น HTTPException
        raise HTTPException(413, "Image must be 10 MB or smaller")
    # เริ่มบล็อกที่ลองเปิดรูป (อาจผิดพลาดถ้าไฟล์ไม่ใช่รูป)
    try:
        # เปิดรูปจากไบต์ที่ห่อด้วย BytesIO | input: content | output: image = วัตถุรูปของ Pillow (ยังไม่โหลดพิกเซล)
        image = Image.open(io.BytesIO(content))
        # บังคับโหลดพิกเซลทั้งหมดเพื่อตรวจว่าไฟล์เสียหรือไม่ | input: - | output: - (ยกข้อยกเว้นถ้าอ่านไม่ได้)
        image.load()
    # ดักข้อผิดพลาดเมื่อไฟล์ไม่ใช่รูปหรืออ่านไม่ได้
    except (UnidentifiedImageError, OSError) as error:
        # ตอบ HTTP 400 (อ่านรูปไม่ได้) | input: ข้อความ | output: ยกข้อยกเว้น HTTPException
        raise HTTPException(400, "File is not a readable image") from error
    # ส่งรูปให้ Inspector ตรวจแล้วคืนผลลัพธ์ | input: image (PIL.Image) | output: dict ผลตรวจ
    return inspector.inspect(image)
