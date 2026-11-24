"""Evaluate per-pill inspection on simulated multi-pill scenes.

Run `calibrate` first. It sets the threshold from validation pills only. Run `test` once after that.
"""
# นำเข้า argparse: อ่านอาร์กิวเมนต์บรรทัดคำสั่ง | input: - | output: โมดูล argparse
import argparse
# นำเข้า json: อ่าน/เขียนไฟล์ JSON | input: - | output: โมดูล json
import json
# นำเข้า os: อ่านตัวแปรสภาพแวดล้อม | input: - | output: โมดูล os
import os
# นำเข้า sys: แก้ sys.path | input: - | output: โมดูล sys
import sys
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า Image ของ Pillow: เปิดรูป | input: - | output: คลาส Image
from PIL import Image
# นำเข้า roc_auc_score: คำนวณ AUROC | input: - | output: ฟังก์ชัน roc_auc_score
from sklearn.metrics import roc_auc_score

# โฟลเดอร์รากของโปรเจกต์ | input: __file__ | output: ROOT = Path
ROOT = Path(__file__).resolve().parents[1]
# เพิ่ม backend และ src ไว้หน้าสุดของ sys.path เพื่อนำเข้าโมดูลของโปรเจกต์ | input: ROOT | output: sys.path ถูกแก้ไข
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "src")]

# นำเข้าฟังก์ชันเลือก threshold | input: - | output: pick_threshold
from ad_metrics import pick_threshold  # noqa: E402
# นำเข้าพื้นที่สีแปลกขั้นต่ำ | input: - | output: MIN_AREA
from colour_check import MIN_AREA  # noqa: E402
# นำเข้าคลาสตัวตรวจ | input: - | output: Inspector
from inference import Inspector  # noqa: E402
# นำเข้าจำนวนเม็ดต่อฉาก, ฟังก์ชันประกอบฉาก และตัดเม็ด | input: - | output: PILLS_PER_SCENE, compose_scene, cut_pill
from make_pill_scenes import PILLS_PER_SCENE, compose_scene, cut_pill  # noqa: E402
# นำเข้าฟังก์ชันหาเม็ดยาในรูป | input: - | output: find_pills
from segmentation import find_pills  # noqa: E402

# โฟลเดอร์ข้อมูลรูป MVTec pill | input: ROOT | output: DATA = Path
DATA = ROOT / "data" / "mvtec" / "pill"
# ไฟล์ผลลัพธ์ที่จะเขียน | input: ROOT | output: OUTPUT = Path
OUTPUT = ROOT / "results" / "multi_pill" / "multi_pill_results.json"
# ค่า IoU ขั้นต่ำที่นับว่าเม็ดที่หาเจอตรงกับเม็ดจริง | input: - | output: MIN_IOU = 0.5
MIN_IOU = 0.5
# จำนวนรอบสุ่มฉากตอน calibrate | input: - | output: CALIBRATION_ROUNDS = 5
CALIBRATION_ROUNDS = 5
# เมล็ดสุ่มที่ใช้ตอน test | input: - | output: TEST_SEEDS = (1, 2, 3)
TEST_SEEDS = (1, 2, 3)


# ฟังก์ชันโหลดรูปเป็นอาร์เรย์ RGB | input: path = path ไฟล์รูป | output: uint8 (H, W, 3)
def load(path):
    # เปิดรูป แปลงเป็น RGB และแปลงเป็นอาร์เรย์ | input: path | output: อาร์เรย์
    return np.asarray(Image.open(path).convert("RGB"))


# ฟังก์ชันให้คะแนนเม็ดยาในฉากจำลองหนึ่งฉาก | input: inspector = ตัวตรวจ, items = list ของ (ชื่อ, ชนิดความบกพร่อง, รูป RGB, mask เม็ด), rng = ตัวสุ่ม, touching = ให้เม็ดแตะกันหรือไม่ | output: tuple (list ของ dict ผลรายเม็ด, จำนวนเม็ดที่ segmentation หาเจอ)
def run_scene(inspector, items, rng, touching):
    """Score one scene. Each item is (name, defect, RGB image, pill mask)."""
    # ตัดเม็ดยาแต่ละเม็ดเป็น cut-out แบบสุ่มหมุน/ปรับขนาด | input: items, rng | output: cutouts = list RGBA
    cutouts = [cut_pill(image, mask, rng) for _, _, image, mask in items]
    # ประกอบฉาก และได้ mask ความจริง (ground truth) ของแต่ละเม็ด | input: cutouts, rng, touching | output: scene = รูป, truth = list ของ mask
    scene, truth = compose_scene(cutouts, rng, touching)
    # หาเม็ดยาในฉากด้วย segmentation | input: scene | output: found = list ของ mask
    found = find_pills(scene)
    # รวมทุกเม็ดที่หาเจอเป็น mask เดียว (None ถ้าไม่เจอเลย) | input: found | output: everything = bool (H, W) หรือ None
    everything = np.any(found, axis=0) if found else None
    # เตรียม list ผลรายเม็ด | input: - | output: rows = []
    rows = []
    # วนทีละเม็ดจริงคู่กับ mask ความจริง
    for (name, defect, _, _), mask in zip(items, truth):
        # คำนวณ IoU ระหว่าง mask จริงกับทุกเม็ดที่หาเจอ | input: mask, found | output: overlaps = list ของ float
        overlaps = [(mask & other).sum() / (mask | other).sum() for other in found]
        # ดัชนีของเม็ดที่ซ้อนทับมากสุด (-1 ถ้าไม่มีเม็ดที่หาเจอ) | input: overlaps | output: best = int
        best = int(np.argmax(overlaps)) if overlaps else -1
        # สร้างแถวผลเริ่มต้น (ยังไม่มีคะแนน) | input: name, defect, touching | output: row = dict
        row = {"name": name, "defect": defect, "touching": touching, "score": None, "colour_area": None}
        # ถ้ามีเม็ดที่หาเจอตรงกับเม็ดจริง (IoU >= 0.5)
        if overlaps and overlaps[best] >= MIN_IOU:
            # ให้คะแนนเม็ดที่หาเจอ โดยลบเม็ดอื่นออกจากรูป; ค่าที่ไม่ใช้เก็บด้วย _ | input: scene, found[best], everything & ~found[best] | output: score, colour_area
            score, _, _, colour_area, _ = inspector.score_pill(scene, found[best], everything & ~found[best])
            # บันทึกคะแนนและพื้นที่สีแปลกลงแถว | input: score, colour_area | output: row ถูกอัปเดต
            row["score"], row["colour_area"] = score, colour_area
        # เก็บแถวนี้ | input: row | output: rows ยาวขึ้น 1
        rows.append(row)
    # คืนแถวทั้งหมดและจำนวนเม็ดที่หาเจอ | input: rows, found | output: (rows, int)
    return rows, len(found)


# ฟังก์ชันรันครบทุกเม็ดหนึ่งรอบ โดยแบ่งเป็นฉากละ PILLS_PER_SCENE เม็ด | input: inspector, pills = list ของ (ชื่อ, ความบกพร่อง, รูป, mask), seed, touching | output: tuple (list ผลรายเม็ด, จำนวนเม็ดที่หาเจอรวม)
def run_pass(inspector, pills, seed, touching):
    # สร้างตัวสุ่มตาม seed | input: seed | output: rng
    rng = np.random.default_rng(seed)
    # สุ่มลำดับของเม็ดทั้งหมด | input: len(pills) | output: order = อาร์เรย์ดัชนี
    order = rng.permutation(len(pills))
    # เตรียมตัวเก็บผลรวมและจำนวนที่หาเจอรวม | input: - | output: rows = [], found = 0
    rows, found = [], 0
    # วนทีละฉาก (ก้าวละ PILLS_PER_SCENE เม็ด)
    for start in range(0, len(order), PILLS_PER_SCENE):
        # เลือกเม็ดของฉากนี้ตามลำดับที่สุ่ม | input: pills, order, start | output: chunk = list
        chunk = [pills[index] for index in order[start:start + PILLS_PER_SCENE]]
        # รันฉากนี้ | input: inspector, chunk, rng, touching | output: scene_rows, scene_found
        scene_rows, scene_found = run_scene(inspector, chunk, rng, touching)
        # เพิ่มแถวผลของฉากนี้ | input: scene_rows | output: rows ยาวขึ้น
        rows += scene_rows
        # บวกจำนวนเม็ดที่หาเจอ | input: scene_found | output: found เพิ่มขึ้น
        found += scene_found
    # คืนผลรวมทุกฉาก | input: rows, found | output: (rows, found)
    return rows, found


# ฟังก์ชันโหลดเม็ดยา (แต่ละรูปต้องมีเม็ดเดียว) | input: paths = list ของ (path รูป, ชนิดความบกพร่อง) | output: list ของ (ชื่อ "โฟลเดอร์/ไฟล์", ความบกพร่อง, รูป RGB, mask เม็ด)
def prepare(paths):
    """Load pills. Each source image must hold exactly one pill."""
    # เตรียม list ผลลัพธ์ | input: - | output: pills = []
    pills = []
    # วนทุกรูป
    for path, defect in paths:
        # โหลดรูป | input: path | output: image
        image = load(path)
        # หาเม็ดยาในรูป | input: image | output: masks = list
        masks = find_pills(image)
        # ถ้าจำนวนเม็ดไม่ใช่ 1
        if len(masks) != 1:
            # แจ้งข้อผิดพลาดพร้อมจำนวนที่พบ | input: path, len(masks) | output: ยกข้อยกเว้น ValueError
            raise ValueError(f"{path} gave {len(masks)} pills")
        # เก็บเม็ดนี้พร้อมชื่อ "โฟลเดอร์แม่/ชื่อไฟล์" | input: path, defect, image, masks[0] | output: pills ยาวขึ้น 1
        pills.append((f"{path.parent.name}/{path.name}", defect, image, masks[0]))
    # คืนรายการเม็ดทั้งหมด | input: pills | output: list
    return pills


# ฟังก์ชันตั้ง threshold จากเม็ดชุด validation เท่านั้น แล้วบันทึกลงไฟล์ | input: inspector = ตัวตรวจ | output: ไม่คืนค่า; เขียนไฟล์ OUTPUT และพิมพ์ผลออกหน้าจอ
def calibrate(inspector):
    # อ่านไฟล์แบ่งชุดข้อมูล | input: splits/pill_seed42.json | output: split = dict
    split = json.loads((ROOT / "splits" / "pill_seed42.json").read_text())
    # โหลดเม็ดปกติชุด validation | input: ชื่อไฟล์ใน split["validation"] | output: pills
    pills = prepare([(DATA / "train" / "good" / name, "good") for name in split["validation"]])
    # เตรียม list ผลรวม | input: - | output: rows = []
    rows = []
    # วนตามจำนวนรอบ (แต่ละรอบใช้ seed ต่างกัน)
    for seed in range(CALIBRATION_ROUNDS):
        # ลองทั้งแบบเม็ดแยกกันและแบบแตะกัน
        for touching in (False, True):
            # รันหนึ่งรอบ แล้วเอาเฉพาะแถวผล (ค่าแรก) | input: inspector, pills, seed, touching | output: rows ยาวขึ้น
            rows += run_pass(inspector, pills, seed, touching)[0]
    # คะแนนของเม็ดที่ segmentation หาเจอ | input: rows | output: scores = list ของ float
    scores = [row["score"] for row in rows if row["score"] is not None]
    # พื้นที่สีแปลกของเม็ดที่หาเจอ | input: rows | output: areas = list ของ int
    areas = [row["colour_area"] for row in rows if row["score"] is not None]
    # สรุปผล calibrate
    result = {
        # threshold ของคะแนน = ควอนไทล์ 99 ของคะแนนปกติ
        "threshold": float(pick_threshold(scores)),
        # threshold ของพื้นที่สี (ไม่ต่ำกว่า MIN_AREA)
        "colour_threshold": max(float(pick_threshold(areas)), MIN_AREA),
        # พื้นที่สีแปลกสูงสุดที่เจอในเม็ดปกติ
        "validation_colour_area_max": max(areas),
        # จำนวนเม็ดทั้งหมดที่ทดสอบ
        "validation_pills": len(rows),
        # จำนวนเม็ดที่ segmentation หาไม่เจอ
        "validation_missed_by_segmentation": len(rows) - len(scores),
        # เปอร์เซ็นไทล์ของคะแนน validation (0, 50, 95, 99, 100)
        "validation_score_percentiles": {
            str(q): float(np.percentile(scores, q)) for q in (0, 50, 95, 99, 100)
        },
    }
    # สร้างโฟลเดอร์ของไฟล์ผลลัพธ์ถ้ายังไม่มี | input: OUTPUT | output: โฟลเดอร์ใหม่
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # บันทึกผลเป็น JSON | input: result | output: ไฟล์ OUTPUT
    OUTPUT.write_text(json.dumps(result, indent=2))
    # พิมพ์ผลออกหน้าจอ | input: result | output: ข้อความ JSON
    print(json.dumps(result, indent=2))


# ฟังก์ชันสรุปเมตริกของแถวผลรายเม็ด | input: rows = list ของ dict ผลรายเม็ด, threshold = เกณฑ์คะแนน, colour_threshold = เกณฑ์พื้นที่สี (หรือ None) | output: dict {pills, missed_by_segmentation, true_positive, false_positive, false_negative, true_negative, precision, recall, f1, auroc (ถ้ามีทั้งสองคลาส)}
def summarize(rows, threshold, colour_threshold=None):
    """Without `colour_threshold` (pill types with no colour table), PatchCore decides alone."""
    # แถวของเม็ดที่บกพร่อง | input: rows | output: defective = list
    defective = [row for row in rows if row["defect"] != "good"]
    # แถวของเม็ดปกติ | input: rows | output: normal = list
    normal = [row for row in rows if row["defect"] == "good"]

    # ฟังก์ชันซ้อนในรวมสองการตรวจเป็นคะแนนเดียว | input: row | output: float (>1 แปลว่าผิดปกติ)
    def combined(row):
        """Each check divided by its own threshold, larger one kept."""
        # ถ้าไม่มีเกณฑ์สี
        if colour_threshold is None:
            # ใช้คะแนน PatchCore หารเกณฑ์อย่างเดียว | input: row["score"], threshold | output: float
            return row["score"] / threshold
        # ใช้ค่าที่มากกว่าระหว่างคะแนนสัมพัทธ์ของ PatchCore กับของสี | input: row, threshold, colour_threshold | output: float
        return max(row["score"] / threshold, row["colour_area"] / colour_threshold)

    # ฟังก์ชันซ้อนในบอกว่าเม็ดนี้ถูกแจ้งผิดปกติหรือไม่ | input: row | output: bool
    def flagged(row):
        # ต้องมีคะแนน (หาเจอ) และคะแนนรวมเกิน 1 | input: row | output: bool
        return row["score"] is not None and combined(row) > 1

    # จำนวนเม็ดบกพร่องที่ถูกจับได้ | input: defective | output: true_positive = int
    true_positive = sum(flagged(row) for row in defective)
    # จำนวนเม็ดปกติที่ถูกแจ้งผิด | input: normal | output: false_positive = int
    false_positive = sum(flagged(row) for row in normal)
    # แถวที่มีคะแนน (หาเจอ) | input: rows | output: scored = list
    scored = [row for row in rows if row["score"] is not None]
    # recall = TP / จำนวนเม็ดบกพร่อง (กันหารศูนย์) | input: true_positive, defective | output: float
    recall = true_positive / max(len(defective), 1)
    # precision = TP / (TP + FP) (กันหารศูนย์) | input: true_positive, false_positive | output: float
    precision = true_positive / max(true_positive + false_positive, 1)
    # รวมเมตริกเป็น dict
    summary = {
        # จำนวนเม็ดทั้งหมด
        "pills": len(rows),
        # จำนวนเม็ดที่ segmentation หาไม่เจอ
        "missed_by_segmentation": len(rows) - len(scored),
        # TP
        "true_positive": true_positive,
        # FP
        "false_positive": false_positive,
        # FN = เม็ดบกพร่องที่จับไม่ได้
        "false_negative": len(defective) - true_positive,
        # TN = เม็ดปกติที่ไม่ถูกแจ้ง
        "true_negative": len(normal) - false_positive,
        # precision
        "precision": precision,
        # recall
        "recall": recall,
        # F1 (กันหารศูนย์ด้วย 1e-9)
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }
    # ถ้ามีทั้งเม็ดปกติและบกพร่อง จึงคำนวณ AUROC ได้
    if normal and defective:
        # ป้ายกำกับของแถวที่มีคะแนน (True = บกพร่อง) | input: scored | output: labels = list ของ bool
        labels = [row["defect"] != "good" for row in scored]
        # AUROC จากคะแนนรวม | input: labels, combined scores | output: summary["auroc"] = float
        summary["auroc"] = roc_auc_score(labels, [combined(row) for row in scored])
    # คืนสรุป | input: summary | output: dict
    return summary


# ฟังก์ชันทดสอบด้วยเม็ดชุด test โดยใช้ threshold ที่ calibrate ไว้ แล้วบันทึกผล | input: inspector = ตัวตรวจ | output: ไม่คืนค่า; เขียนไฟล์ OUTPUT และพิมพ์สรุป
def test(inspector):
    # อ่านผล calibrate ที่บันทึกไว้ | input: OUTPUT | output: result = dict
    result = json.loads(OUTPUT.read_text())
    # รวม threshold คะแนนและ threshold สี | input: result | output: thresholds = tuple 2 ค่า
    thresholds = result["threshold"], result["colour_threshold"]
    # รายการเม็ดทดสอบ
    paths = [
        # คู่ (path รูป, ชนิดความบกพร่อง = ชื่อโฟลเดอร์แม่)
        (path, path.parent.name)
        # วนทุกไฟล์ png ในโฟลเดอร์ test (เรียงชื่อ)
        for path in sorted((DATA / "test").glob("*/*.png"))
    ]
    # โหลดเม็ดทั้งหมด | input: paths | output: pills
    pills = prepare(paths)
    # เตรียมตัวเก็บแถวผลรวม และจำนวนที่หาเจอของแต่ละรอบ | input: - | output: rows = [], scene_counts = {}
    rows, scene_counts = [], {}
    # วนทุก seed
    for seed in TEST_SEEDS:
        # ลองทั้งแบบแยกและแบบแตะกัน
        for touching in (False, True):
            # รันหนึ่งรอบ | input: inspector, pills, seed, touching | output: pass_rows, found
            pass_rows, found = run_pass(inspector, pills, seed, touching)
            # เพิ่มแถวผลของรอบนี้ | input: pass_rows | output: rows ยาวขึ้น
            rows += pass_rows
            # บันทึกจำนวนเม็ดจริงกับจำนวนที่หาเจอ | input: pass_rows, found | output: scene_counts ถูกเพิ่ม
            scene_counts[f"seed{seed}_touching{touching}"] = {"truth": len(pass_rows), "found": found}
            # พิมพ์สรุปของรอบนี้ทันที | input: seed, touching, summary | output: ข้อความบนหน้าจอ
            print(seed, touching, summarize(pass_rows, *thresholds), flush=True)
    # เก็บจำนวนเม็ดที่หาเจอของทุกรอบ | input: scene_counts | output: result["segmentation_counts"]
    result["segmentation_counts"] = scene_counts
    # เก็บคะแนนรายเม็ดทั้งหมด | input: rows | output: result["pill_scores"]
    result["pill_scores"] = rows
    # สรุปรวมทุกรอบ | input: rows, thresholds | output: result["overall"]
    result["overall"] = summarize(rows, *thresholds)
    # สรุปเฉพาะรอบที่เม็ดแยกกัน | input: rows ที่ touching เป็น False | output: result["spaced"]
    result["spaced"] = summarize([row for row in rows if not row["touching"]], *thresholds)
    # สรุปเฉพาะรอบที่เม็ดแตะกัน | input: rows ที่ touching เป็น True | output: result["touching"]
    result["touching"] = summarize([row for row in rows if row["touching"]], *thresholds)
    # สรุปแยกตามชนิดความบกพร่อง
    result["by_defect"] = {
        # สรุปของแต่ละชนิด | input: rows ของชนิดนั้น | output: dict
        defect: summarize([row for row in rows if row["defect"] == defect], *thresholds)
        # วนทุกชนิดที่มี (เรียงตามตัวอักษร)
        for defect in sorted({row["defect"] for row in rows})
    }
    # บันทึกผลทั้งหมดลงไฟล์ | input: result | output: ไฟล์ OUTPUT
    OUTPUT.write_text(json.dumps(result, indent=2))
    # พิมพ์สรุปรวม แยกรอบ และแตะกัน | input: result | output: ข้อความ JSON
    print(json.dumps({key: result[key] for key in ("overall", "spaced", "touching")}, indent=2))


# รันเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # สร้างตัวอ่านอาร์กิวเมนต์ | input: - | output: parser
    parser = argparse.ArgumentParser()
    # อาร์กิวเมนต์ stage ต้องเป็น calibrate หรือ test | input: - | output: args.stage
    parser.add_argument("stage", choices=["calibrate", "test"])
    # อ่านอาร์กิวเมนต์และดึงค่า stage | input: sys.argv | output: stage = str
    stage = parser.parse_args().stage
    # โหลดตัวตรวจจากไฟล์โมเดล (ใช้ตัวแปรสภาพแวดล้อม MODEL_PATH ถ้ามี) | input: env MODEL_PATH | output: model = Inspector
    model = Inspector(os.environ.get("MODEL_PATH", ROOT / "models" / "patchcore_pill_512.pt"))
    # เลือกฟังก์ชันตาม stage แล้วเรียกกับตัวตรวจ | input: model | output: ผลของ calibrate หรือ test
    (calibrate if stage == "calibrate" else test)(model)
