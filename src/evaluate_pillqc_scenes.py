"""Set and test the multi-pill threshold for pillQC tablets, as was done for MVTec.

pillQC has no multi-pill photos, so scenes are simulated: each crop's pill is cut out,
resized, and pasted on a dark background, 8 pills per scene. Pills are not turned: these
round pills show a side band and shadow from one light, which a turn would move. `calibrate` sets
the threshold from the 54 validation pills only (5 rounds, spaced and touching). Run
`test` once after that, on the 230 test pills (3 seeds, spaced and touching).

    backend/.venv/bin/python src/evaluate_pillqc_scenes.py calibrate
    backend/.venv/bin/python src/evaluate_pillqc_scenes.py test
"""
# นำเข้า argparse: อ่านอาร์กิวเมนต์บรรทัดคำสั่ง | input: - | output: โมดูล argparse
import argparse
# นำเข้า json: อ่าน/เขียนไฟล์ JSON | input: - | output: โมดูล json
import json
# นำเข้า sys: แก้ sys.path | input: - | output: โมดูล sys
import sys
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า Image ของ Pillow: เปิดรูป | input: - | output: คลาส Image
from PIL import Image

# โฟลเดอร์รากของโปรเจกต์ | input: __file__ | output: ROOT = Path
ROOT = Path(__file__).resolve().parents[1]
# เพิ่ม backend และ src ไว้หน้าสุดของ sys.path | input: ROOT | output: sys.path ถูกแก้ไข
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "src")]

# นำเข้าฟังก์ชันเลือก threshold | input: - | output: pick_threshold
from ad_metrics import pick_threshold  # noqa: E402
# นำเข้าค่าคงที่และฟังก์ชันประเมินแบบหลายเม็ดที่ใช้ร่วมกับ MVTec | input: - | output: CALIBRATION_ROUNDS, TEST_SEEDS, run_pass, summarize
from evaluate_multi_pill import CALIBRATION_ROUNDS, TEST_SEEDS, run_pass, summarize  # noqa: E402
# นำเข้าท่าวางของ pillQC และคลาสตัวตรวจ | input: - | output: PILLQC_POSE, Inspector
from inference import PILLQC_POSE, Inspector  # noqa: E402
# นำเข้าฟังก์ชันหาเม็ดชมพูและสร้างพื้นที่วงกลม | input: - | output: pink_pill_mask, round_pill_region
from segmentation import pink_pill_mask, round_pill_region  # noqa: E402
# นำเข้าโมดูลสร้างฉากเพื่อปรับค่าคงที่ของมัน | input: - | output: โมดูล make_pill_scenes
import make_pill_scenes  # noqa: E402

# Round pills take more room than MVTec's ovals, so 8 of them need a larger floor.
# ขยายขนาดฉากเป็น 1300x1800 เพราะเม็ดกลมกินที่มากกว่าเม็ดรูปไข่ | input: - | output: make_pill_scenes.SCENE_SIZE ถูกแก้
make_pill_scenes.SCENE_SIZE = (1300, 1800)
# In a real photo every pill has the same light, so the side band and shadow keep their direction.
# ไม่หมุนเม็ด (ช่วงมุม 0-0) เพื่อให้ทิศของแสงและเงาคงเดิม | input: - | output: make_pill_scenes.TURN_RANGE ถูกแก้
make_pill_scenes.TURN_RANGE = (0, 0)


# ฟังก์ชันหา mask เม็ดชมพูโดยตัดสิ่งสกปรกบนพื้นที่ติดเม็ดออก | input: image = RGB (H, W, 3) | output: bool (H, W)
def pill_mask(image):
    """The pink pill without floor dirt that touches it, which would be pasted along.

    The circle also cuts off chips on the rim, so scenes understate chip recall. Cutting at
    the pink outline instead pasted floor dirt with normal validation pills and raised the
    threshold from 48.65 to 56.23.
    """
    # หา mask เม็ดชมพู (อาจรวมสิ่งสกปรกที่ติดกัน) | input: image | output: mask = bool (H, W)
    mask = pink_pill_mask(image)
    # ตัดให้เหลือเฉพาะส่วนที่อยู่ในวงกลมครอบเม็ด | input: mask | output: bool (H, W)
    return mask & round_pill_region(mask)

# โฟลเดอร์รูปของ pillQC | input: ROOT | output: IMAGES = Path
IMAGES = ROOT / "data" / "pillqc" / "images"
# ไฟล์แบ่งชุดข้อมูล | input: ROOT | output: SPLIT = Path
SPLIT = ROOT / "splits" / "pillqc_seed42.json"
# ไฟล์ผลลัพธ์ที่จะเขียน | input: ROOT | output: OUTPUT = Path
OUTPUT = ROOT / "results" / "pillqc_model" / "pillqc_scene_results.json"



# ฟังก์ชันโหลดเม็ดยาจากรายการ (โฟลเดอร์, ชื่อไฟล์) | input: entries = list ของ (ชื่อโฟลเดอร์คลาส, ชื่อไฟล์) | output: list ของ (ชื่อ "โฟลเดอร์/ไฟล์", ความบกพร่อง ("good" หรือชื่อโฟลเดอร์), รูป RGB, mask เม็ด)
def prepare(entries):
    """entries: (class folder, file name). Normal pills are labelled "good" for `summarize`."""
    # เตรียม list ผลลัพธ์ | input: - | output: pills = []
    pills = []
    # วนทุกรายการ
    for folder, name in entries:
        # เปิดรูปเป็น RGB อาร์เรย์ | input: IMAGES, folder, name | output: image
        image = np.asarray(Image.open(IMAGES / folder / name).convert("RGB"))
        # เก็บเม็ด: ชื่อ, ชนิดความบกพร่อง (normal -> "good"), รูป, mask | input: folder, name, image | output: pills ยาวขึ้น 1
        pills.append((f"{folder}/{name}", "good" if folder == "normal" else folder, image, pill_mask(image)))
    # คืนรายการเม็ด | input: pills | output: list
    return pills


# ฟังก์ชันตั้ง threshold จากเม็ดชุด validation แล้วบันทึกลงไฟล์ | input: inspector = ตัวตรวจ | output: ไม่คืนค่า; เขียนไฟล์ OUTPUT และพิมพ์ผล
def calibrate(inspector):
    # อ่านไฟล์แบ่งชุดข้อมูล | input: SPLIT | output: split = dict
    split = json.loads(SPLIT.read_text())
    # โหลดเม็ดปกติชุด validation | input: split["validation"] | output: pills
    pills = prepare([("normal", name) for name in split["validation"]])
    # เตรียม list ผลรวม | input: - | output: rows = []
    rows = []
    # วนตามจำนวนรอบ (แต่ละรอบ seed ต่างกัน)
    for seed in range(CALIBRATION_ROUNDS):
        # ลองทั้งเม็ดแยกกันและแตะกัน
        for touching in (False, True):
            # รันหนึ่งรอบ แล้วเอาแถวผล (ค่าแรก) | input: inspector, pills, seed, touching | output: rows ยาวขึ้น
            rows += run_pass(inspector, pills, seed, touching)[0]
    # คะแนนของเม็ดที่หาเจอ | input: rows | output: scores = list
    scores = [row["score"] for row in rows if row["score"] is not None]
    # สรุปผล calibrate
    result = {
        # threshold = ควอนไทล์ 99 ของคะแนนเม็ดปกติ
        "threshold": float(pick_threshold(scores)),
        # จำนวนเม็ดที่ทดสอบ
        "validation_pills": len(rows),
        # จำนวนเม็ดที่ segmentation หาไม่เจอ
        "validation_missed_by_segmentation": len(rows) - len(scores),
        # เปอร์เซ็นไทล์ของคะแนน
        "validation_score_percentiles": {str(q): float(np.percentile(scores, q)) for q in (0, 50, 95, 99, 100)},
    }
    # บันทึกผลเป็น JSON | input: result | output: ไฟล์ OUTPUT
    OUTPUT.write_text(json.dumps(result, indent=2))
    # พิมพ์ผลออกหน้าจอ | input: result | output: ข้อความ JSON
    print(json.dumps(result, indent=2))


# ฟังก์ชันทดสอบด้วยเม็ดชุด test โดยใช้ threshold ที่ calibrate ไว้ | input: inspector = ตัวตรวจ | output: ไม่คืนค่า; เขียนไฟล์ OUTPUT และพิมพ์สรุป
def test(inspector):
    # อ่านผล calibrate | input: OUTPUT | output: result = dict
    result = json.loads(OUTPUT.read_text())
    # threshold ที่ calibrate ไว้ | input: result | output: threshold = float
    threshold = result["threshold"]
    # อ่านไฟล์แบ่งชุดข้อมูล | input: SPLIT | output: split = dict
    split = json.loads(SPLIT.read_text())
    # รายการเม็ดปกติชุด test | input: split["test"] | output: entries = list
    entries = [("normal", name) for name in split["test"]]
    # เพิ่มเม็ดบกพร่อง (chip, dirt) ทุกไฟล์ jpg | input: โฟลเดอร์ chip, dirt | output: entries ยาวขึ้น
    entries += [(folder, path.name) for folder in ("chip", "dirt") for path in sorted((IMAGES / folder).glob("*.jpg"))]
    # โหลดเม็ดทั้งหมด | input: entries | output: pills
    pills = prepare(entries)
    # เตรียมตัวเก็บแถวผลรวม และจำนวนที่หาเจอ | input: - | output: rows = [], counts = {}
    rows, counts = [], {}
    # วนทุก seed
    for seed in TEST_SEEDS:
        # ลองทั้งแยกกันและแตะกัน
        for touching in (False, True):
            # รันหนึ่งรอบ | input: inspector, pills, seed, touching | output: pass_rows, found
            pass_rows, found = run_pass(inspector, pills, seed, touching)
            # เพิ่มแถวผลของรอบนี้ | input: pass_rows | output: rows ยาวขึ้น
            rows += pass_rows
            # บันทึกจำนวนเม็ดจริงและจำนวนที่หาเจอ | input: pass_rows, found | output: counts ถูกเพิ่ม
            counts[f"seed{seed}_touching{touching}"] = {"truth": len(pass_rows), "found": found}
            # พิมพ์สรุปของรอบนี้ | input: seed, touching, summary | output: ข้อความบนหน้าจอ
            print(seed, touching, summarize(pass_rows, threshold), flush=True)
    # เก็บจำนวนเม็ดที่หาเจอของทุกรอบ | input: counts | output: result["segmentation_counts"]
    result["segmentation_counts"] = counts
    # เก็บคะแนนรายเม็ด | input: rows | output: result["pill_scores"]
    result["pill_scores"] = rows
    # สรุปรวมทุกรอบ | input: rows, threshold | output: result["overall"]
    result["overall"] = summarize(rows, threshold)
    # สรุปเฉพาะรอบที่เม็ดแยกกัน | input: rows ที่ touching เป็น False | output: result["spaced"]
    result["spaced"] = summarize([row for row in rows if not row["touching"]], threshold)
    # สรุปเฉพาะรอบที่เม็ดแตะกัน | input: rows ที่ touching เป็น True | output: result["touching"]
    result["touching"] = summarize([row for row in rows if row["touching"]], threshold)
    # สรุปแยกตามชนิดความบกพร่อง
    result["by_defect"] = {
        # สรุปของแต่ละชนิด | input: rows ของชนิดนั้น | output: dict
        defect: summarize([row for row in rows if row["defect"] == defect], threshold)
        # วนทุกชนิดที่มี (เรียงตามตัวอักษร)
        for defect in sorted({row["defect"] for row in rows})
    }
    # บันทึกผลทั้งหมดลงไฟล์ | input: result | output: ไฟล์ OUTPUT
    OUTPUT.write_text(json.dumps(result, indent=2))
    # พิมพ์สรุปรวม แยกรอบ แตะกัน และแยกตามความบกพร่อง | input: result | output: ข้อความ JSON
    print(json.dumps({key: result[key] for key in ("overall", "spaced", "touching", "by_defect")}, indent=2))


# รันเมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # สร้างตัวอ่านอาร์กิวเมนต์ | input: - | output: parser
    parser = argparse.ArgumentParser()
    # อาร์กิวเมนต์ stage ต้องเป็น calibrate หรือ test | input: - | output: args.stage
    parser.add_argument("stage", choices=["calibrate", "test"])
    # อ่านอาร์กิวเมนต์และดึงค่า stage | input: sys.argv | output: stage = str
    stage = parser.parse_args().stage
    # โหลดตัวตรวจ pillQC พร้อมท่าวางของ pillQC | input: ไฟล์โมเดล pillqc, PILLQC_POSE | output: model = Inspector
    model = Inspector(ROOT / "models" / "patchcore_pillqc_256.pt", pose=PILLQC_POSE)
    # เลือกฟังก์ชันตาม stage แล้วเรียกกับตัวตรวจ | input: model | output: ผลของ calibrate หรือ test
    (calibrate if stage == "calibrate" else test)(model)
