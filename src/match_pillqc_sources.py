"""Find the source photo of every pillQC normal crop, then split the crops by photo.

The 149 normal crops were cut from 6 photos. A random split would put crops of one
photo in the pool, validation, and test sets at once, so the split is made by photo.

    python src/match_pillqc_sources.py --root data/pillqc --out splits/pillqc_seed42.json
"""
# นำเข้า argparse: อ่านอาร์กิวเมนต์บรรทัดคำสั่ง | input: - | output: โมดูล argparse
import argparse
# นำเข้า json: เขียนไฟล์ JSON | input: - | output: โมดูล json
import json
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า OpenCV: อ่านรูปและทำ template matching | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np

# นำเข้าฟังก์ชันแบ่งชุดข้อมูลตามรูปต้นฉบับ | input: - | output: make_pillqc_split
from prepare_dataset import make_pillqc_split

# จำนวนรูปที่คาดว่าจะมีในแต่ละโฟลเดอร์ | input: - | output: EXPECTED = dict
EXPECTED = {"normal": 149, "chip": 43, "dirt": 138}
# คะแนนจับคู่ขั้นต่ำ (ครอปที่จับคู่กับรูปต้นฉบับของมันเองได้ 0.98 ขึ้นไป) | input: - | output: MIN_SCORE = 0.95
MIN_SCORE = 0.95  # A crop matched to its own photo scores 0.98 or more.
# อัตราย่อรูปเหลือครึ่งตอนจับคู่เพื่อความเร็ว | input: - | output: SHRINK = 0.5
SHRINK = 0.5  # Match at half size for speed.


# ฟังก์ชันหาศูนย์กลางและเส้นผ่านศูนย์กลางของเม็ดชมพูแต่ละเม็ดบนพื้นเขียวน้ำเงิน | input: image = รูป (H, W, 3) ลำดับสี BGR จาก cv2.imread, min_area/max_area = ช่วงพื้นที่เป็นพิกเซลที่ยอมรับ | output: list ของ (x, y, เส้นผ่านศูนย์กลาง)
def find_pills(image, min_area, max_area):
    """Centre and diameter of each pink pill on the teal background."""
    # แยกช่องสี (ลำดับ blue, green, red) เป็น int16 | input: image | output: blue, green, red
    blue, green, red = (image[..., i].astype(np.int16) for i in range(3))
    # พิกเซลที่แดงมาก (>150) และแดงเกินเขียวอย่างน้อย 10 ถือว่าเป็นสีชมพู | input: red, green | output: mask = uint8 (H, W)
    mask = ((red > 150) & (red > green + 10)).astype(np.uint8)
    # หากลุ่มพิกเซลที่ติดกัน พร้อมสถิติ (พื้นที่) และจุดศูนย์กลาง | input: mask | output: count, _, stats, centres (ค่าที่สองไม่ใช้)
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    # คืนรายการเม็ดที่ผ่านเกณฑ์
    return [
        # (x กลาง, y กลาง, เส้นผ่านศูนย์กลางที่คำนวณจากพื้นที่วงกลม) | input: centres[i], stats[i] | output: tuple
        (int(centres[i][0]), int(centres[i][1]), 2 * float(np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi)))
        # วนทุกกลุ่ม ข้ามกลุ่ม 0 (พื้นหลัง)
        for i in range(1, count)
        # เก็บเฉพาะกลุ่มที่พื้นที่อยู่ในช่วง
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]


# ฟังก์ชันหาคะแนนจับคู่ที่ดีที่สุดของครอปกับเม็ดใดเม็ดหนึ่งในรูปต้นฉบับ | input: crop = รูปครอป, photo = รูปต้นฉบับที่เติมขอบแล้ว, pills = list ของเม็ดในรูปต้นฉบับ, pad = ขนาดขอบที่เติม | output: tuple (คะแนนที่ดีที่สุด float, ดัชนีเม็ดนั้น int; (0.0, -1) ถ้าไม่เจอ)
def best_match(crop, photo, pills, pad):
    """Best template score of the crop around any pill of one photo, and that pill's index."""
    # เส้นผ่านศูนย์กลางเม็ดที่ใหญ่สุดในครอป | input: crop | output: diameter = float
    diameter = max(pill[2] for pill in find_pills(crop, 2000, crop.size))
    # ย่อครอปเป็นแม่แบบ (template) สำหรับจับคู่ | input: crop, SHRINK | output: template
    template = cv2.resize(crop, None, fx=SHRINK, fy=SHRINK, interpolation=cv2.INTER_AREA)
    # คะแนนดีที่สุดเริ่มต้น | input: - | output: best = (0.0, -1)
    best = (0.0, -1)
    # วนทุกเม็ดในรูปต้นฉบับ
    for index, (x, y, pill_diameter) in enumerate(pills):
        # ปรับสเกลรูปต้นฉบับให้เท่าครอป เพราะครอปถูกย่อขยายมา | input: diameter, pill_diameter | output: scale = float
        scale = diameter / pill_diameter  # Crops were resized, so bring the photo to the crop's scale.
        # ครึ่งหนึ่งของความกว้างหน้าต่างค้นหา (ไม่เกิน pad) | input: crop.shape, scale, pad | output: half = int
        half = min(int(0.9 * crop.shape[0] / scale), pad)
        # ตัดหน้าต่างรอบเม็ดนี้จากรูปที่เติมขอบ | input: photo, x, y, pad, half | output: window
        window = photo[y + pad - half:y + pad + half, x + pad - half:x + pad + half]
        # ปรับหน้าต่างให้สเกลตรงกับ template | input: window, scale, SHRINK | output: window ที่ปรับขนาดแล้ว
        window = cv2.resize(window, None, fx=scale * SHRINK, fy=scale * SHRINK, interpolation=cv2.INTER_AREA)
        # ถ้าหน้าต่างเล็กกว่า template จับคู่ไม่ได้
        if min(window.shape[:2]) < template.shape[0]:
            # ข้ามเม็ดนี้ | input: - | output: -
            continue
        # จับคู่ template และเอาคะแนนสูงสุด | input: window, template | output: score = float
        score = float(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED).max())
        # เก็บคู่ (คะแนน, ดัชนี) ที่ดีที่สุด | input: best, (score, index) | output: best
        best = max(best, (score, index))
    # คืนผลที่ดีที่สุด | input: best | output: (score, index)
    return best


# ฟังก์ชันหาว่าแต่ละครอปปกติมาจากรูปต้นฉบับใบไหน | input: root = โฟลเดอร์ pillQC, pad = ขนาดขอบที่เติมให้รูปต้นฉบับ | output: dict {ชื่อรูปต้นฉบับ: [ชื่อไฟล์ครอป]}
def match_sources(root, pad=600):
    """Return {photo name: [crop names]} for the normal crops."""
    # แปลง root เป็น Path | input: root | output: root = Path
    root = Path(root)
    # ตรวจจำนวนรูปในแต่ละโฟลเดอร์
    for name, expected in EXPECTED.items():
        # นับไฟล์ jpg ในโฟลเดอร์ | input: root, name | output: found = int
        found = len(list((root / "images" / name).glob("*.jpg")))
        # ถ้าจำนวนไม่ตรงที่คาดไว้
        if found != expected:
            # หยุดโปรแกรมพร้อมข้อความ | input: name, expected, found | output: SystemExit
            raise SystemExit(f"{name}: expected {expected} images, found {found}")

    # เตรียม dict เก็บรูปต้นฉบับ | input: - | output: photos = {}
    photos = {}
    # วนทุกรูปต้นฉบับที่ชื่อขึ้นต้นด้วย normal (เรียงชื่อ)
    for path in sorted((root / "sourceImages").glob("normal*.jpg")):
        # อ่านรูปต้นฉบับ (ลำดับสี BGR) | input: path | output: image
        image = cv2.imread(str(path))
        # เติมขอบสะท้อน (reflect) รอบรูปเพื่อให้ตัดหน้าต่างริมขอบได้ | input: image, pad | output: padded
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        # เก็บรูปที่เติมขอบ และตำแหน่งเม็ด (พื้นที่ 4000-60000) ของรูปนี้ | input: padded, find_pills(...) | output: photos[ชื่อรูป]
        photos[path.stem] = (padded, find_pills(image, 4000, 60000))

    # เตรียมกลุ่มครอปตามรูปต้นฉบับ และเซตของ (รูป, เม็ด) ที่ถูกใช้แล้ว | input: photos | output: groups, used
    groups, used = {name: [] for name in photos}, set()
    # วนทุกครอปปกติ (เรียงชื่อ)
    for path in sorted((root / "images" / "normal").glob("*.jpg")):
        # อ่านครอป | input: path | output: crop
        crop = cv2.imread(str(path))
        # จับคู่กับทุกรูปต้นฉบับ แล้วเลือกรูปที่ได้คะแนนสูงสุด | input: crop, photos | output: score, pill, name
        (score, pill), name = max((best_match(crop, photo, pills, pad), name) for name, (photo, pills) in photos.items())
        # Every crop must be a different pill of its photo, or the matching is not trustworthy.
        # ถ้าคะแนนต่ำเกินไป หรือเม็ดนี้ถูกครอปอื่นใช้ไปแล้ว ถือว่าจับคู่ไม่น่าเชื่อถือ
        if score < MIN_SCORE or (name, pill) in used:
            # หยุดโปรแกรมพร้อมรายละเอียด | input: path.name, score, name, pill | output: SystemExit
            raise SystemExit(f"{path.name}: no trustworthy source photo (score {score:.3f}, {name} pill {pill})")
        # จดว่าเม็ดนี้ของรูปนี้ถูกใช้แล้ว | input: name, pill | output: used ถูกเพิ่ม
        used.add((name, pill))
        # เพิ่มชื่อครอปในกลุ่มของรูปต้นฉบับนั้น | input: path.name | output: groups[name] ยาวขึ้น
        groups[name].append(path.name)
    # คืนกลุ่มครอป | input: groups | output: dict
    return groups


# ฟังก์ชันหลัก: จับคู่ครอปกับรูปต้นฉบับ แล้วแบ่งชุดข้อมูลตามรูป | input: อาร์กิวเมนต์บรรทัดคำสั่ง --root (บังคับ), --out, --seed | output: ไฟล์ JSON การแบ่งชุดข้อมูล และสรุปที่พิมพ์ออกหน้าจอ
def main():
    # สร้างตัวอ่านอาร์กิวเมนต์ โดยใช้ docstring ของไฟล์เป็นคำอธิบาย | input: __doc__ | output: parser
    parser = argparse.ArgumentParser(description=__doc__)
    # อาร์กิวเมนต์ --root: โฟลเดอร์ pillQC (บังคับ) | input: - | output: args.root
    parser.add_argument("--root", required=True, help="pillQC repository folder")
    # อาร์กิวเมนต์ --out: path ไฟล์ผลลัพธ์ (ค่าเริ่มต้น splits/pillqc_seed42.json) | input: - | output: args.out
    parser.add_argument("--out", default="splits/pillqc_seed42.json")
    # อาร์กิวเมนต์ --seed: เมล็ดสุ่มสำหรับแบ่งชุด (ค่าเริ่มต้น 42) | input: - | output: args.seed
    parser.add_argument("--seed", type=int, default=42)
    # อ่านอาร์กิวเมนต์จากบรรทัดคำสั่ง | input: sys.argv | output: args
    args = parser.parse_args()

    # จับคู่ครอปกับรูปต้นฉบับ | input: args.root | output: groups = dict
    groups = match_sources(args.root)
    # แบ่งชุด pool/validation/test ตามรูปต้นฉบับ | input: groups, args.seed | output: split = dict
    split = make_pillqc_split(groups, args.seed)
    # แนบข้อมูลว่าครอปไหนมาจากรูปไหน | input: groups | output: split["sources"]
    split["sources"] = groups
    # บันทึกการแบ่งชุดเป็นไฟล์ JSON ลงท้ายด้วยบรรทัดใหม่ | input: split, args.out | output: ไฟล์ JSON
    Path(args.out).write_text(json.dumps(split, indent=2) + "\n")
    # วนแต่ละรูปต้นฉบับ
    for name, crops in groups.items():
        # พิมพ์ชื่อรูป จำนวนครอป ครอปแรกและครอปสุดท้าย | input: name, crops | output: ข้อความบนหน้าจอ
        print(name, len(crops), crops[0], crops[-1])
    # พิมพ์สรุปว่าแต่ละชุดใช้รูปต้นฉบับใดและมีกี่ครอป | input: split | output: dict บนหน้าจอ
    print({part: (split["photos"][part], len(split[part])) for part in ("test", "validation", "pool")})


# รัน main เมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เรียกฟังก์ชันหลัก | input: - | output: -
    main()
