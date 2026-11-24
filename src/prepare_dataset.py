"""Validate the MVTec AD Pill folder and write a reproducible train/validation split.

Uses only the standard library so it runs anywhere, including Colab.

    python src/prepare_dataset.py --root /path/to/mvtec --out splits/pill_seed42.json
"""
# นำเข้า argparse: อ่านอาร์กิวเมนต์บรรทัดคำสั่ง | input: - | output: โมดูล argparse
import argparse
# นำเข้า json: เขียนไฟล์ JSON | input: - | output: โมดูล json
import json
# นำเข้า random: สับลำดับรายชื่อไฟล์แบบทำซ้ำได้ | input: - | output: โมดูล random
import random
# นำเข้า sys: เขียน error ออก stderr และคืน exit code | input: - | output: โมดูล sys
import sys
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# จำนวนรูปปกติในโฟลเดอร์ train ที่คาดไว้ | input: - | output: EXPECTED_TRAIN_GOOD = 267
EXPECTED_TRAIN_GOOD = 267
# จำนวนรูปปกติในโฟลเดอร์ test ที่คาดไว้ | input: - | output: EXPECTED_TEST_GOOD = 26
EXPECTED_TEST_GOOD = 26
# จำนวนรูปบกพร่องในโฟลเดอร์ test ที่คาดไว้ | input: - | output: EXPECTED_TEST_DEFECT = 141
EXPECTED_TEST_DEFECT = 141
# ชื่อโฟลเดอร์ชนิดความบกพร่องที่คาดไว้ (set ของข้อความ) | input: - | output: EXPECTED_DEFECTS = set
EXPECTED_DEFECTS = {
    # สี
    "color",
    # หลายชนิดรวมกัน
    "combined",
    # สิ่งปนเปื้อน
    "contamination",
    # รอยแตก
    "crack",
    # ตัวพิมพ์บนเม็ดยาผิด
    "faulty_imprint",
    # ชนิดเม็ดยาผิด
    "pill_type",
    # รอยขีดข่วน
    "scratch",
}


# ข้อยกเว้นที่ใช้เมื่อโฟลเดอร์ข้อมูลไม่ตรงโครงสร้างที่คาด
class DatasetError(Exception):
    """The dataset folder does not match the expected structure."""


# ฟังก์ชันหาโฟลเดอร์หมวดหมู่ (เช่น pill) | input: root = โฟลเดอร์ที่เป็นหมวดหมู่เองหรือโฟลเดอร์แม่, category = ชื่อหมวด | output: Path ของโฟลเดอร์หมวดหมู่ (ยก DatasetError ถ้าไม่พบ)
def find_category_root(root, category="pill"):
    """Return the category folder, whether root is that folder or a parent of it."""
    # แปลง root เป็น Path | input: root | output: root = Path
    root = Path(root)
    # ถ้า root มีทั้ง train/ และ test/ แปลว่าเป็นโฟลเดอร์หมวดหมู่อยู่แล้ว
    if (root / "train").is_dir() and (root / "test").is_dir():
        # คืน root เลย | input: root | output: Path
        return root
    # ไม่ใช่ ก็ค้นหาโฟลเดอร์ชื่อ category ลึกลงไป (เรียงชื่อ)
    for candidate in sorted(root.rglob(category)):
        # ถ้าผู้สมัครมีทั้ง train/ และ test/
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            # คืนผู้สมัครนั้น | input: candidate | output: Path
            return candidate
    # หาไม่พบเลย แจ้งข้อผิดพลาด | input: category, root | output: ยกข้อยกเว้น DatasetError
    raise DatasetError(f"No '{category}' folder with train/ and test/ under {root}")


# ฟังก์ชันตัวช่วยรวมรายชื่อไฟล์ .png ในโฟลเดอร์ | input: folder = Path ของโฟลเดอร์ | output: list ของชื่อไฟล์ .png เรียงตามตัวอักษร
def _pngs(folder):
    # หาไฟล์ .png แล้วเก็บเฉพาะชื่อ เรียงลำดับ | input: folder | output: list ของข้อความ
    return sorted(p.name for p in folder.glob("*.png"))


# ฟังก์ชันตรวจชุดข้อมูล: นับรูปและตรวจว่าทุกรูปบกพร่องมี mask | input: pill_root = โฟลเดอร์ pill | output: dict {train_good, test_good, test_defect, defect_counts} (ยก DatasetError ถ้าพบปัญหา)
def inspect_dataset(pill_root):
    """Count images and check every defect image has a mask. Raises DatasetError."""
    # แปลงเป็น Path | input: pill_root | output: pill_root = Path
    pill_root = Path(pill_root)
    # list เก็บข้อความปัญหาที่พบ | input: - | output: problems = []
    problems = []

    # รายชื่อรูปปกติในชุดฝึก | input: pill_root/train/good | output: train_good = list
    train_good = _pngs(pill_root / "train" / "good")
    # รายชื่อรูปปกติในชุดทดสอบ | input: pill_root/test/good | output: test_good = list
    test_good = _pngs(pill_root / "test" / "good")
    # ชื่อโฟลเดอร์ย่อยทั้งหมดใน test | input: pill_root/test | output: test_dirs = set
    test_dirs = {p.name for p in (pill_root / "test").iterdir() if p.is_dir()}
    # ชื่อโฟลเดอร์ความบกพร่อง = โฟลเดอร์ใน test ที่ไม่ใช่ good | input: test_dirs | output: defects = list เรียงลำดับ
    defects = sorted(test_dirs - {"good"})

    # ถ้าชุดโฟลเดอร์ความบกพร่องไม่ตรงที่คาด
    if set(defects) != EXPECTED_DEFECTS:
        # จดปัญหา: บอกโฟลเดอร์ที่หายและโฟลเดอร์ที่เกินมา | input: defects, EXPECTED_DEFECTS | output: problems ยาวขึ้น 1
        problems.append(
            f"defect folders differ: missing {sorted(EXPECTED_DEFECTS - set(defects))}, "
            f"unexpected {sorted(set(defects) - EXPECTED_DEFECTS)}"
        )

    # dict เก็บจำนวนรูปของแต่ละความบกพร่อง | input: - | output: defect_counts = {}
    defect_counts = {}
    # วนทีละชนิดความบกพร่อง
    for defect in defects:
        # รายชื่อรูปของชนิดนี้ | input: pill_root/test/defect | output: images = list
        images = _pngs(pill_root / "test" / defect)
        # บันทึกจำนวนรูป | input: images | output: defect_counts[defect] = int
        defect_counts[defect] = len(images)
        # โฟลเดอร์ mask ของชนิดนี้ | input: pill_root, defect | output: mask_dir = Path
        mask_dir = pill_root / "ground_truth" / defect
        # รายชื่อ mask ที่มี (เซตว่างถ้าไม่มีโฟลเดอร์) | input: mask_dir | output: masks = set
        masks = set(_pngs(mask_dir)) if mask_dir.is_dir() else set()
        # ชื่อ mask ที่ควรมี: <ชื่อรูป>_mask.png | input: images | output: expected_masks = set
        expected_masks = {f"{Path(name).stem}_mask.png" for name in images}
        # mask ที่ขาดไป | input: expected_masks, masks | output: missing = list เรียงลำดับ
        missing = sorted(expected_masks - masks)
        # ถ้ามี mask ขาด
        if missing:
            # จดปัญหาพร้อมจำนวนและตัวอย่างแรก | input: defect, missing | output: problems ยาวขึ้น 1
            problems.append(f"{defect}: {len(missing)} missing masks, e.g. {missing[0]}")

    # จำนวนรูปบกพร่องรวมทุกชนิด | input: defect_counts | output: total_defect = int
    total_defect = sum(defect_counts.values())
    # ตรวจจำนวนรูปแต่ละส่วนเทียบกับที่คาด
    for label, actual, expected in (
        # รูปปกติของชุดฝึก
        ("train/good", len(train_good), EXPECTED_TRAIN_GOOD),
        # รูปปกติของชุดทดสอบ
        ("test/good", len(test_good), EXPECTED_TEST_GOOD),
        # รูปบกพร่องของชุดทดสอบ
        ("test defects", total_defect, EXPECTED_TEST_DEFECT),
    ):
        # ถ้าจำนวนจริงไม่ตรงที่คาด
        if actual != expected:
            # จดปัญหา | input: label, expected, actual | output: problems ยาวขึ้น 1
            problems.append(f"{label}: expected {expected} images, found {actual}")

    # ถ้าพบปัญหาใด ๆ
    if problems:
        # แจ้งข้อผิดพลาดรวมทุกปัญหาคั่นด้วย "; " | input: problems | output: ยกข้อยกเว้น DatasetError
        raise DatasetError("; ".join(problems))

    # คืนสรุปจำนวนรูป
    return {
        # จำนวนรูปปกติชุดฝึก
        "train_good": len(train_good),
        # จำนวนรูปปกติชุดทดสอบ
        "test_good": len(test_good),
        # จำนวนรูปบกพร่องชุดทดสอบ
        "test_defect": total_defect,
        # จำนวนรูปแยกตามความบกพร่อง
        "defect_counts": defect_counts,
    }


# ฟังก์ชันแบ่งรูปปกติชุดฝึกเป็น train/validation (ไม่แตะชุดทดสอบ) | input: pill_root = โฟลเดอร์ pill, val_ratio = สัดส่วน validation (0-1), seed = เมล็ดสุ่ม | output: dict {seed, val_ratio, train, validation} (list ชื่อไฟล์เรียงลำดับ)
def make_split(pill_root, val_ratio=0.15, seed=42):
    """Split train/good only. The test set is never touched."""
    # ถ้าสัดส่วนไม่อยู่ระหว่าง 0 และ 1
    if not 0 < val_ratio < 1:
        # แจ้งข้อผิดพลาด | input: - | output: ยกข้อยกเว้น ValueError
        raise ValueError("val_ratio must be between 0 and 1")
    # รายชื่อรูปปกติชุดฝึก | input: pill_root | output: names = list
    names = _pngs(Path(pill_root) / "train" / "good")
    # สับลำดับด้วยตัวสุ่มที่กำหนด seed (ผลซ้ำได้) | input: names, seed | output: names ถูกสับในที่เดิม
    random.Random(seed).shuffle(names)
    # จำนวนรูปใน validation (ปัดเศษ) | input: len(names), val_ratio | output: n_val = int
    n_val = round(len(names) * val_ratio)
    # คืนผลการแบ่ง
    return {
        # เมล็ดสุ่มที่ใช้
        "seed": seed,
        # สัดส่วน validation
        "val_ratio": val_ratio,
        # รูปที่เหลือเป็นชุดฝึก (เรียงลำดับ)
        "train": sorted(names[n_val:]),
        # n_val รูปแรกเป็นชุด validation (เรียงลำดับ)
        "validation": sorted(names[:n_val]),
    }


# ฟังก์ชันแบ่งครอป pillQC ปกติตามรูปต้นฉบับ เพื่อไม่ให้รูปต้นฉบับเดียวกันอยู่หลายชุด | input: groups = dict {ชื่อรูปต้นฉบับ: [ชื่อไฟล์ครอป]}, seed = เมล็ดสุ่ม | output: dict {seed, method, photos, test, validation, pool}
def make_pillqc_split(groups, seed=42):
    """Split pillQC normal crops by source photo, so one photo never feeds two sets.

    groups: {photo name: [crop file names]}. Two photos go to test, two to
    validation, and the rest form the pool that memory banks are built from.
    """
    # รายชื่อรูปต้นฉบับ (เรียงลำดับก่อนเพื่อให้ผลสุ่มซ้ำได้) | input: groups | output: photos = list
    photos = sorted(groups)
    # ถ้ามีรูปต้นฉบับน้อยกว่า 5 รูป แบ่งไม่ได้
    if len(photos) < 5:
        # แจ้งข้อผิดพลาด | input: - | output: ยกข้อยกเว้น ValueError
        raise ValueError("need at least 5 source photos: 2 test, 2 validation, 1 pool")
    # สับลำดับรูปต้นฉบับด้วย seed | input: photos, seed | output: photos ถูกสับในที่เดิม
    random.Random(seed).shuffle(photos)
    # 2 รูปแรก = test, 2 รูปถัดไป = validation, ที่เหลือ = pool | input: photos | output: parts = dict
    parts = {"test": photos[:2], "validation": photos[2:4], "pool": photos[4:]}
    # เริ่มโครงผลลัพธ์ | input: seed, parts | output: split = dict
    split = {"seed": seed, "method": "by_source_photo", "photos": parts}
    # วนแต่ละชุด
    for part, names in parts.items():
        # รวมชื่อครอปของทุกรูปต้นฉบับในชุดนั้น เรียงลำดับ | input: names, groups | output: split[part] = list
        split[part] = sorted(crop for name in names for crop in groups[name])
    # คืนผลการแบ่ง | input: split | output: dict
    return split


# ฟังก์ชันหลักของโปรแกรมบรรทัดคำสั่ง | input: argv = list อาร์กิวเมนต์ (None = ใช้ sys.argv) | output: int exit code (0 = สำเร็จ, 1 = ชุดข้อมูลไม่ถูกต้อง) พร้อมเขียนไฟล์ JSON การแบ่งชุด
def main(argv=None):
    # สร้างตัวอ่านอาร์กิวเมนต์ โดยใช้ docstring ของไฟล์เป็นคำอธิบาย | input: __doc__ | output: parser
    parser = argparse.ArgumentParser(description=__doc__)
    # อาร์กิวเมนต์ --root: โฟลเดอร์ pill หรือโฟลเดอร์แม่ (บังคับ) | input: - | output: args.root
    parser.add_argument("--root", required=True, help="pill folder or a parent of it")
    # อาร์กิวเมนต์ --out: path ไฟล์ผลลัพธ์ | input: - | output: args.out
    parser.add_argument("--out", default="splits/pill_seed42.json")
    # อาร์กิวเมนต์ --val-ratio: สัดส่วน validation (ค่าเริ่มต้น 0.15) | input: - | output: args.val_ratio
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # อาร์กิวเมนต์ --seed: เมล็ดสุ่ม (ค่าเริ่มต้น 42) | input: - | output: args.seed
    parser.add_argument("--seed", type=int, default=42)
    # อ่านอาร์กิวเมนต์จาก argv | input: argv | output: args
    args = parser.parse_args(argv)

    # ลองหาโฟลเดอร์และตรวจชุดข้อมูล
    try:
        # หาโฟลเดอร์ pill | input: args.root | output: pill_root = Path
        pill_root = find_category_root(args.root)
        # ตรวจชุดข้อมูล | input: pill_root | output: summary = dict
        summary = inspect_dataset(pill_root)
    # ถ้าชุดข้อมูลไม่ถูกต้อง
    except DatasetError as error:
        # พิมพ์ข้อความผิดพลาดไปที่ stderr | input: error | output: ข้อความบนหน้าจอ
        print(f"Dataset check failed: {error}", file=sys.stderr)
        # คืน exit code 1 (ล้มเหลว) | input: - | output: 1
        return 1

    # แบ่งชุดข้อมูล | input: pill_root, args.val_ratio, args.seed | output: split = dict
    split = make_split(pill_root, args.val_ratio, args.seed)
    # path ไฟล์ผลลัพธ์ | input: args.out | output: out = Path
    out = Path(args.out)
    # สร้างโฟลเดอร์แม่ถ้ายังไม่มี | input: out | output: โฟลเดอร์ใหม่
    out.parent.mkdir(parents=True, exist_ok=True)
    # เขียนผลการแบ่งเป็น JSON ลงท้ายด้วยบรรทัดใหม่ | input: split | output: ไฟล์ JSON
    out.write_text(json.dumps(split, indent=2) + "\n")

    # พิมพ์สรุปชุดข้อมูล | input: summary | output: ข้อความ JSON บนหน้าจอ
    print(json.dumps(summary, indent=2))
    # พิมพ์จำนวน train/validation และ path ไฟล์ | input: split, out | output: ข้อความบนหน้าจอ
    print(f"train={len(split['train'])} validation={len(split['validation'])} -> {out}")
    # คืน exit code 0 (สำเร็จ) | input: - | output: 0
    return 0


# รัน main เมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เรียก main แล้วใช้ค่าที่คืนเป็น exit code ของโปรแกรม | input: - | output: exit code
    sys.exit(main())
