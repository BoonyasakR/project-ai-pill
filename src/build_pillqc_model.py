"""Build the PatchCore model file for pillQC tablets, for the web prototype.

Same settings as the main MVTec model and the round 2 notebook: 256x256 images,
WideResNet50-2, layers 2+3, coreset 0.1, seed 42. The memory bank uses all 46
pool images and the threshold is the 99th percentile of the 54 validation images,
so this is the "all images, seed 42" run of round 2, rebuilt on this machine.

Scores count the pill only, inside `round_pill_region`. Whole-image scores let white
dirt on the floor of some normal validation crops set the threshold (55.21), so most
defects passed. Both are saved; the model file uses the pill-only threshold.

The memory bank also holds each pool pill cut out onto the plain fill used for
multi-pill scenes, so scene pills are compared with pills cut out the same way.

    backend/.venv/bin/python src/build_pillqc_model.py --root data/pillqc
"""
# นำเข้า argparse: อ่านอาร์กิวเมนต์จากบรรทัดคำสั่ง | input: - | output: โมดูล argparse
import argparse
# นำเข้า json: อ่าน/เขียนไฟล์ JSON | input: - | output: โมดูล json
import json
# นำเข้า os: สร้าง symlink | input: - | output: โมดูล os
import os
# นำเข้า sys: แก้ลำดับการค้นหาโมดูล (sys.path) | input: - | output: โมดูล sys
import sys
# นำเข้า tempfile: สร้างโฟลเดอร์ชั่วคราว | input: - | output: โมดูล tempfile
import tempfile
# นำเข้า time: จับเวลาที่ใช้สร้างโมเดล | input: - | output: โมดูล time
import time
# นำเข้า Path: จัดการ path ไฟล์ | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า OpenCV: ปรับขนาดและกัดขอบ mask | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า PyTorch: รันโมเดลและบันทึกไฟล์ .pt | input: - | output: โมดูล torch
import torch
# นำเข้า Image ของ Pillow: เปิด/บันทึกรูป | input: - | output: คลาส Image
from PIL import Image
# นำเข้า Compose, Normalize, Resize: ประกอบขั้นแปลงรูป | input: - | output: คลาสดังกล่าว
from torchvision.transforms.v2 import Compose, Normalize, Resize
# นำเข้า pil_to_tensor และ to_dtype: แปลงรูป PIL เป็น tensor float | input: - | output: ฟังก์ชันดังกล่าว
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

# หาโฟลเดอร์รากของโปรเจกต์ (ขึ้นไป 1 ระดับจาก src) | input: __file__ | output: ROOT = Path
ROOT = Path(__file__).resolve().parents[1]
# เพิ่มโฟลเดอร์ src และ backend ไว้หน้าสุดของ sys.path เพื่อให้นำเข้าโมดูลของโปรเจกต์ได้ | input: ROOT | output: sys.path ถูกแก้ไข
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]
# นำเข้าเมตริกของโปรเจกต์ (noqa = บอกเครื่องมือตรวจโค้ดว่าไม่ต้องเตือนเรื่องลำดับ import) | input: - | output: โมดูล ad_metrics
import ad_metrics  # noqa: E402
# นำเข้าความกว้างขอบที่ตัดทิ้งและท่าวางของ pillQC | input: - | output: PILL_EDGE_PIXELS, PILLQC_POSE
from inference import PILL_EDGE_PIXELS, PILLQC_POSE  # noqa: E402
# นำเข้าฟังก์ชันจัดท่าเม็ด, หาเม็ดชมพู, และสร้างพื้นที่วงกลม | input: - | output: ฟังก์ชันดังกล่าว
from segmentation import align_pill, pink_pill_mask, round_pill_region  # noqa: E402

# ขนาดรูปที่โมเดลรับเข้า = 256x256 | input: - | output: SIZE = 256
SIZE = 256
# ชื่อ backbone ที่ใช้ดึงฟีเจอร์ | input: - | output: BACKBONE = "wide_resnet50_2"
BACKBONE = "wide_resnet50_2"
# ชั้นของ backbone ที่ใช้ดึงฟีเจอร์ | input: - | output: LAYERS = ["layer2", "layer3"]
LAYERS = ["layer2", "layer3"]
# เมล็ดสุ่มเพื่อให้ทำซ้ำผลได้ | input: - | output: SEED = 42
SEED = 42


# ฟังก์ชันสร้างขั้นแปลงรูปก่อนเข้าโมเดล | input: - | output: วัตถุ Compose (ปรับขนาดเป็น 256x256 แล้ว normalize แบบ ImageNet)
def transform():
    # ประกอบขั้นแปลงรูปแล้วคืนค่า
    return Compose([
        # ปรับขนาดเป็น SIZE x SIZE แบบลดรอยหยัก | input: tensor รูป | output: tensor (3, 256, 256)
        Resize((SIZE, SIZE), antialias=True),
        # ปรับค่าสีตามค่าเฉลี่ย/ส่วนเบี่ยงเบนของ ImageNet | input: tensor 0-1 | output: tensor ที่ normalize แล้ว
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ฟังก์ชันหาพื้นที่เม็ดยาที่ใช้คิดคะแนน ในขนาดของโมเดล โดยตัดขอบที่พื้นหลังซึมเข้ามา | input: path = path ไฟล์รูป | output: bool (256, 256)
def pill_region(path):
    """The scored pill area at model size, without the rim where the background bleeds in."""
    # เปิดรูปเป็น RGB อาร์เรย์ หา mask เม็ดชมพู แล้วทำเป็นวงกลมครอบเม็ด | input: path | output: region = bool (H, W)
    region = round_pill_region(pink_pill_mask(np.asarray(Image.open(path).convert("RGB"))))
    # ย่อ region เป็น 256x256 (จุดใกล้สุด) | input: region | output: small = uint8 (256, 256)
    small = cv2.resize(region.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    # กัดขอบเข้า 5 พิกเซล แล้วแปลงเป็น bool แล้วคืนค่า | input: small | output: bool (256, 256)
    return cv2.erode(small, np.ones((PILL_EDGE_PIXELS, PILL_EDGE_PIXELS), np.uint8)) > 0


# ปิดการคำนวณ gradient ตอนอนุมาน
@torch.inference_mode()
# ฟังก์ชันคำนวณคะแนนทั้งรูปและคะแนนเฉพาะเม็ดยา | input: model = โมเดลที่ฝึกแล้ว, paths = list path รูป, batch_size = ขนาด batch | output: tuple (อาร์เรย์คะแนนทั้งรูป, อาร์เรย์คะแนนเฉพาะเม็ด)
def predict(model, paths, batch_size=8):
    """Return whole-image scores and pill-only scores."""
    # เตรียม list เก็บคะแนนทั้งรูป และคะแนนเฉพาะเม็ด | input: - | output: whole, pill = [], []
    whole, pill = [], []
    # วนทีละ batch
    for start in range(0, len(paths), batch_size):
        # ตัดรายการ path ของ batch นี้ | input: paths, start | output: chunk = list
        chunk = paths[start:start + batch_size]
        # แปลงรูปแต่ละรูปเป็น tensor แล้วซ้อนเป็น batch
        batch = torch.stack([
            # เปิดรูป -> tensor float 0-1 -> ปรับขนาดและ normalize | input: path p | output: tensor (3, 256, 256)
            transform()(to_dtype(pil_to_tensor(Image.open(p).convert("RGB")), torch.float32, scale=True))
            # วนทุก path ใน chunk
            for p in chunk
        ])
        # รันโมเดลกับ batch | input: batch | output: output (pred_score, anomaly_map)
        output = model.model(batch)
        # เก็บคะแนนทั้งรูปของ batch นี้ | input: output.pred_score | output: whole ยาวขึ้น
        whole += output.pred_score.flatten().float().tolist()
        # ดึง anomaly map เป็นอาร์เรย์ (N, 256, 256) | input: output.anomaly_map | output: maps
        maps = output.anomaly_map.squeeze(1).float().numpy()
        # คะแนนเฉพาะเม็ด = ค่าสูงสุดของ map ภายใน region ของรูปนั้น | input: maps, chunk | output: pill ยาวขึ้น
        pill += [float(score_map[pill_region(path)].max()) for score_map, path in zip(maps, chunk)]
    # คืนคะแนนทั้งสองแบบเป็นอาร์เรย์ | input: whole, pill | output: (np.array, np.array)
    return np.array(whole), np.array(pill)


# ฟังก์ชันฝึก PatchCore (สร้าง memory bank) | input: pool_paths = รูปปกติที่ใช้สร้าง memory bank, val_paths = รูปปกติชุดตรวจสอบ, work = โฟลเดอร์ทำงานชั่วคราว | output: โมเดล Patchcore ที่ฝึกแล้วในโหมด eval
def train(pool_paths, val_paths, work):
    # นำเข้า Folder: ตัวโหลดข้อมูลจากโฟลเดอร์ | input: - | output: คลาส Folder
    from anomalib.data import Folder
    # นำเข้า Engine: ตัวรันการฝึก | input: - | output: คลาส Engine
    from anomalib.engine import Engine
    # นำเข้า Patchcore: ตัวโมเดล | input: - | output: คลาส Patchcore
    from anomalib.models import Patchcore
    # นำเข้า PreProcessor: ห่อขั้นแปลงรูปให้โมเดล | input: - | output: คลาส PreProcessor
    from anomalib.pre_processing import PreProcessor
    # นำเข้า seed_everything: ตั้งเมล็ดสุ่มทุกไลบรารี | input: - | output: ฟังก์ชัน seed_everything
    from lightning.pytorch import seed_everything

    # ตั้งเมล็ดสุ่มให้ผลซ้ำได้ | input: SEED | output: ตัวสุ่มถูกตั้งค่า
    seed_everything(SEED)
    # วนสร้างโฟลเดอร์ train_good (pool) และ val_good (validation)
    for name, paths in (("train_good", pool_paths), ("val_good", val_paths)):
        # สร้างโฟลเดอร์ย่อยใน work | input: work, name | output: โฟลเดอร์ใหม่
        (work / name).mkdir()
        # วนทุกรูปในกลุ่มนี้
        for path in paths:
            # สร้างทางลัด (symlink) ไปยังรูปต้นฉบับ แทนการคัดลอก | input: path | output: ลิงก์ในโฟลเดอร์ทำงาน
            os.symlink(path, work / name / path.name)
    # Scene pills are cut out onto a plain fill, so their rim never touches the teal floor.
    # The memory bank also gets each pool pill cut out that way, or every scene rim looks abnormal.
    # วนทุกรูปใน pool เพื่อสร้างเม็ดที่ตัดออกมาวางบนพื้นเรียบ แบบเดียวกับฉากหลายเม็ด
    for path in pool_paths:
        # เปิดรูปเป็น RGB อาร์เรย์ | input: path | output: image = uint8 (H, W, 3)
        image = np.asarray(Image.open(path).convert("RGB"))
        # หา mask เม็ดชมพู | input: image | output: mask = bool (H, W)
        mask = pink_pill_mask(image)
        # จัดท่าเม็ด (เฉพาะส่วนที่อยู่ในวงกลม) ตามท่าของ pillQC; ค่าที่สองไม่ใช้ | input: image, mask, ค่า align | output: view = uint8 (225, 225, 3)
        view, _ = align_pill(image, mask & round_pill_region(mask), **PILLQC_POSE["align"])
        # บันทึกรูปที่ตัดแล้วเข้า train_good ด้วยชื่อ cut_<ชื่อเดิม>.png | input: view | output: ไฟล์ PNG
        Image.fromarray(view).save(work / "train_good" / f"cut_{path.stem}.png")
    # สร้างตัวโหลดข้อมูลจากโฟลเดอร์ที่เตรียมไว้
    datamodule = Folder(
        # ชื่อชุดข้อมูล, โฟลเดอร์ราก, โฟลเดอร์รูปปกติสำหรับฝึก และสำหรับทดสอบ
        name="pillqc", root=work, normal_dir="train_good", normal_test_dir="val_good",
        # ขนาด batch ตอนฝึก/ประเมิน และไม่ใช้ worker เพิ่ม
        train_batch_size=8, eval_batch_size=8, num_workers=0,
        # แบ่งชุดทดสอบจากโฟลเดอร์ และใช้ชุดทดสอบเดียวกับชุดตรวจสอบ
        test_split_mode="from_dir", val_split_mode="same_as_test",
    )
    # สร้างโมเดล PatchCore
    model = Patchcore(
        # ใช้ backbone และ layer ที่กำหนด เก็บ 10% ของ patch ใน memory bank (coreset)
        backbone=BACKBONE, layers=tuple(LAYERS), coreset_sampling_ratio=0.1,
        # ใช้ขั้นแปลงรูปของเรา และปิดส่วนหลังประมวลผล/ประเมิน/แสดงผลของ anomalib
        pre_processor=PreProcessor(transform=transform()), post_processor=False, evaluator=False, visualizer=False,
    )
    # สร้างตัวรันการฝึกบน CPU 1 อุปกรณ์ ไม่เก็บ log และไม่แสดงแถบความคืบหน้า
    engine = Engine(default_root_dir=work / "runs", logger=False, accelerator="cpu", devices=1,
                    enable_progress_bar=False, num_sanity_val_steps=0)
    # ฝึกโมเดล (สร้าง memory bank จากรูปปกติ) | input: model, datamodule | output: model ที่ฝึกแล้ว
    engine.fit(model=model, datamodule=datamodule)
    # สลับเป็นโหมดประเมินผลแล้วคืนโมเดล | input: model | output: model
    return model.eval()


# ฟังก์ชันหลัก: ฝึกโมเดล, คำนวณ threshold, บันทึกไฟล์โมเดลและผลลัพธ์ | input: อาร์กิวเมนต์บรรทัดคำสั่ง --root (บังคับ) และ --out | output: ไฟล์โมเดล .pt, ไฟล์ผลลัพธ์ .json และสรุปที่พิมพ์ออกหน้าจอ
def main():
    # สร้างตัวอ่านอาร์กิวเมนต์ โดยใช้ docstring ของไฟล์เป็นคำอธิบาย | input: __doc__ | output: parser
    parser = argparse.ArgumentParser(description=__doc__)
    # อาร์กิวเมนต์ --root: โฟลเดอร์ของ pillQC (บังคับ) | input: - | output: args.root
    parser.add_argument("--root", required=True, help="pillQC repository folder")
    # อาร์กิวเมนต์ --out: path ไฟล์โมเดลที่จะบันทึก (ค่าเริ่มต้น models/patchcore_pillqc_256.pt) | input: - | output: args.out
    parser.add_argument("--out", default=str(ROOT / "models" / "patchcore_pillqc_256.pt"))
    # อ่านอาร์กิวเมนต์จากบรรทัดคำสั่ง | input: sys.argv | output: args
    args = parser.parse_args()

    # โฟลเดอร์รูปของ pillQC | input: args.root | output: images = Path
    images = Path(args.root).resolve() / "images"
    # อ่านไฟล์แบ่งชุดข้อมูล (pool/validation/test) | input: ไฟล์ splits/pillqc_seed42.json | output: split = dict
    split = json.loads((ROOT / "splits" / "pillqc_seed42.json").read_text())
    # path รูปปกติของชุด pool (ใช้สร้าง memory bank) | input: split["pool"] | output: pool = list ของ Path
    pool = [images / "normal" / name for name in split["pool"]]
    # path รูปปกติของชุด validation (ใช้ตั้ง threshold) | input: split["validation"] | output: val = list ของ Path
    val = [images / "normal" / name for name in split["validation"]]
    # path รูปปกติของชุด test | input: split["test"] | output: test = list ของ Path
    test = [images / "normal" / name for name in split["test"]]
    # เพิ่มรูปบกพร่อง (chip = บิ่น, dirt = สกปรก) เข้าชุดทดสอบ | input: โฟลเดอร์ chip, dirt | output: test ยาวขึ้น
    test += sorted((images / "chip").glob("*.jpg")) + sorted((images / "dirt").glob("*.jpg"))
    # ชื่อกลุ่มของแต่ละรูปทดสอบ (ชื่อโฟลเดอร์แม่: normal/chip/dirt) | input: test | output: groups = อาร์เรย์ของข้อความ
    groups = np.array([p.parent.name for p in test])
    # ป้ายกำกับ: 0 = ปกติ, 1 = บกพร่อง | input: groups | output: labels = อาร์เรย์ int
    labels = (groups != "normal").astype(int)

    # เริ่มจับเวลา | input: - | output: start = float (วินาที)
    start = time.perf_counter()
    # สร้างโฟลเดอร์ชั่วคราวที่ลบเองเมื่อจบบล็อก
    with tempfile.TemporaryDirectory() as work:
        # ฝึกโมเดลในโฟลเดอร์ชั่วคราว | input: pool, val, Path(work) | output: model
        model = train(pool, val, Path(work))
    # คำนวณคะแนนของรูปชุด validation (ทั้งรูปและเฉพาะเม็ด) | input: model, val | output: val_whole, val_scores
    val_whole, val_scores = predict(model, val)
    # threshold เฉพาะเม็ด = ควอนไทล์ที่ 99 ของคะแนน validation | input: val_scores | output: threshold = float
    threshold = ad_metrics.pick_threshold(val_scores)
    # คำนวณคะแนนของรูปชุดทดสอบ | input: model, test | output: test_whole, test_scores
    test_whole, test_scores = predict(model, test)
    # threshold แบบทั้งรูป (เก็บไว้เปรียบเทียบ) | input: val_whole | output: whole_threshold = float
    whole_threshold = ad_metrics.pick_threshold(val_whole)

    # path ไฟล์โมเดลปลายทาง | input: args.out | output: out = Path
    out = Path(args.out)
    # สร้างโฟลเดอร์แม่ของไฟล์ถ้ายังไม่มี | input: out | output: โฟลเดอร์ใหม่
    out.parent.mkdir(parents=True, exist_ok=True)
    # บันทึกไฟล์โมเดล
    torch.save({
        # น้ำหนักของโมเดล (รวม memory bank)
        "state_dict": model.model.state_dict(),
        # threshold ที่ใช้ตัดสิน
        "threshold": threshold,
        # ขนาดรูปที่โมเดลรับเข้า
        "image_size": SIZE,
        # ชื่อ backbone
        "backbone": BACKBONE,
        # ชั้นที่ใช้
        "layers": LAYERS,
        # บอกให้ตอนใช้งานให้คะแนนเฉพาะเม็ดยา
        "score_area": "pill",
    }, out)

    # สร้างสรุปผลการสร้างโมเดล
    summary = {
        # การตั้งค่าที่ใช้
        "settings": {"size": SIZE, "backbone": BACKBONE, "layers": LAYERS, "coreset": 0.1, "seed": SEED,
                     "pool_images": len(pool), "cut_out_pool_images": len(pool),
                     "validation_images": len(val), "device": "cpu"},
        # threshold เฉพาะเม็ด
        "threshold": threshold,
        # เมตริกระดับรูปบนชุดทดสอบ | input: labels, test_scores, threshold | output: dict
        "overall": ad_metrics.image_metrics(labels, test_scores, threshold),
        # recall แยกตามชนิดความบกพร่อง (chip, dirt) | output: dict {ชนิด: float}
        "recall_by_defect": {g: float(np.mean(test_scores[groups == g] > threshold)) for g in ("chip", "dirt")},
        # คะแนนของรูป validation
        "validation_scores": val_scores.tolist(),
        # คะแนนของแต่ละรูปทดสอบ โดยใช้ "กลุ่ม/ชื่อไฟล์" เป็นคีย์ | output: dict
        "test_scores": {f"{g}/{p.name}": float(s) for g, p, s in zip(groups, test, test_scores)},
        # พื้นที่ที่ใช้ให้คะแนน
        "score_area": "pill",
        # ผลของการให้คะแนนแบบทั้งรูป เพื่อเปรียบเทียบ
        "whole_image": {
            # threshold แบบทั้งรูป
            "threshold": whole_threshold,
            # เมตริกระดับรูปแบบทั้งรูป
            "overall": ad_metrics.image_metrics(labels, test_whole, whole_threshold),
            # recall แยกตามความบกพร่อง แบบทั้งรูป
            "recall_by_defect": {g: float(np.mean(test_whole[groups == g] > whole_threshold)) for g in ("chip", "dirt")},
            # คะแนน validation แบบทั้งรูป
            "validation_scores": val_whole.tolist(),
        },
        # เวลาที่ใช้สร้างโมเดล (วินาที ปัดเศษ 1 ตำแหน่ง)
        "build_seconds": round(time.perf_counter() - start, 1),
    }
    # โฟลเดอร์เก็บผลลัพธ์ | input: ROOT | output: results = Path
    results = ROOT / "results" / "pillqc_model"
    # สร้างโฟลเดอร์ถ้ายังไม่มี | input: results | output: โฟลเดอร์ใหม่
    results.mkdir(parents=True, exist_ok=True)
    # บันทึกสรุปเป็นไฟล์ JSON | input: summary | output: ไฟล์ pillqc_model_results.json
    (results / "pillqc_model_results.json").write_text(json.dumps(summary, indent=1))
    # พิมพ์เฉพาะหัวข้อสำคัญของสรุปออกหน้าจอ | input: summary | output: ข้อความ JSON บนหน้าจอ
    print(json.dumps({k: summary[k] for k in ("threshold", "overall", "recall_by_defect", "build_seconds")}, indent=1))


# รัน main เมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เรียกฟังก์ชันหลัก | input: - | output: -
    main()
