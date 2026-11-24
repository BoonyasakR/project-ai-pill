"""Train PatchCore on this machine and evaluate it, with the same settings as the Colab notebooks.

    backend/.venv/bin/python src/train_patchcore.py --size 256 --out models/patchcore_pill_256.pt

The threshold comes from normal validation images only. Test results are printed and saved.
"""
# นำเข้า argparse: อ่านอาร์กิวเมนต์บรรทัดคำสั่ง | input: - | output: โมดูล argparse
import argparse
# นำเข้า json: อ่าน/เขียนไฟล์ JSON | input: - | output: โมดูล json
import json
# นำเข้า sys: แก้ sys.path | input: - | output: โมดูล sys
import sys
# นำเข้า time: จับเวลาการฝึก | input: - | output: โมดูล time
import time
# นำเข้า Path: จัดการ path | input: - | output: คลาส Path
from pathlib import Path

# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า PyTorch: tensor และโมเดล | input: - | output: โมดูล torch
import torch
# นำเข้า SparseRandomProjection: ลดมิติฟีเจอร์ก่อนเลือก coreset | input: - | output: คลาส SparseRandomProjection
from anomalib.models.components.dimensionality_reduction import SparseRandomProjection
# นำเข้าโมเดล PatchCore | input: - | output: คลาส PatchcoreModel
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
# นำเข้า seed_everything: ตั้งเมล็ดสุ่มทุกไลบรารี | input: - | output: ฟังก์ชัน seed_everything
from lightning.pytorch import seed_everything
# นำเข้า Image ของ Pillow: เปิดรูป | input: - | output: คลาส Image
from PIL import Image
# นำเข้า Compose, Normalize, Resize: ประกอบขั้นแปลงรูป | input: - | output: คลาสดังกล่าว
from torchvision.transforms.v2 import Compose, Normalize, Resize
# นำเข้า pil_to_tensor และ to_dtype: แปลงรูป PIL เป็น tensor float | input: - | output: ฟังก์ชันดังกล่าว
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

# โฟลเดอร์รากของโปรเจกต์ | input: __file__ | output: ROOT = Path
ROOT = Path(__file__).resolve().parents[1]
# เพิ่ม src และ backend ไว้หน้าสุดของ sys.path | input: ROOT | output: sys.path ถูกแก้ไข
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]
# นำเข้าเมตริกของโปรเจกต์ | input: - | output: โมดูล ad_metrics
import ad_metrics  # noqa: E402
# นำเข้าพื้นที่สีแปลกขั้นต่ำ, ตัวสร้างตารางสี, ตัววัดสีแปลก | input: - | output: MIN_AREA, build_seen, novel_colour
from colour_check import MIN_AREA, build_seen, novel_colour  # noqa: E402
# นำเข้าฟังก์ชันหาเม็ดยาในรูป | input: - | output: find_pills
from segmentation import find_pills  # noqa: E402

# โฟลเดอร์ข้อมูลรูป MVTec pill | input: ROOT | output: DATA = Path
DATA = ROOT / "data" / "mvtec" / "pill"
# ชื่อ backbone | input: - | output: BACKBONE = "wide_resnet50_2"
BACKBONE = "wide_resnet50_2"
# ชั้นของ backbone ที่ใช้ | input: - | output: LAYERS = ["layer2", "layer3"]
LAYERS = ["layer2", "layer3"]
# เลือกอุปกรณ์: GPU ของ Apple (mps) ถ้ามี ไม่งั้น CPU | input: - | output: DEVICE = "mps" หรือ "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# ขนาด batch | input: - | output: BATCH = 8
BATCH = 8


# ฟังก์ชันสร้างขั้นแปลงรูปก่อนเข้าโมเดล | input: size = ขนาดด้านของรูปที่โมเดลรับ | output: วัตถุ Compose (ปรับขนาดเป็น size x size แล้ว normalize แบบ ImageNet)
def make_transform(size):
    # ประกอบขั้นแปลงรูปแล้วคืนค่า
    return Compose([
        # ปรับขนาดเป็น size x size แบบลดรอยหยัก | input: tensor รูป | output: tensor (3, size, size)
        Resize((size, size), antialias=True),
        # ปรับค่าสีตามค่าเฉลี่ย/ส่วนเบี่ยงเบนของ ImageNet | input: tensor 0-1 | output: tensor ที่ normalize แล้ว
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ฟังก์ชันโหลดรูปเป็น tensor | input: path = path ไฟล์รูป | output: tensor float32 (3, H, W) ค่า 0-1
def load_image(path):
    # เปิดรูป แปลงเป็น RGB แล้วเป็น tensor float ปรับค่าเป็น 0-1 | input: path | output: tensor
    return to_dtype(pil_to_tensor(Image.open(path).convert("RGB")), torch.float32, scale=True)


# ตัวสร้าง (generator) แบ่งรูปเป็น batch พร้อมแปลงและย้ายไปอุปกรณ์ | input: paths = list path รูป, size = ขนาดรูปที่โมเดลรับ | output: ผลิต tensor (ขนาด batch, 3, size, size) บน DEVICE ทีละ batch
def batches(paths, size):
    # สร้างขั้นแปลงรูปครั้งเดียว | input: size | output: transform
    transform = make_transform(size)
    # วนเป็นก้าวละ BATCH รูป
    for start in range(0, len(paths), BATCH):
        # โหลดและแปลงรูปใน batch นี้ ซ้อนเป็น tensor ย้ายไปอุปกรณ์ แล้วส่งออก | input: paths[start:start+BATCH] | output: tensor batch
        yield torch.stack([transform(load_image(path)) for path in paths[start:start + BATCH]]).to(DEVICE)


# ฟังก์ชันเลือกจุดตัวแทน (coreset) แบบ k-center greedy | input: embedding = tensor ฟีเจอร์ (จำนวน patch, มิติ), ratio = สัดส่วนที่จะเก็บ | output: tensor ดัชนีของแถวที่เลือก (บน DEVICE)
def coreset_indices(embedding, ratio):
    """k-center greedy selection, as in anomalib's KCenterGreedy, picking the same points.

    Distances come from one matrix-vector product per step, which is far faster on MPS than
    anomalib's per-row norm.
    """
    # ลดมิติฟีเจอร์ด้วย random projection แล้วย้ายไปอุปกรณ์ | input: embedding | output: features = tensor (N, มิติที่ลดแล้ว)
    features = SparseRandomProjection(eps=0.9).fit(embedding).transform(embedding).to(DEVICE)
    # กำลังสองของความยาวเวกเตอร์ของแต่ละจุด | input: features | output: squared = tensor (N,)
    squared = (features * features).sum(1)
    # ระยะใกล้สุดไปยังจุดที่เลือกแล้ว เริ่มที่อนันต์ | input: squared | output: nearest = tensor (N,)
    nearest = torch.full_like(squared, float("inf"))
    # สุ่มจุดเริ่มต้น 1 จุด | input: len(features) | output: index = tensor สเกลาร์
    index = torch.randint(len(features), (1,), device=features.device).squeeze()
    # list ของดัชนีที่เลือก เริ่มจากจุดแรก | input: index | output: chosen = [index]
    chosen = [index]
    # วนเลือกจุดที่เหลือจนครบจำนวนที่ต้องการ
    for _ in range(int(len(features) * ratio) - 1):
        # ระยะกำลังสองจากทุกจุดไปยังจุดที่เพิ่งเลือก (ตัดค่าติดลบจากความคลาดเคลื่อนเป็น 0) | input: squared, features, index | output: distance = tensor (N,)
        distance = (squared + squared[index] - 2 * (features @ features[index])).clamp_min_(0)
        # ปรับระยะใกล้สุดให้เป็นค่าน้อยกว่าระหว่างเดิมกับใหม่ | input: nearest, distance | output: nearest
        nearest = torch.minimum(nearest, distance)
        # จุดที่เลือกแล้วมีระยะเป็น 0 | input: nearest, index | output: nearest ถูกแก้
        nearest[index] = 0
        # เลือกจุดที่อยู่ไกลที่สุดจากชุดที่เลือกไว้ | input: nearest | output: index ใหม่
        index = torch.argmax(nearest)
        # เพิ่มดัชนีนี้เข้า list | input: index | output: chosen ยาวขึ้น 1
        chosen.append(index)
    # รวมดัชนีเป็น tensor แล้วคืนค่า | input: chosen | output: tensor (จำนวนที่เลือก,)
    return torch.stack(chosen)


# ฟังก์ชันฝึกโมเดล PatchCore (สร้าง memory bank) | input: paths = รูปปกติชุดฝึก, size = ขนาดรูป, coreset = สัดส่วน coreset | output: โมเดลที่ฝึกแล้วในโหมด eval
def train(paths, size, coreset):
    # ตั้งเมล็ดสุ่มให้ผลซ้ำได้ | input: 42 | output: ตัวสุ่มถูกตั้งค่า
    seed_everything(42)
    # สร้างโมเดลแล้วย้ายไปอุปกรณ์ | input: LAYERS, BACKBONE | output: model
    model = PatchcoreModel(layers=LAYERS, backbone=BACKBONE).to(DEVICE)
    # สลับเป็นโหมดฝึกเพื่อให้โมเดลคืนฟีเจอร์ของ patch | input: - | output: model ในโหมด train
    model.train()
    # Fill one CPU tensor batch by batch. At 512 pixels the features take 5.7 GB, too much for MPS.
    # เตรียมตัวแปร embedding (สร้างตอนรู้ขนาด) และตำแหน่งแถวที่จะเขียน | input: - | output: embedding = None, row = 0
    embedding, row = None, 0
    # วนทีละ batch ของรูปฝึก
    for batch in batches(paths, size):
        # ดึงฟีเจอร์ของ patch ทุก patch ใน batch แล้วย้ายไป CPU | input: batch | output: part = tensor (patch ใน batch, มิติ)
        part = model(batch).cpu()
        # ล้างที่เก็บฟีเจอร์ภายในของโมเดล | input: - | output: embedding_store ว่าง
        model.embedding_store.clear()
        # ถ้ายังไม่ได้สร้าง tensor ใหญ่
        if embedding is None:
            # จองที่ว่างให้ทุกรูป โดยประมาณจากจำนวน patch ต่อรูปของ batch แรก | input: len(paths), part, batch | output: embedding = tensor ว่าง
            embedding = torch.empty(len(paths) * len(part) // len(batch), part.shape[1])
        # เขียนฟีเจอร์ของ batch นี้ลงตำแหน่งต่อจากเดิม | input: part, row | output: embedding ถูกเติม
        embedding[row:row + len(part)] = part
        # เลื่อนตำแหน่งแถวถัดไป | input: row, len(part) | output: row
        row += len(part)
    # เลือก coreset แล้วเก็บเป็น memory bank บนอุปกรณ์ | input: embedding, coreset | output: model.memory_bank
    model.memory_bank = embedding[coreset_indices(embedding, coreset).cpu()].to(DEVICE)
    # สลับเป็นโหมดประเมินแล้วคืนโมเดล | input: model | output: model
    return model.eval()


# ปิดการคำนวณ gradient ตอนอนุมาน
@torch.inference_mode()
# ฟังก์ชันคำนวณคะแนนและ anomaly map ของรูป | input: model = โมเดลที่ฝึกแล้ว, paths = list path รูป, size = ขนาดรูป | output: tuple (อาร์เรย์คะแนนต่อรูป (N,), อาร์เรย์ anomaly map (N, size, size))
def predict(model, paths, size):
    # เตรียม list เก็บคะแนนและ map ของแต่ละ batch | input: - | output: scores, maps = [], []
    scores, maps = [], []
    # วนทีละ batch
    for batch in batches(paths, size):
        # รันโมเดล | input: batch | output: output (pred_score, anomaly_map)
        output = model(batch)
        # เก็บคะแนนของ batch เป็นอาร์เรย์ใน CPU | input: output.pred_score | output: scores ยาวขึ้น 1
        scores.append(output.pred_score.flatten().float().cpu().numpy())
        # เก็บ anomaly map ของ batch เป็นอาร์เรย์ใน CPU | input: output.anomaly_map | output: maps ยาวขึ้น 1
        maps.append(output.anomaly_map.squeeze(1).float().cpu().numpy())
    # ต่อทุก batch เข้าด้วยกันแล้วคืนค่า | input: scores, maps | output: (อาร์เรย์, อาร์เรย์)
    return np.concatenate(scores), np.concatenate(maps)


# ฟังก์ชันโหลด mask ความบกพร่องที่ปรับขนาดแล้ว | input: path = path รูปทดสอบ, size = ขนาดผลลัพธ์ | output: bool (size, size) True = พิกเซลบกพร่อง
def load_mask(path, size):
    # ถ้าเป็นรูปปกติ (อยู่ในโฟลเดอร์ good)
    if path.parent.name == "good":
        # คืน mask ว่าง (ไม่มีพิกเซลบกพร่อง) | input: size | output: bool (size, size) ค่า False
        return np.zeros((size, size), dtype=bool)
    # path ของ mask ความจริง: ground_truth/<ชนิด>/<ชื่อรูป>_mask.png | input: path | output: mask_path
    mask_path = DATA / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    # เปิด mask เป็นขาวดำ ย่อเป็น size x size (จุดใกล้สุด) และแปลงเป็น bool (พิกเซล > 0) | input: mask_path, size | output: bool (size, size)
    return np.asarray(Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)) > 0


# ฟังก์ชันโหลดรูปและ mask ของเม็ดยาในรูปนั้น | input: path = path รูปที่มีเม็ดเดียว | output: tuple (รูป RGB (H, W, 3), mask เม็ดแรกที่หาเจอ bool (H, W))
def pill(path):
    # เปิดรูปเป็น RGB อาร์เรย์ | input: path | output: image
    image = np.asarray(Image.open(path).convert("RGB"))
    # คืนรูปพร้อม mask ของเม็ดแรก (ใหญ่สุด) | input: image | output: (image, mask)
    return image, find_pills(image)[0]


# ฟังก์ชันคำนวณพื้นที่สีแปลกของแต่ละรูป | input: seen = ตารางสีที่เคยเห็น, paths = list path รูป | output: อาร์เรย์ int (N,) พื้นที่กลุ่มสีแปลกใหญ่สุดของแต่ละรูป
def colour_areas(seen, paths):
    # วนทุกรูป หาเม็ดแล้ววัดสีแปลก เอาเฉพาะพื้นที่ (ค่าที่ 0) | input: seen, paths | output: np.array
    return np.array([novel_colour(seen, *pill(path))[0] for path in paths])


# ฟังก์ชันหลัก: ฝึก ประเมิน และบันทึกโมเดล | input: อาร์กิวเมนต์บรรทัดคำสั่ง --size, --coreset, --out (บังคับ) | output: ไฟล์โมเดล .pt, ไฟล์ผลลัพธ์ .json และข้อความสรุปบนหน้าจอ
def main():
    # สร้างตัวอ่านอาร์กิวเมนต์ | input: - | output: parser
    parser = argparse.ArgumentParser()
    # อาร์กิวเมนต์ --size: ขนาดรูป (ค่าเริ่มต้น 256) | input: - | output: args.size
    parser.add_argument("--size", type=int, default=256)
    # อาร์กิวเมนต์ --coreset: สัดส่วน coreset (ค่าเริ่มต้น 0.1) | input: - | output: args.coreset
    parser.add_argument("--coreset", type=float, default=0.1)
    # อาร์กิวเมนต์ --out: path ไฟล์โมเดลที่จะบันทึก (บังคับ) | input: - | output: args.out
    parser.add_argument("--out", type=Path, required=True)
    # อ่านอาร์กิวเมนต์จากบรรทัดคำสั่ง | input: sys.argv | output: args
    args = parser.parse_args()

    # อ่านไฟล์แบ่งชุดข้อมูล | input: splits/pill_seed42.json | output: split = dict
    split = json.loads((ROOT / "splits" / "pill_seed42.json").read_text())
    # path รูปชุดฝึก | input: split["train"] | output: train_paths = list ของ Path
    train_paths = [DATA / "train" / "good" / name for name in split["train"]]
    # path รูปชุด validation | input: split["validation"] | output: val_paths = list ของ Path
    val_paths = [DATA / "train" / "good" / name for name in split["validation"]]
    # path รูปชุดทดสอบทั้งหมด (เรียงชื่อ) | input: DATA/test | output: test_paths = list ของ Path
    test_paths = sorted((DATA / "test").glob("*/*.png"))
    # ชื่อกลุ่มของแต่ละรูปทดสอบ (ชื่อโฟลเดอร์แม่) | input: test_paths | output: groups = list ของข้อความ
    groups = [path.parent.name for path in test_paths]

    # เริ่มจับเวลา | input: - | output: start = float
    start = time.perf_counter()
    # ฝึกโมเดล | input: train_paths, args.size, args.coreset | output: model
    model = train(train_paths, args.size, args.coreset)
    # พิมพ์เวลาที่ใช้ฝึกและขนาด memory bank | input: เวลา, model.memory_bank | output: ข้อความบนหน้าจอ
    print("fit seconds", round(time.perf_counter() - start, 1), "memory bank", tuple(model.memory_bank.shape))
    # คำนวณคะแนนของรูป validation (ไม่ใช้ map จึงรับด้วย _) | input: model, val_paths, args.size | output: val_scores
    val_scores, _ = predict(model, val_paths, args.size)
    # เลือก threshold จากควอนไทล์ที่ 99 ของคะแนน validation | input: val_scores | output: threshold = float
    threshold = ad_metrics.pick_threshold(val_scores)

    # คำนวณคะแนนและ anomaly map ของรูปทดสอบ | input: model, test_paths, args.size | output: scores, maps
    scores, maps = predict(model, test_paths, args.size)
    # โหลด mask ความจริงของทุกรูปทดสอบ ซ้อนเป็นอาร์เรย์ | input: test_paths, args.size | output: masks = bool (N, size, size)
    masks = np.stack([load_mask(path, args.size) for path in test_paths])
    # ป้ายกำกับ: 0 = ปกติ, 1 = บกพร่อง | input: groups | output: labels = list ของ int
    labels = [int(group != "good") for group in groups]
    # รวมผลประเมิน
    result = {
        # เมตริกรวมทุกรูป | input: labels, scores, masks, maps, threshold | output: dict
        "overall": ad_metrics.evaluate(labels, scores, masks, maps, threshold),
        # เมตริกแยกตามชนิดความบกพร่อง | input: groups, scores, masks, maps, threshold | output: dict
        "by_defect": ad_metrics.evaluate_by_defect(groups, scores, masks, maps, threshold),
        # คะแนน validation | output: list ของ float
        "validation_scores": val_scores.tolist(),
        # คะแนนของแต่ละรูปทดสอบ โดยใช้ "กลุ่ม/ชื่อไฟล์" เป็นคีย์ | output: dict
        "test_scores": {f"{g}/{p.name}": float(s) for g, p, s in zip(groups, test_paths, scores)},
    }
    # พิมพ์เมตริกรวม ปัดเศษ 4 ตำแหน่ง | input: result["overall"] | output: ข้อความบนหน้าจอ
    print("patchcore", json.dumps({key: round(value, 4) for key, value in result["overall"].items()}))
    # วนพิมพ์เมตริกของแต่ละชนิดความบกพร่อง
    for defect, row in result["by_defect"].items():
        # พิมพ์ชื่อชนิดและเมตริก ปัดเศษ 3 ตำแหน่ง | input: defect, row | output: ข้อความบนหน้าจอ
        print(defect, json.dumps({key: round(value, 3) for key, value in row.items()}))

    # Colour check: colours from training pills, threshold from validation pills.
    # สร้างตารางสีที่เคยเห็นจากเม็ดยาในชุดฝึก | input: train_paths | output: seen = bool (64, 64, 64)
    seen = build_seen(pill(path) for path in train_paths)
    # พื้นที่สีแปลกของรูป validation | input: seen, val_paths | output: val_areas = อาร์เรย์
    val_areas = colour_areas(seen, val_paths)
    # threshold ของสี = ควอนไทล์ที่ 99 ของ validation แต่ไม่ต่ำกว่า MIN_AREA | input: val_areas | output: colour_threshold = float
    colour_threshold = max(float(np.quantile(val_areas, 0.99)), MIN_AREA)
    # พื้นที่สีแปลกของรูปทดสอบ | input: seen, test_paths | output: areas = อาร์เรย์
    areas = colour_areas(seen, test_paths)
    # One score for AUROC: each check divided by its own threshold, larger one kept.
    # คะแนนรวม: แต่ละการตรวจหารด้วยเกณฑ์ของตัวเอง แล้วเอาค่าที่มากกว่า (>1 = ผิดปกติ) | input: scores, threshold, areas, colour_threshold | output: combined = อาร์เรย์ (N,)
    combined = np.maximum(scores / threshold, areas / colour_threshold)
    # เก็บผลของแบบที่รวมการตรวจสี
    result["with_colour"] = {
        # threshold ของสี
        "colour_threshold": colour_threshold,
        # พื้นที่สีแปลกของรูป validation
        "validation_colour_areas": val_areas.tolist(),
        # พื้นที่สีแปลกของแต่ละรูปทดสอบ | output: dict
        "test_colour_areas": {f"{g}/{p.name}": int(a) for g, p, a in zip(groups, test_paths, areas)},
        # เมตริกรวม โดยใช้คะแนนรวมกับเกณฑ์ 1.0 | input: labels, combined, masks, maps, 1.0 | output: dict
        "overall": ad_metrics.evaluate(labels, combined, masks, maps, 1.0),
        # เมตริกแยกตามความบกพร่อง | input: groups, combined, masks, maps, 1.0 | output: dict
        "by_defect": ad_metrics.evaluate_by_defect(groups, combined, masks, maps, 1.0),
    }
    # พิมพ์เมตริกรวมของแบบมีการตรวจสี | input: result["with_colour"]["overall"] | output: ข้อความบนหน้าจอ
    print("with colour", json.dumps({key: round(value, 4) for key, value in result["with_colour"]["overall"].items()}))
    # วนพิมพ์เมตริกของแต่ละชนิดความบกพร่อง (แบบมีการตรวจสี)
    for defect, row in result["with_colour"]["by_defect"].items():
        # พิมพ์ชื่อชนิดและเมตริก | input: defect, row | output: ข้อความบนหน้าจอ
        print(defect, json.dumps({key: round(value, 3) for key, value in row.items()}))

    # สร้างโฟลเดอร์ของไฟล์โมเดลถ้ายังไม่มี | input: args.out | output: โฟลเดอร์ใหม่
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # บันทึกไฟล์โมเดล
    torch.save({
        # น้ำหนักของโมเดล (ย้ายมา CPU ก่อน)
        "state_dict": model.cpu().state_dict(),
        # threshold ที่ใช้ตัดสิน
        "threshold": threshold,
        # ขนาดรูปที่โมเดลรับ
        "image_size": args.size,
        # ชื่อ backbone
        "backbone": BACKBONE,
        # ชั้นที่ใช้
        "layers": LAYERS,
        # ตารางสีที่เคยเห็น บีบอัดเป็นบิตแล้วแปลงเป็น tensor
        "colour_seen": torch.from_numpy(np.packbits(seen)),
        # threshold ของสี
        "colour_threshold": colour_threshold,
    }, args.out)
    # บันทึกผลประเมินเป็น JSON ข้างไฟล์โมเดล (เปลี่ยนนามสกุลเป็น .json) | input: result | output: ไฟล์ .json
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1))


# รัน main เมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    # เรียกฟังก์ชันหลัก | input: - | output: -
    main()
