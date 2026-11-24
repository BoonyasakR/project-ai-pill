"""Evaluation metrics for anomaly detection. Model independent."""
# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า ndimage ของ SciPy: ติดป้ายบริเวณที่เชื่อมกัน | input: - | output: โมดูล ndimage
from scipy import ndimage
# นำเข้า roc_auc_score ของ scikit-learn: คำนวณพื้นที่ใต้กราฟ ROC | input: - | output: ฟังก์ชัน roc_auc_score
from sklearn.metrics import roc_auc_score


# ฟังก์ชันเลือก threshold ระดับรูปจากรูปปกติชุดตรวจสอบเท่านั้น | input: validation_scores = คะแนนของรูปปกติ, quantile = ควอนไทล์ (ค่าเริ่มต้น 0.99) | output: float ค่า threshold
def pick_threshold(validation_scores, quantile=0.99):
    """Image-score threshold taken from normal validation images only."""
    # แปลงคะแนนเป็นอาร์เรย์ float64 แล้วหาค่าควอนไทล์ | input: validation_scores, quantile | output: float
    return float(np.quantile(np.asarray(validation_scores, dtype=np.float64), quantile))


# ฟังก์ชันคำนวณ AU-PRO (พื้นที่ใต้กราฟสัดส่วนการซ้อนทับรายบริเวณ จนถึง FPR ที่กำหนด) | input: masks = bool (N, H, W), maps = float (N, H, W), fpr_limit = FPR สูงสุด, steps = จำนวนขั้น | output: float ค่า 0-1
def au_pro(masks, maps, fpr_limit=0.3, steps=200):
    """Area under the per-region-overlap curve up to fpr_limit, scaled to 0-1.

    masks: bool array (N, H, W). maps: float array (N, H, W).
    Each connected defect region counts equally, however large it is.
    """
    # แปลง masks เป็น bool | input: masks | output: masks = bool (N, H, W)
    masks = np.asarray(masks, dtype=bool)
    # แปลง maps เป็น float32 | input: maps | output: maps = float32 (N, H, W)
    maps = np.asarray(maps, dtype=np.float32)
    # ถ้ารูปร่างของสองอาร์เรย์ไม่เท่ากัน
    if masks.shape != maps.shape:
        # แจ้งข้อผิดพลาด | input: - | output: ยกข้อยกเว้น ValueError
        raise ValueError("masks and maps must have the same shape")
    # ถ้าไม่มีพิกเซลบกพร่องเลย
    if not masks.any():
        # แจ้งข้อผิดพลาด เพราะคำนวณไม่ได้ | input: - | output: ยกข้อยกเว้น ValueError
        raise ValueError("au_pro needs at least one defect pixel")

    # สร้างค่า FPR เรียงจาก 0 ถึง fpr_limit จำนวน steps+1 ค่า | input: fpr_limit, steps | output: fprs = อาร์เรย์ 1 มิติ
    fprs = np.linspace(0.0, fpr_limit, steps + 1)
    # หา threshold ของแต่ละ FPR จากคะแนนของพิกเซลปกติ (นอกบริเวณบกพร่อง) | input: maps[~masks], fprs | output: thresholds = อาร์เรย์ 1 มิติ
    thresholds = np.quantile(maps[~masks], 1.0 - fprs)

    # เตรียม list เก็บสัดส่วนการซ้อนทับของแต่ละบริเวณ | input: - | output: overlaps = []
    overlaps = []
    # วนทีละรูป (mask คู่กับ map)
    for mask, score_map in zip(masks, maps):
        # ติดป้ายแต่ละบริเวณบกพร่องที่เชื่อมกัน | input: mask | output: labels = อาร์เรย์เลขบริเวณ, count = จำนวนบริเวณ
        labels, count = ndimage.label(mask)
        # วนทีละบริเวณ
        for region in range(1, count + 1):
            # เรียงคะแนนของพิกเซลในบริเวณจากน้อยไปมาก | input: score_map, labels | output: scores = อาร์เรย์ 1 มิติ
            scores = np.sort(score_map[labels == region])
            # นับพิกเซลในบริเวณที่คะแนนไม่เกินแต่ละ threshold | input: scores, thresholds | output: below = อาร์เรย์ int
            below = np.searchsorted(scores, thresholds, side="right")
            # สัดส่วนที่ซ้อนทับ = สัดส่วนพิกเซลที่คะแนนสูงกว่า threshold แล้วเก็บลง list | input: below, scores.size | output: overlaps ยาวขึ้น 1 รายการ
            overlaps.append(1.0 - below / scores.size)
    # เฉลี่ยทุกบริเวณให้เท่ากัน ไม่ว่าบริเวณจะใหญ่แค่ไหน | input: overlaps | output: pro = อาร์เรย์ 1 มิติ
    pro = np.mean(overlaps, axis=0)

    # คำนวณพื้นที่ใต้กราฟด้วยกฎสี่เหลี่ยมคางหมู | input: pro, fprs | output: area = float
    area = np.sum((pro[1:] + pro[:-1]) / 2 * np.diff(fprs))
    # หารด้วย fpr_limit ให้เป็นช่วง 0-1 | input: area, fpr_limit | output: float
    return float(area / fpr_limit)


# ฟังก์ชันคำนวณเมตริกระดับรูปอย่างเดียว (สำหรับชุดข้อมูลที่ไม่มี mask) | input: labels = 0/1 (1 = บกพร่อง), scores = คะแนนต่อรูป, threshold = เกณฑ์ | output: dict {image_auroc, threshold, precision, recall, f1, true_positive, false_positive, false_negative, true_negative}
def image_metrics(labels, scores, threshold):
    """Image-level metrics only, for datasets without masks. labels: 1 means defective."""
    # แปลง labels เป็นจำนวนเต็ม | input: labels | output: labels = int array
    labels = np.asarray(labels, dtype=int)
    # แปลง scores เป็น float64 | input: scores | output: scores = float64 array
    scores = np.asarray(scores, dtype=np.float64)
    # ทำนายว่าบกพร่องเมื่อคะแนนเกิน threshold | input: scores, threshold | output: predicted = bool array
    predicted = scores > threshold

    # นับรูปที่ทำนายบกพร่องและบกพร่องจริง | input: predicted, labels | output: true_positive = int
    true_positive = int(np.sum(predicted & (labels == 1)))
    # นับรูปที่ทำนายบกพร่องแต่จริงปกติ | input: predicted, labels | output: false_positive = int
    false_positive = int(np.sum(predicted & (labels == 0)))
    # นับรูปที่ทำนายปกติแต่จริงบกพร่อง | input: predicted, labels | output: false_negative = int
    false_negative = int(np.sum(~predicted & (labels == 1)))
    # นับรูปที่ทำนายปกติและปกติจริง | input: predicted, labels | output: true_negative = int
    true_negative = int(np.sum(~predicted & (labels == 0)))

    # precision = TP / (TP + FP) (กันหารศูนย์ด้วย max กับ 1) | input: true_positive, false_positive | output: float
    precision = true_positive / max(true_positive + false_positive, 1)
    # recall = TP / (TP + FN) (กันหารศูนย์) | input: true_positive, false_negative | output: float
    recall = true_positive / max(true_positive + false_negative, 1)
    # คืนเมตริกทั้งหมดเป็น dict
    return {
        # AUROC ระดับรูป | input: labels, scores | output: float
        "image_auroc": float(roc_auc_score(labels, scores)),
        # threshold ที่ใช้
        "threshold": float(threshold),
        # ความแม่นยำของที่ทำนายว่าบกพร่อง
        "precision": precision,
        # สัดส่วนของที่บกพร่องจริงที่จับได้
        "recall": recall,
        # ค่าเฉลี่ยฮาร์โมนิกของ precision และ recall (กันหารศูนย์ด้วย 1e-12)
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        # จำนวน true positive
        "true_positive": true_positive,
        # จำนวน false positive
        "false_positive": false_positive,
        # จำนวน false negative
        "false_negative": false_negative,
        # จำนวน true negative
        "true_negative": true_negative,
    }


# ฟังก์ชันคำนวณเมตริกระดับรูปและพิกเซลของโมเดลหนึ่งตัว | input: labels = 0/1, scores = คะแนนต่อรูป, masks = bool (N, H, W), maps = float (N, H, W), threshold | output: dict {image_auroc, pixel_auroc, au_pro, ...เมตริกระดับรูปที่เหลือ}
def evaluate(labels, scores, masks, maps, threshold):
    """Image and pixel metrics for one model. labels: 1 means defective."""
    # แปลง masks เป็น bool | input: masks | output: masks = bool array
    masks = np.asarray(masks, dtype=bool)
    # คำนวณเมตริกระดับรูป | input: labels, scores, threshold | output: image = dict
    image = image_metrics(labels, scores, threshold)
    # คืนผลรวม
    return {
        # ดึง image_auroc ออกจาก dict เพื่อจัดลำดับให้มาก่อน | input: image | output: float
        "image_auroc": image.pop("image_auroc"),
        # AUROC ระดับพิกเซล เทียบ mask กับ map ที่คลี่เป็น 1 มิติ | input: masks, maps | output: float
        "pixel_auroc": float(roc_auc_score(masks.ravel(), np.asarray(maps).ravel())),
        # AU-PRO | input: masks, maps | output: float
        "au_pro": au_pro(masks, maps),
        # รวมเมตริกระดับรูปที่เหลือ
        **image,
    }


# ฟังก์ชันคำนวณผลแยกตามชนิดความบกพร่อง โดยเทียบแต่ละชนิดกับรูปปกติ | input: groups = ชื่อกลุ่มต่อรูป ("good" = ปกติ), scores, masks, maps, threshold | output: dict {ชื่อความบกพร่อง: {images, image_auroc, pixel_auroc, au_pro, recall}}
def evaluate_by_defect(groups, scores, masks, maps, threshold):
    """Per-defect results. Each defect is compared against the good images."""
    # แปลง groups เป็นอาร์เรย์ | input: groups | output: groups = array
    groups = np.asarray(groups)
    # แปลง scores เป็น float64 | input: scores | output: scores = float64 array
    scores = np.asarray(scores, dtype=np.float64)
    # แปลง masks เป็น bool | input: masks | output: masks = bool array
    masks = np.asarray(masks, dtype=bool)
    # แปลง maps เป็นอาร์เรย์ | input: maps | output: maps = array
    maps = np.asarray(maps)
    # ตำแหน่งที่เป็นรูปปกติ | input: groups | output: good = bool array
    good = groups == "good"

    # เตรียม dict เก็บผลของแต่ละความบกพร่อง | input: - | output: rows = {}
    rows = {}
    # วนทุกชนิดความบกพร่อง (ตัด "good" ออก) เรียงตามตัวอักษร
    for defect in sorted(set(groups.tolist()) - {"good"}):
        # ตำแหน่งของรูปชนิดนี้ | input: groups, defect | output: selected = bool array
        selected = groups == defect
        # ตำแหน่งของรูปชนิดนี้รวมกับรูปปกติ | input: selected, good | output: both = bool array
        both = selected | good
        # บันทึกผลของชนิดนี้
        rows[defect] = {
            # จำนวนรูปของชนิดนี้
            "images": int(selected.sum()),
            # AUROC ระดับรูป: ชนิดนี้เทียบกับรูปปกติ | input: selected[both], scores[both] | output: float
            "image_auroc": float(roc_auc_score(selected[both], scores[both])),
            # AUROC ระดับพิกเซล เฉพาะรูปในกลุ่มนี้และรูปปกติ | input: masks[both], maps[both] | output: float
            "pixel_auroc": float(roc_auc_score(masks[both].ravel(), maps[both].ravel())),
            # AU-PRO ของกลุ่มนี้ | input: masks[both], maps[both] | output: float
            "au_pro": au_pro(masks[both], maps[both]),
            # recall = สัดส่วนรูปชนิดนี้ที่คะแนนเกิน threshold | input: scores[selected], threshold | output: float
            "recall": float(np.mean(scores[selected] > threshold)),
        }
    # คืนผลทุกชนิด | input: rows | output: dict
    return rows
