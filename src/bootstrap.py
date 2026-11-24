"""Bootstrap confidence intervals for image-level anomaly detection metrics.

Images are resampled with replacement, normal and defective separately, so every
resample keeps both classes. The threshold is held fixed, so the intervals do not
include the uncertainty of the threshold itself.
"""
# นำเข้า NumPy: สุ่มและคำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า roc_auc_score: คำนวณ AUROC | input: - | output: ฟังก์ชัน roc_auc_score
from sklearn.metrics import roc_auc_score

# นำเข้าโมดูลเมตริกของโปรเจกต์ | input: - | output: โมดูล ad_metrics
import ad_metrics

# รายชื่อเมตริกที่จะหาช่วงความเชื่อมั่น | input: - | output: METRICS = tuple ของชื่อเมตริก
METRICS = ("image_auroc", "precision", "recall", "f1")


# ตัวสร้าง (generator) ชุดดัชนีที่สุ่มซ้ำแบบ bootstrap แยกปกติกับบกพร่อง | input: labels = อาร์เรย์ 0/1, rounds = จำนวนรอบ, seed = เมล็ดสุ่ม | output: ผลิตอาร์เรย์ดัชนีทีละรอบ (ความยาวเท่า labels)
def _resamples(labels, rounds, seed):
    # หาดัชนีรูปปกติ (0) และรูปบกพร่อง (1) | input: labels | output: normal, defect = อาร์เรย์ดัชนี
    normal, defect = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    # ถ้าขาดคลาสใดคลาสหนึ่ง
    if normal.size == 0 or defect.size == 0:
        # แจ้งข้อผิดพลาด เพราะ AUROC คำนวณไม่ได้ | input: - | output: ยกข้อยกเว้น ValueError
        raise ValueError("bootstrap needs both normal and defective images")
    # สร้างตัวสุ่มที่ทำซ้ำผลได้ตาม seed | input: seed | output: rng
    rng = np.random.default_rng(seed)
    # วนตามจำนวนรอบ
    for _ in range(rounds):
        # สุ่มเลือกดัชนีแบบใส่คืนจากแต่ละคลาสให้ได้จำนวนเท่าเดิม แล้วต่อกัน แล้วส่งออก | input: normal, defect | output: อาร์เรย์ดัชนีหนึ่งชุด
        yield np.concatenate([rng.choice(normal, normal.size), rng.choice(defect, defect.size)])


# ฟังก์ชันสร้างช่วงความเชื่อมั่นแบบเปอร์เซ็นไทล์ | input: value = ค่าจริง, draws = ค่าจากการสุ่มซ้ำ, level = ระดับความเชื่อมั่น | output: dict {value, low, high}
def _interval(value, draws, level):
    # ความน่าจะเป็นของหางแต่ละข้าง | input: level | output: tail = float
    tail = (1 - level) / 2
    # หาค่าควอนไทล์ล่างและบนของค่าที่สุ่มได้ | input: draws, tail | output: low, high
    low, high = np.quantile(draws, [tail, 1 - tail])
    # คืนค่าจริงและขอบเขตช่วง | input: value, low, high | output: dict
    return {"value": float(value), "low": float(low), "high": float(high)}


# ฟังก์ชันหาช่วงความเชื่อมั่นของเมตริกระดับรูปแต่ละตัว | input: labels = 0/1 (1 = บกพร่อง), scores, threshold, rounds, seed, level | output: dict {ชื่อเมตริก: {value, low, high}}
def image_ci(labels, scores, threshold, rounds=2000, seed=42, level=0.95):
    """Percentile interval of each image metric. labels: 1 means defective."""
    # แปลง labels เป็นจำนวนเต็ม | input: labels | output: labels = int array
    labels = np.asarray(labels, dtype=int)
    # แปลง scores เป็น float64 | input: scores | output: scores = float64 array
    scores = np.asarray(scores, dtype=np.float64)
    # คำนวณเมตริกจากข้อมูลจริง | input: labels, scores, threshold | output: point = dict
    point = ad_metrics.image_metrics(labels, scores, threshold)
    # คำนวณเมตริกซ้ำสำหรับทุกชุดที่สุ่ม
    draws = [
        # เมตริกของชุดสุ่มหนึ่งชุด | input: labels[index], scores[index], threshold | output: dict
        ad_metrics.image_metrics(labels[index], scores[index], threshold)
        # วนทุกชุดดัชนีที่สุ่ม
        for index in _resamples(labels, rounds, seed)
    ]
    # สร้างช่วงความเชื่อมั่นของแต่ละเมตริกใน METRICS | input: point, draws, level | output: dict
    return {name: _interval(point[name], [draw[name] for draw in draws], level) for name in METRICS}


# ฟังก์ชันหาช่วงความเชื่อมั่นของผลต่าง AUROC ระหว่างโมเดล a กับ b บนชุดสุ่มเดียวกัน | input: labels, scores_a, scores_b, rounds, seed, level | output: dict {value, low, high}
def paired_auroc_diff_ci(labels, scores_a, scores_b, rounds=2000, seed=42, level=0.95):
    """Interval of image AUROC(a) minus AUROC(b), both scored on the same resamples."""
    # แปลง labels เป็นจำนวนเต็ม | input: labels | output: labels = int array
    labels = np.asarray(labels, dtype=int)
    # แปลงคะแนนของโมเดล a เป็น float64 | input: scores_a | output: scores_a = float64 array
    scores_a = np.asarray(scores_a, dtype=np.float64)
    # แปลงคะแนนของโมเดล b เป็น float64 | input: scores_b | output: scores_b = float64 array
    scores_b = np.asarray(scores_b, dtype=np.float64)
    # ผลต่าง AUROC จากข้อมูลจริง | input: labels, scores_a, scores_b | output: point = float
    point = roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b)
    # คำนวณผลต่าง AUROC ซ้ำสำหรับทุกชุดที่สุ่ม
    draws = [
        # ผลต่าง AUROC ของชุดสุ่มหนึ่งชุด | input: labels[index], scores_a[index], scores_b[index] | output: float
        roc_auc_score(labels[index], scores_a[index]) - roc_auc_score(labels[index], scores_b[index])
        # วนทุกชุดดัชนีที่สุ่ม
        for index in _resamples(labels, rounds, seed)
    ]
    # สร้างช่วงความเชื่อมั่นของผลต่าง | input: point, draws, level | output: dict {value, low, high}
    return _interval(point, draws, level)
