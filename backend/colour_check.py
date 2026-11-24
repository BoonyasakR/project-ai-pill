"""Flag pill colours that never appear on normal training pills.

Normal pills are white with dull red specks. Colour defects are stains in new colours (bright red,
pink, yellow, blue), often small or on the rim, where PatchCore at 256 pixels misses them.
"""
# นำเข้า OpenCV: ใช้เบลอรูปและแปลงสี | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: ใช้คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า ndimage ของ SciPy: ใช้กรองค่าสูงสุดและติดป้ายกลุ่มพิกเซล | input: - | output: โมดูล ndimage
from scipy import ndimage

# ความกว้างของแต่ละช่องฮิสโตแกรม = 4 หน่วยสี Lab (สเกล Lab 8 บิตของ OpenCV) | input: - | output: BIN = 4
BIN = 4  # Lab units per histogram bin, on OpenCV's 8-bit Lab scale.
# จำนวนช่องต่อแกนสี = 256 / 4 = 64 | input: BIN | output: BINS = 64
BINS = 256 // BIN
# จำนวนช่องข้างเคียงที่ถือว่า "เคยเห็นแล้ว" ด้วย | input: - | output: TOLERANCE = 1
TOLERANCE = 1  # Neighbouring bins that also count as seen.
# พื้นที่ขั้นต่ำเป็นพิกเซลบนผืนผ้าใบ 800x800; กลุ่มที่เล็กกว่านี้ถือเป็นสัญญาณรบกวน | input: - | output: MIN_AREA = 20
MIN_AREA = 20  # Pixels on the 800 x 800 canvas. Smaller blobs are noise, whatever validation says.


# ฟังก์ชันแปลงสีของทุกพิกเซลเป็นเลขช่อง (bin) ในปริภูมิสี Lab | input: image = อาร์เรย์ RGB รูปร่าง (H, W, 3) | output: อาร์เรย์ int64 รูปร่าง (H, W, 3) เลขช่อง L, a, b ของแต่ละพิกเซล
def colour_bins(image):
    """Quantised Lab colour of each pixel in an RGB array, lightly blurred against noise."""
    # เบลอรูปเบา ๆ ด้วยเคอร์เนล 3x3 ลดสัญญาณรบกวน แล้วแปลง RGB เป็น Lab | input: image | output: lab = อาร์เรย์ uint8 (H, W, 3)
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_RGB2LAB)
    # แปลงเป็น int64 แล้วหารปัดลงด้วย 4 เพื่อจัดค่าสีเข้าช่อง | input: lab | output: อาร์เรย์เลขช่อง (H, W, 3)
    return lab.astype(np.int64) // BIN


# ฟังก์ชันสร้างตารางสีที่ "เคยเห็น" บนเม็ดยาปกติ | input: pills = ตัวแจกแจง (iterable) ของคู่ (รูป RGB, mask เม็ดยา) | output: อาร์เรย์ bool รูปร่าง (64, 64, 64) True = สีนี้เคยเห็น
def build_seen(pills):
    """Return a boolean table of colours seen on normal pills. `pills` yields (RGB array, mask)."""
    # สร้างตารางนับจำนวนพิกเซลต่อช่องสี เริ่มที่ 0 ทั้งหมด | input: BINS | output: counts = int64 (64, 64, 64)
    counts = np.zeros((BINS,) * 3, np.int64)
    # วนอ่านเม็ดยาปกติทีละเม็ด
    for image, mask in pills:
        # หาเลขช่องสีเฉพาะพิกเซลที่อยู่ในเม็ดยา | input: image, mask | output: bins = อาร์เรย์ (N, 3) N = จำนวนพิกเซลในเม็ด
        bins = colour_bins(image)[mask]
        # บวกจำนวนนับ 1 ให้แต่ละช่องสีที่พบ (รองรับช่องซ้ำ) | input: bins | output: counts ถูกอัปเดตในที่เดิม
        np.add.at(counts, (bins[:, 0], bins[:, 1], bins[:, 2]), 1)
    # ช่องที่เคยนับได้ (>0) ถูกขยายออกให้ครอบช่องข้างเคียงด้วยตัวกรองค่าสูงสุด | input: counts | output: อาร์เรย์ bool (64, 64, 64)
    return ndimage.maximum_filter(counts > 0, size=2 * TOLERANCE + 1)


# ฟังก์ชันวัดพื้นที่กลุ่มสีแปลกปลอมที่ใหญ่ที่สุดในเม็ดยา | input: seen = ตารางสีที่เคยเห็น, image = รูป RGB (H, W, 3), mask = mask เม็ดยา bool (H, W) | output: (พื้นที่กลุ่มใหญ่สุดเป็นจำนวนพิกเซล int, mask พิกเซลสีแปลกทั้งหมด bool (H, W))
def novel_colour(seen, image, mask):
    """Return the area of the largest unseen-colour blob inside the pill, and all unseen pixels."""
    # หาเลขช่องสีของทุกพิกเซลในรูป | input: image | output: bins (H, W, 3)
    bins = colour_bins(image)
    # พิกเซลที่ช่องสีไม่เคยเห็น (กลับค่าด้วย ~) และอยู่ในเม็ดยา | input: seen, bins, mask | output: novel = bool (H, W)
    novel = ~seen[bins[..., 0], bins[..., 1], bins[..., 2]] & mask
    # ติดป้ายหมายเลขให้พิกเซลสีแปลกที่ติดกันเป็นกลุ่มเดียวกัน | input: novel | output: labels = อาร์เรย์เลขกลุ่ม, count = จำนวนกลุ่ม
    labels, count = ndimage.label(novel)
    # ถ้ามีกลุ่ม: นับพิกเซลของแต่ละกลุ่ม (ตัดกลุ่ม 0 = พื้นหลังออก) แล้วเอากลุ่มใหญ่สุด; ถ้าไม่มีให้เป็น 0 | input: labels, count | output: (พื้นที่ int, novel)
    return (int(np.bincount(labels.ravel())[1:].max()) if count else 0), novel
