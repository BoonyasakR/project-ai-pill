"""Find separate pills on a dark background and align each one like the training images."""
# นำเข้า OpenCV: ประมวลผลรูป (เบลอ, threshold, หมุน, วาด) | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np
# นำเข้า ndimage ของ SciPy: เติมรู, ระยะทาง, ติดป้ายกลุ่ม | input: - | output: โมดูล ndimage
from scipy import ndimage
# นำเข้า peak_local_max: หาจุดยอดเฉพาะที่ของภาพระยะทาง | input: - | output: ฟังก์ชัน peak_local_max
from skimage.feature import peak_local_max
# นำเข้า watershed: แยกวัตถุที่ติดกันออกจากกัน | input: - | output: ฟังก์ชัน watershed
from skimage.segmentation import watershed

# ขนาดผืนผ้าใบรูปฝึก = 800x800 พิกเซล | input: - | output: CANVAS = 800
CANVAS = 800  # Training images are 800 x 800.
# ความยาวแกนยาวมัธยฐานของเม็ดยาในชุดฝึก (พิกเซล) | input: - | output: PILL_LENGTH = 693
PILL_LENGTH = 693  # Median long-axis length of a training pill, in pixels.
# สีพื้นหลังมัธยฐานของรูปฝึก ในลำดับ BGR | input: - | output: BACKGROUND_BGR = (26, 15, 15)
BACKGROUND_BGR = (26, 15, 15)  # Median training background.
# กลุ่มพิกเซลที่เล็กกว่า 20% ของกลุ่มใหญ่สุดจะไม่ถูกตรวจเป็นเม็ดยา | input: - | output: MIN_RELATIVE_AREA = 0.2
MIN_RELATIVE_AREA = 0.2  # Smaller blobs are not inspected as pills.
# กลุ่มที่เล็กกว่า 1% ของกลุ่มใหญ่สุดคือเศษผง ไม่ต้องเตือน | input: - | output: MIN_WARNED_AREA = 0.01
MIN_WARNED_AREA = 0.01  # Blobs below this share of the largest are specks. No warning needed.
# ความยาวขั้นต่ำ (พิกเซล) เม็ดปกติที่สั้นกว่านี้จะแจ้งผิดเมื่อขยายภาพ | input: - | output: MIN_RELIABLE_LENGTH = 180
MIN_RELIABLE_LENGTH = 180  # Pixels. Shorter normal pills raise false alarms once enlarged.
# เก็บพื้นที่รอบ mask ไว้ 2% ของความยาวเม็ดยา | input: - | output: MASK_MARGIN = 0.02
MASK_MARGIN = 0.02  # Keep this share of the pill length around the mask.
# Widest pose that still counts as a training image. Every MVTec pill fits:
# at most 1.3 degrees of tilt, 1.9% size change, and 6 pixels off centre.
# เอียงได้มากสุด 2 องศาจึงยังนับเป็นท่าเดียวกับรูปฝึก | input: - | output: MAX_TILT = 2
MAX_TILT = 2  # Degrees.
# ขนาดเปลี่ยนได้มากสุด 2.5% | input: - | output: MAX_SIZE_CHANGE = 0.025
MAX_SIZE_CHANGE = 0.025
# เลื่อนจากกึ่งกลางได้มากสุด 10 พิกเซลบนผืนผ้าใบฝึก | input: - | output: MAX_SHIFT = 10
MAX_SHIFT = 10  # Pixels on the training canvas.


# ฟังก์ชันหาเม็ดยาทั้งหมดในรูป | input: image = อาร์เรย์ RGB (H, W, 3) | output: list ของ mask bool (H, W) หนึ่งอันต่อเม็ด เรียงจากใหญ่ไปเล็ก
def find_pills(image):
    """Return one boolean mask per pill in an RGB array, largest first."""
    # หากลุ่มวัตถุสว่างทั้งหมด แล้วทิ้งกลุ่มที่เล็กเกินไป | input: image | output: list ของ mask เม็ดยา
    return keep_pills(find_blobs(image))


# ฟังก์ชันกรองกลุ่มที่เล็กกว่ากลุ่มใหญ่สุดมากทิ้ง | input: blobs = list ของ mask เรียงใหญ่ไปเล็ก (ต้องไม่ว่าง) | output: list ของ mask ที่มีพื้นที่ >= 20% ของกลุ่มใหญ่สุด
def keep_pills(blobs):
    """Drop blobs far smaller than the largest one. `blobs` must be largest first."""
    # เก็บเฉพาะ mask ที่จำนวนพิกเซลไม่ต่ำกว่าเกณฑ์สัมพัทธ์ เทียบกับ mask แรก (ใหญ่สุด) | input: blobs | output: list ของ mask ที่ผ่านเกณฑ์
    return [blob for blob in blobs if blob.sum() >= MIN_RELATIVE_AREA * blobs[0].sum()]


# ฟังก์ชันแยกวัตถุสว่างแต่ละชิ้นออกจากพื้นหลังมืด | input: image = อาร์เรย์ RGB (H, W, 3) | output: list ของ mask bool (H, W) เรียงจากใหญ่ไปเล็ก (ว่างถ้าไม่พบวัตถุ)
def find_blobs(image):
    """Return one boolean mask per separate bright object, largest first.

    ponytail: Otsu needs bright pills on a dark background. Use a trained
    instance segmenter if pills ever overlap or sit on a light surface.
    """
    # แปลงรูปเป็นขาวดำ แล้วเบลอ 5x5 ลดสัญญาณรบกวน | input: image | output: gray = อาร์เรย์ uint8 (H, W)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    # แปลงเป็นภาพขาว-ดำล้วนด้วยวิธี Otsu (เลือกค่าตัดอัตโนมัติ) | input: gray | output: binary = อาร์เรย์ uint8 ค่า 0 หรือ 255 (ตัวแรกที่คืนคือค่าตัดที่ไม่ใช้ จึงรับด้วย _)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ลบจุดขาวเล็ก ๆ ด้วย morphological opening เคอร์เนล 5x5 | input: binary | output: binary ที่สะอาดขึ้น
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # เติมรูในวัตถุให้ทึบ แล้วแปลงเป็น bool | input: binary | output: foreground = bool (H, W)
    foreground = ndimage.binary_fill_holes(binary > 0)
    # ถ้าไม่มีพิกเซลวัตถุเลย
    if not foreground.any():
        # คืน list ว่าง | input: - | output: []
        return []

    # คำนวณระยะจากแต่ละพิกเซลวัตถุไปถึงขอบพื้นหลังที่ใกล้ที่สุด | input: foreground | output: distance = float (H, W)
    distance = ndimage.distance_transform_edt(foreground)
    # A pill's distance ridge is one peak. Touching pills give one peak each.
    # เบลอภาพระยะทางให้เรียบ (sigma=3) เพื่อให้แต่ละเม็ดมียอดเดียว | input: distance | output: smooth = float32 (H, W)
    smooth = cv2.GaussianBlur(distance.astype(np.float32), (0, 0), 3)
    # ติดป้ายหมายเลขให้วัตถุที่ติดกันเป็นก้อนเดียว | input: foreground | output: blobs = อาร์เรย์เลขก้อน (ตัวที่สองคือจำนวนก้อน ไม่ใช้)
    blobs, _ = ndimage.label(foreground)
    # ระยะทางสูงสุด = ครึ่งความกว้างของส่วนที่หนาที่สุด | input: distance | output: widest = float
    widest = distance.max()
    # หาจุดยอดเฉพาะที่ (ศูนย์กลางเม็ดยา) โดยยอดต้องห่างกันอย่างน้อย 80% ของ widest และสูงกว่า 40% ของ widest | input: smooth, blobs | output: peaks = อาร์เรย์ (จำนวนยอด, 2) พิกัด (แถว, คอลัมน์)
    peaks = peak_local_max(
        smooth, min_distance=max(int(widest * 0.8), 5), threshold_abs=widest * 0.4, labels=blobs
    )
    # สร้างอาร์เรย์ marker ว่าง (ค่า 0) ขนาดเท่ารูป | input: foreground.shape | output: markers = int32 (H, W)
    markers = np.zeros(foreground.shape, np.int32)
    # ใส่หมายเลข 1..N ลงที่ตำแหน่งของแต่ละยอด | input: peaks | output: markers ที่มีหมายเลขที่ยอด
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # แยกวัตถุด้วย watershed จาก marker (กลับเครื่องหมายเพื่อให้ยอดกลายเป็นแอ่ง) จำกัดในพื้นที่วัตถุ | input: smooth, markers, foreground | output: labels = อาร์เรย์เลขเม็ดยา (0 = พื้นหลัง/ไม่มีเจ้าของ)
    labels = watershed(-smooth, markers, mask=foreground)

    # สร้าง mask bool ของแต่ละเม็ดจากหมายเลข 1..max | input: labels | output: masks = list ของ bool (H, W)
    masks = [labels == index for index in range(1, labels.max() + 1)]
    # Objects too thin to give a peak get no watershed marker, so add them back.
    # หาวัตถุที่ไม่มี marker (บางเกินจนไม่มียอด) และติดป้ายแยกกัน | input: foreground, labels | output: unmarked = อาร์เรย์เลขกลุ่ม, count = จำนวนกลุ่ม
    unmarked, count = ndimage.label(foreground & (labels == 0))
    # เพิ่ม mask ของวัตถุที่ตกหล่นกลับเข้า list | input: unmarked, count | output: masks ยาวขึ้น
    masks += [unmarked == index for index in range(1, count + 1)]
    # เรียงตามจำนวนพิกเซลจากมากไปน้อย แล้วคืนค่า | input: masks | output: list ของ mask เรียงใหญ่ไปเล็ก
    return sorted(masks, key=np.sum, reverse=True)


# ฟังก์ชันหาท่าวางของเม็ดยาจาก mask | input: mask = bool (H, W) | output: tuple (จุดกึ่งกลาง [x, y], มุมแกนยาวเป็นองศา float, ความยาวแกนยาวเป็นพิกเซล float)
def pill_pose(mask):
    """Return the centre, long-axis angle in degrees, and long-axis length of a mask."""
    # หาพิกัดแถว (ys) และคอลัมน์ (xs) ของทุกพิกเซลใน mask | input: mask | output: ys, xs = อาร์เรย์ 1 มิติ
    ys, xs = np.nonzero(mask)
    # รวมเป็นจุด (x, y) ทุกจุด ชนิด float64 | input: xs, ys | output: points = อาร์เรย์ (N, 2)
    points = np.column_stack([xs, ys]).astype(np.float64)
    # หาจุดกึ่งกลางเป็นค่าเฉลี่ยของทุกจุด | input: points | output: centre = อาร์เรย์ [x, y]
    centre = points.mean(axis=0)
    # หา eigenvector ของ covariance ของจุด (ตัวสุดท้ายคือทิศที่กระจายมากสุด = แกนยาว) | input: points, centre | output: vectors = เมทริกซ์ 2x2 (ค่า eigenvalue ตัวแรกไม่ใช้)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    # เลือก eigenvector ตัวที่ใหญ่สุดเป็นทิศแกนยาว | input: vectors | output: axis = อาร์เรย์ [dx, dy]
    axis = vectors[:, 1]
    # ความยาวแกนยาว = ช่วงห่าง (max-min) ของจุดที่ฉายลงบนแกน | input: points, centre, axis | output: length = float
    length = np.ptp((points - centre) @ axis)
    # คืนจุดกึ่งกลาง, มุมของแกน (เรเดียน -> องศา) และความยาว | input: centre, axis, length | output: (centre, มุมองศา, ความยาว)
    return centre, float(np.degrees(np.arctan2(axis[1], axis[0]))), float(length)


# ฟังก์ชันตรวจว่ารูปเม็ดเดียววางท่าเหมือนรูปฝึกอยู่แล้วหรือไม่ | input: mask = bool (H, W) ของเม็ดยาเม็ดเดียว | output: bool (True = ท่าตรงกับรูปฝึก)
def in_training_pose(mask):
    """Return whether a one-pill image already sits like a training image.

    The long axis has no direction, so an upside-down pill also passes.
    """
    # อ่านความสูงและความกว้างของรูป | input: mask | output: height, width
    height, width = mask.shape
    # รูปที่ไม่เป็นสี่เหลี่ยมจัตุรัสไม่ตรงกับรูปฝึกแน่นอน
    if height != width:
        # คืน False | input: - | output: False
        return False
    # หาท่าวางของเม็ดยา | input: mask | output: centre, angle, length
    centre, angle, length = pill_pose(mask)
    # อัตราส่วนปรับขนาดจากรูปนี้ไปเป็นผืนผ้าใบฝึก | input: width | output: scale = float
    scale = CANVAS / width
    # มุมเอียงจากแนวนอน (ค่าสัมบูรณ์ ปรับให้อยู่ในช่วง 0-90 องศา) | input: angle | output: tilt = float
    tilt = abs((angle + 90) % 180 - 90)
    # ระยะที่จุดกึ่งกลางเม็ดเลื่อนจากกึ่งกลางผืนผ้าใบ (หน่วยพิกเซลบนผืนผ้าใบฝึก) | input: centre, scale | output: shift = float
    shift = np.hypot(*(centre * scale - CANVAS / 2))
    # คืน True เมื่อเอียง, ขนาด และตำแหน่งอยู่ในเกณฑ์ทั้งสามข้อ | input: tilt, length, shift | output: bool
    return tilt <= MAX_TILT and abs(length * scale / PILL_LENGTH - 1) <= MAX_SIZE_CHANGE and shift <= MAX_SHIFT


# ฟังก์ชันหมุน ย่อ/ขยาย และจัดเม็ดยาให้อยู่กึ่งกลางผืนผ้าใบแบบรูปฝึก | input: image = RGB (H, W, 3), mask = mask เม็ดยา, others = mask เม็ดข้างเคียง (หรือ None), flip = หมุนเพิ่ม 180 องศา, canvas = ขนาดผืนผ้าใบ, pill_length = ความยาวเป้าหมาย, rotate = หมุนตามแกนยาวหรือไม่ | output: tuple (รูปที่จัดแล้ว uint8 (canvas, canvas, 3), mask เม็ดยาบนรูปนั้น bool)
def align_pill(image, mask, others=None, flip=False, canvas=CANVAS, pill_length=PILL_LENGTH, rotate=True):
    """Rotate, scale, and centre one pill on a training-style canvas.

    `others` marks neighbouring pills, which are removed. The long axis has no
    direction, so `flip` gives the other 180-degree option. Round pills pass
    `rotate=False` and keep their angle. Returns the aligned image and a boolean
    mask of the pill on it.
    """
    # หาท่าวางเดิมของเม็ดยา | input: mask | output: centre, angle, length
    centre, angle, length = pill_pose(mask)
    # มุมที่ต้องหมุน = มุมแกนยาว (ถ้า rotate) บวก 180 องศา (ถ้า flip) | input: angle, rotate, flip | output: turn = องศา
    turn = (angle if rotate else 0) + (180 if flip else 0)
    # สร้างเมทริกซ์หมุนรอบจุดกึ่งกลางพร้อมปรับขนาดให้ความยาวเท่าเป้าหมาย | input: centre, turn, pill_length/length | output: matrix = เมทริกซ์ 2x3
    matrix = cv2.getRotationMatrix2D(tuple(centre), turn, pill_length / length)
    # เลื่อนให้จุดกึ่งกลางเม็ดไปอยู่กึ่งกลางผืนผ้าใบ | input: matrix, canvas, centre | output: matrix ที่ถูกปรับค่าเลื่อน
    matrix[:, 2] += canvas / 2 - centre
    # ขนาดผืนผ้าใบปลายทาง (กว้าง, สูง) | input: canvas | output: size = tuple
    size = (canvas, canvas)
    # ย้ายรูปไปยังผืนผ้าใบด้วย matrix (ประมาณค่าแบบ cubic) | input: image, matrix, size | output: pill = รูป (canvas, canvas, 3)
    pill = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_CUBIC)
    # ย้าย mask ด้วย matrix เดียวกัน (ประมาณค่าแบบจุดใกล้สุดเพื่อให้ยังเป็น 0/1) | input: mask, matrix, size | output: aligned_mask = uint8 (canvas, canvas)
    aligned_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
    # ขนาดเคอร์เนลสำหรับขยาย mask = 2% ของความยาว ขยายสองข้าง +1 ให้เป็นเลขคี่ | input: pill_length | output: margin = int
    margin = int(pill_length * MASK_MARGIN) * 2 + 1
    # ขยาย mask ออกเล็กน้อยเพื่อเก็บขอบเม็ดยา | input: aligned_mask, margin | output: keep = uint8 (canvas, canvas) พื้นที่ที่จะเก็บ
    keep = cv2.dilate(aligned_mask, np.ones((margin, margin), np.uint8))
    # ถ้ามีเม็ดข้างเคียงให้ลบออก
    if others is not None:
        # ย้าย mask เม็ดข้างเคียงด้วย matrix เดียวกัน | input: others, matrix, size | output: neighbours = uint8 (canvas, canvas)
        neighbours = cv2.warpAffine(others.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
        # ขยายเม็ดข้างเคียงอีก 5x5 แล้วตั้งพื้นที่นั้นใน keep เป็น 0 (ตัดทิ้ง) | input: neighbours | output: keep ที่ตัดเม็ดข้างเคียงออก
        keep[cv2.dilate(neighbours, np.ones((5, 5), np.uint8)) > 0] = 0
    # เบลอขอบ keep ให้นุ่ม แล้วเพิ่มมิติสุดท้ายเพื่อคูณกับรูปสีได้ | input: keep | output: keep = float32 (canvas, canvas, 1) ค่า 0..1
    keep = cv2.GaussianBlur(keep.astype(np.float32), (9, 9), 0)[..., None]
    # สีพื้นหลังฝึก กลับลำดับ BGR เป็น RGB | input: BACKGROUND_BGR | output: background = float32 [R, G, B]
    background = np.array(BACKGROUND_BGR[::-1], np.float32)
    # ผสมรูปกับสีพื้นหลังตามน้ำหนัก keep แล้วจำกัดค่าให้อยู่ใน 0-255 | input: pill, keep, background | output: aligned = uint8 (canvas, canvas, 3)
    aligned = (pill * keep + background * (1 - keep)).clip(0, 255).astype(np.uint8)
    # คืนรูปที่จัดแล้วและ mask เม็ดยา (แปลงเป็น bool) | input: aligned, aligned_mask | output: (aligned, bool mask)
    return aligned, aligned_mask > 0


# ฟังก์ชันหาเม็ดยาสีชมพูในรูป pillQC | input: image = RGB (H, W, 3) | output: mask bool (H, W) ของเม็ดยาสีชมพู
def pink_pill_mask(image):
    """The pink pill in a pillQC crop. Dirt on the pill stays inside; dirt on the floor does not."""
    # แยกช่องแดงและเขียวเป็น int16 เพื่อไม่ให้ค่าล้นตอนลบ | input: image | output: red, green = อาร์เรย์ (H, W)
    red, green = image[..., 0].astype(np.int16), image[..., 1].astype(np.int16)
    # พิกเซลที่แดงมาก (>150) และแดงเกินเขียวอย่างน้อย 10 ถือว่าชมพู | input: red, green | output: pink = uint8 (H, W) ค่า 0/1
    pink = ((red > 150) & (red > green + 10)).astype(np.uint8)
    # หาเส้นขอบนอกสุดของกลุ่มสีชมพูทั้งหมด | input: pink | output: contours = list ของเส้นขอบ (ตัวที่สองไม่ใช้)
    contours, _ = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # สร้าง mask ว่างขนาดเท่ารูป | input: pink | output: mask = uint8 (H, W) ค่า 0
    mask = np.zeros_like(pink)
    # วาดเส้นขอบที่ใหญ่ที่สุดเป็นพื้นที่ทึบ (ค่า 1) เพื่อรวมรอยเปื้อนในเม็ด แต่ไม่รวมสิ่งสกปรกบนพื้น | input: contours | output: mask ที่มีเม็ดยาเป็น 1
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    # แปลงเป็น bool แล้วคืนค่า | input: mask | output: bool (H, W)
    return mask > 0


# ตัวคูณขยายรัศมีวงกลม 5% เพื่อเก็บขอบที่มักมีรอยบิ่น แต่ไม่รวมสิ่งสกปรกบนพื้นที่ติดเม็ด | input: - | output: ROUND_PILL_GROWTH = 1.05
ROUND_PILL_GROWTH = 1.05  # Keeps the rim, where chips are, but not floor dirt touching the pill.


# ฟังก์ชันสร้างพื้นที่วงกลมครอบเม็ดยากลม | input: mask = bool/uint8 (H, W) ของเม็ดยา | output: bool (H, W) พื้นที่วงกลม
def round_pill_region(mask):
    """A disk over a round pill: the largest circle inside the mask, grown by 5%.

    White dirt on the floor can touch the pill and join its mask. The inscribed
    circle ignores such bumps, so only the pill itself is scored.
    """
    # แปลง mask เป็น uint8 ตามที่ OpenCV ต้องการ | input: mask | output: mask = uint8 (H, W)
    mask = mask.astype(np.uint8)
    # คำนวณระยะจากแต่ละพิกเซลในเม็ดไปถึงขอบ mask ที่ใกล้ที่สุด | input: mask | output: distance = float32 (H, W)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # หาตำแหน่ง (แถว, คอลัมน์) ที่ระยะมากสุด = ศูนย์กลางวงกลมใหญ่สุดที่อยู่ในเม็ด | input: distance | output: y, x = int
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    # สร้างภาพว่างสำหรับวาดวงกลม | input: mask | output: disk = uint8 (H, W) ค่า 0
    disk = np.zeros_like(mask)
    # วาดวงกลมทึบ (ค่า 1) ที่ศูนย์กลางนั้น รัศมี = ระยะมากสุด x 1.05 | input: x, y, distance.max() | output: disk ที่มีวงกลมเป็น 1
    cv2.circle(disk, (int(x), int(y)), int(distance.max() * ROUND_PILL_GROWTH), 1, -1)
    # แปลงเป็น bool แล้วคืนค่า | input: disk | output: bool (H, W)
    return disk > 0
