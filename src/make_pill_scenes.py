"""Paste real MVTec pills into simulated multi-pill scenes with known labels."""
# นำเข้า OpenCV: หมุน/ปรับขนาด/เบลอรูป | input: - | output: โมดูล cv2
import cv2
# นำเข้า NumPy: คำนวณอาร์เรย์ | input: - | output: โมดูล np
import numpy as np

# ขนาดฉาก (สูง, กว้าง) | input: - | output: SCENE_SIZE = (1000, 1400)
SCENE_SIZE = (1000, 1400)  # Height, width.
# จำนวนเม็ดยาต่อฉาก | input: - | output: PILLS_PER_SCENE = 8
PILLS_PER_SCENE = 8
# ช่วงความยาวเม็ดยาที่สุ่ม (พิกเซล) | input: - | output: PILL_LENGTH_RANGE = (240, 330)
PILL_LENGTH_RANGE = (240, 330)
# ช่วงมุมหมุนที่สุ่ม (องศา) | input: - | output: TURN_RANGE = (0, 360)
TURN_RANGE = (0, 360)  # Degrees.
# สีพื้นหลัง RGB | input: - | output: BACKGROUND_RGB = (15, 15, 26)
BACKGROUND_RGB = (15, 15, 26)
# ส่วนเบี่ยงเบนมาตรฐานของสัญญาณรบกวนพื้นหลัง | input: - | output: BACKGROUND_NOISE = 3.0
BACKGROUND_NOISE = 3.0


# ฟังก์ชันตัดเม็ดยาออกมาเป็นรูป RGBA ที่หมุนและปรับขนาดแบบสุ่ม | input: image = RGB (H, W, 3), mask = mask เม็ดยา bool (H, W), rng = ตัวสุ่มของ NumPy | output: อาร์เรย์ uint8 (side, side, 4) รูป RGBA ของเม็ดที่ตัดแล้ว
def cut_pill(image, mask, rng):
    """Return a randomly rotated and scaled RGBA cut-out of one pill."""
    # หาพิกัดแถว/คอลัมน์ของพิกเซลเม็ดยา | input: mask | output: ys, xs
    ys, xs = np.nonzero(mask)
    # ความยาวเม็ด = ความกว้างตามแนวนอน (รูปฝึกวางเม็ดแนวนอนอยู่แล้ว) | input: xs | output: length = int
    length = np.ptp(xs)
    # อัตราส่วนปรับขนาด = ความยาวเป้าหมายที่สุ่มในช่วง ÷ ความยาวจริง | input: PILL_LENGTH_RANGE, length | output: scale = float
    scale = rng.uniform(*PILL_LENGTH_RANGE) / length
    # สร้างช่องอัลฟา: ขยาย mask 9x9 แล้วเบลอ 7x7 ให้ขอบนุ่ม | input: mask | output: alpha = uint8 (H, W)
    alpha = cv2.GaussianBlur(cv2.dilate(mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)), (7, 7), 0)
    # ต่อช่องอัลฟาเป็นช่องที่ 4 ของรูป | input: image, alpha | output: rgba = (H, W, 4)
    rgba = np.dstack([image, alpha])
    # ขนาดด้านของรูปผลลัพธ์ = ความยาวหลังปรับ x 1.2 เผื่อตอนหมุน | input: length, scale | output: side = int
    side = int(length * scale * 1.2)
    # สร้างเมทริกซ์หมุนรอบจุดกึ่งกลางเม็ดด้วยมุมสุ่มและปรับขนาด | input: ศูนย์กลาง, มุมสุ่ม, scale | output: matrix = 2x3
    matrix = cv2.getRotationMatrix2D((float(xs.mean()), float(ys.mean())), rng.uniform(*TURN_RANGE), scale)
    # เลื่อนให้จุดกึ่งกลางเม็ดไปอยู่กึ่งกลางรูปผลลัพธ์ | input: side, xs, ys | output: matrix ที่ปรับแล้ว
    matrix[:, 2] += side / 2 - np.array([xs.mean(), ys.mean()])
    # ย้ายรูป RGBA ไปยังผืนผ้าใบ side x side แล้วคืนค่า | input: rgba, matrix, side | output: อาร์เรย์ (side, side, 4)
    return cv2.warpAffine(rgba, matrix, (side, side), flags=cv2.INTER_AREA)


# ฟังก์ชันวางเม็ดยาที่ตัดแล้วลงฉากโดยไม่ทับกัน | input: cutouts = list ของรูป RGBA, rng = ตัวสุ่ม, touching = ให้เม็ดแตะกันหรือไม่ | output: tuple (ฉาก uint8 (H, W, 3), list ของ mask bool (H, W) ต่อเม็ด)
def compose_scene(cutouts, rng, touching=False):
    """Place cut-outs without overlap. Return the scene and one mask per pill.

    With `touching`, each pill slides toward the centre until it hits another pill.
    """
    # แยกความสูงและความกว้างของฉาก | input: SCENE_SIZE | output: height, width
    height, width = SCENE_SIZE
    # สร้างสัญญาณรบกวนแบบปกติทั่วฉาก | input: rng, BACKGROUND_NOISE | output: noise = float (H, W, 3)
    noise = rng.normal(0, BACKGROUND_NOISE, (height, width, 3))
    # พื้นหลังสีคงที่บวกสัญญาณรบกวน จำกัดค่า 0-255 | input: BACKGROUND_RGB, noise | output: scene = float (H, W, 3)
    scene = (np.array(BACKGROUND_RGB, np.float32) + noise).clip(0, 255)
    # แผนที่บอกว่าพิกเซลไหนมีเม็ดยาแล้ว | input: SCENE_SIZE | output: occupied = bool (H, W)
    occupied = np.zeros(SCENE_SIZE, bool)
    # list เก็บ mask ของเม็ดที่วางแล้ว | input: - | output: masks = []
    masks = []
    # วนวางทีละเม็ด
    for cutout in cutouts:
        # พื้นที่ทึบ = พิกเซลที่อัลฟามากกว่า 127 | input: cutout | output: solid = bool
        solid = cutout[..., 3] > 127
        # หาแถว/คอลัมน์ของพื้นที่ทึบ | input: solid | output: rows, cols
        rows, cols = np.nonzero(solid)
        # ตัดขอบว่างของ mask ให้พอดีเม็ด | input: solid, rows, cols | output: solid = bool ที่ตัดแล้ว
        solid = solid[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        # ตัดรูป RGBA ให้พอดีเม็ดเท่ากัน | input: cutout, rows, cols | output: piece = (h, w, 4)
        piece = cutout[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        # ลองสุ่มตำแหน่งสูงสุด 2000 ครั้ง
        for _ in range(2000):
            # สุ่มขอบบน | input: height, solid | output: top = int
            top = int(rng.integers(0, height - solid.shape[0]))
            # สุ่มขอบซ้าย | input: width, solid | output: left = int
            left = int(rng.integers(0, width - solid.shape[1]))
            # ถ้าตำแหน่งนี้ว่าง (เว้นระยะ 14 พิกเซล หรือ 0 ถ้าให้แตะกัน) ก็ใช้ตำแหน่งนี้
            if free(occupied, solid, top, left, 0 if touching else 14):
                # ออกจากลูปสุ่ม | input: - | output: -
                break
        # ถ้าสุ่มครบ 2000 ครั้งแล้วยังไม่ได้ตำแหน่ง (ไม่ได้ break)
        else:
            # แจ้งข้อผิดพลาดว่าฉากแน่นเกินไป | input: - | output: ยกข้อยกเว้น RuntimeError
            raise RuntimeError("Scene is too crowded")
        # ถ้าให้แตะกันและมีเม็ดวางอยู่แล้ว
        if touching and masks:
            # เลื่อนเม็ดเข้าหากึ่งกลางจนชนเม็ดอื่น | input: occupied, solid, top, left | output: top, left ใหม่
            top, left = slide_to_centre(occupied, solid, top, left)
        # อัลฟา 0-1 ของเม็ด (คงมิติสุดท้ายไว้เพื่อคูณกับสี) | input: piece | output: alpha = float32 (h, w, 1)
        alpha = piece[..., 3:4].astype(np.float32) / 255
        # หน้าต่างบนฉากที่เม็ดจะวาง (เป็น view แก้ไขแล้วมีผลกับฉาก) | input: scene, top, left, solid | output: window
        window = scene[top:top + solid.shape[0], left:left + solid.shape[1]]
        # ผสมสีเม็ดกับพื้นหลังตามอัลฟา | input: piece, alpha, window | output: window ถูกเขียนทับ
        window[:] = piece[..., :3] * alpha + window * (1 - alpha)
        # สร้าง mask ขนาดเต็มฉาก | input: SCENE_SIZE | output: mask = bool (H, W) ค่า False
        mask = np.zeros(SCENE_SIZE, bool)
        # ตั้ง mask ที่ตำแหน่งเม็ดเป็นรูปทึบของเม็ด | input: solid, top, left | output: mask
        mask[top:top + solid.shape[0], left:left + solid.shape[1]] = solid
        # เพิ่มเม็ดนี้ในแผนที่พื้นที่ที่ถูกใช้ | input: mask | output: occupied อัปเดต
        occupied |= mask
        # เก็บ mask เม็ดนี้ | input: mask | output: masks ยาวขึ้น 1
        masks.append(mask)
    # คืนฉากเป็น uint8 และ list ของ mask | input: scene, masks | output: (ฉาก, masks)
    return scene.astype(np.uint8), masks


# ฟังก์ชันตรวจว่าวางรูปทึบที่ตำแหน่งนี้ได้หรือไม่ โดยเว้นระยะ gap | input: occupied = bool (H, W), solid = bool (h, w), top, left = ตำแหน่งมุมบนซ้าย, gap = ระยะเว้น (พิกเซล) | output: bool (True = วางได้)
def free(occupied, solid, top, left, gap):
    """Check that `solid` fits at this position with `gap` pixels of clearance."""
    # อ่านความสูงและความกว้างของรูปทึบ | input: solid | output: height, width
    height, width = solid.shape
    # ถ้าล้นขอบฉากด้านใดด้านหนึ่ง
    if top < 0 or left < 0 or top + height > occupied.shape[0] or left + width > occupied.shape[1]:
        # วางไม่ได้ | input: - | output: False
        return False
    # มุมบนซ้ายของหน้าต่างตรวจ (เผื่อ gap แต่ไม่ต่ำกว่า 0) | input: top, left, gap | output: y0, x0
    y0, x0 = max(top - gap, 0), max(left - gap, 0)
    # ตัดหน้าต่างของ occupied ที่รวมระยะเผื่อ | input: occupied, y0, x0 | output: window = bool
    window = occupied[y0:top + height + gap, x0:left + width + gap]
    # ถ้าต้องเว้นระยะ
    if gap:
        # ขยายพื้นที่ที่ถูกใช้ออก gap พิกเซลทุกทิศ | input: window, gap | output: window = bool ที่ขยายแล้ว
        window = cv2.dilate(window.astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0
        # ตัดเหลือเฉพาะส่วนที่ตรงกับรูปทึบ | input: window, top, y0, left, x0 | output: window ขนาดเท่า solid
        window = window[top - y0:top - y0 + height, left - x0:left - x0 + width]
    # วางได้ถ้าไม่มีพิกเซลใดของรูปทึบทับกับพื้นที่ที่ถูกใช้ | input: window, solid | output: bool
    return not (window & solid).any()


# ฟังก์ชันเลื่อนเม็ดยาเข้าหากึ่งกลางทีละน้อย จนจะชน | input: occupied, solid, top, left | output: tuple (top, left) ตำแหน่งใหม่เป็น int
def slide_to_centre(occupied, solid, top, left):
    """Move a pill toward the centre, one small step at a time, until it would collide."""
    # ตำแหน่งมุมบนซ้ายที่ทำให้เม็ดอยู่กึ่งกลางฉาก | input: occupied.shape, solid.shape | output: target = อาร์เรย์ [แถว, คอลัมน์]
    target = np.array(occupied.shape) / 2 - np.array(solid.shape) / 2
    # ตำแหน่งปัจจุบันเป็น float | input: top, left | output: position = อาร์เรย์
    position = np.array([top, left], float)
    # เวกเตอร์จากตำแหน่งปัจจุบันไปเป้าหมาย | input: target, position | output: step
    step = target - position
    # ระยะทางถึงเป้าหมาย | input: step | output: distance = float
    distance = np.linalg.norm(step)
    # ถ้าอยู่ใกล้เป้าหมายแล้ว (น้อยกว่า 1 พิกเซล)
    if distance < 1:
        # ไม่ต้องเลื่อน | input: - | output: (top, left) เดิม
        return top, left
    # ทำเวกเตอร์ให้ยาว 2 พิกเซลต่อก้าว | input: step, distance | output: step
    step = step / distance * 2
    # เดินสูงสุดเท่าจำนวนก้าวที่ถึงเป้าหมาย
    for _ in range(int(distance / 2)):
        # ตำแหน่งถัดไปหลังเดิน 1 ก้าว | input: position, step | output: following
        following = position + step
        # ถ้าตำแหน่งถัดไปชนเม็ดอื่นหรือล้นขอบ
        if not free(occupied, solid, int(following[0]), int(following[1]), 0):
            # หยุดเลื่อน | input: - | output: -
            break
        # ยอมรับตำแหน่งใหม่ | input: following | output: position
        position = following
    # คืนตำแหน่งสุดท้ายเป็นจำนวนเต็ม | input: position | output: (top, left)
    return int(position[0]), int(position[1])
