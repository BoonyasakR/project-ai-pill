"""Paste real MVTec pills into simulated multi-pill scenes with known labels."""
import cv2
import numpy as np

SCENE_SIZE = (1000, 1400)  # Height, width.
PILLS_PER_SCENE = 8
PILL_LENGTH_RANGE = (240, 330)
BACKGROUND_RGB = (15, 15, 26)
BACKGROUND_NOISE = 3.0


def cut_pill(image, mask, rng):
    """Return a randomly rotated and scaled RGBA cut-out of one pill."""
    ys, xs = np.nonzero(mask)
    length = np.ptp(xs)
    scale = rng.uniform(*PILL_LENGTH_RANGE) / length
    alpha = cv2.GaussianBlur(cv2.dilate(mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)), (7, 7), 0)
    rgba = np.dstack([image, alpha])
    side = int(length * scale * 1.2)
    matrix = cv2.getRotationMatrix2D((float(xs.mean()), float(ys.mean())), rng.uniform(0, 360), scale)
    matrix[:, 2] += side / 2 - np.array([xs.mean(), ys.mean()])
    return cv2.warpAffine(rgba, matrix, (side, side), flags=cv2.INTER_AREA)


def compose_scene(cutouts, rng, touching=False):
    """Place cut-outs without overlap. Return the scene and one mask per pill.

    With `touching`, each pill slides toward the centre until it hits another pill.
    """
    height, width = SCENE_SIZE
    noise = rng.normal(0, BACKGROUND_NOISE, (height, width, 3))
    scene = (np.array(BACKGROUND_RGB, np.float32) + noise).clip(0, 255)
    occupied = np.zeros(SCENE_SIZE, bool)
    masks = []
    for cutout in cutouts:
        solid = cutout[..., 3] > 127
        rows, cols = np.nonzero(solid)
        solid = solid[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        piece = cutout[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        for _ in range(2000):
            top = int(rng.integers(0, height - solid.shape[0]))
            left = int(rng.integers(0, width - solid.shape[1]))
            if free(occupied, solid, top, left, 0 if touching else 14):
                break
        else:
            raise RuntimeError("Scene is too crowded")
        if touching and masks:
            top, left = slide_to_centre(occupied, solid, top, left)
        alpha = piece[..., 3:4].astype(np.float32) / 255
        window = scene[top:top + solid.shape[0], left:left + solid.shape[1]]
        window[:] = piece[..., :3] * alpha + window * (1 - alpha)
        mask = np.zeros(SCENE_SIZE, bool)
        mask[top:top + solid.shape[0], left:left + solid.shape[1]] = solid
        occupied |= mask
        masks.append(mask)
    return scene.astype(np.uint8), masks


def free(occupied, solid, top, left, gap):
    """Check that `solid` fits at this position with `gap` pixels of clearance."""
    height, width = solid.shape
    if top < 0 or left < 0 or top + height > occupied.shape[0] or left + width > occupied.shape[1]:
        return False
    y0, x0 = max(top - gap, 0), max(left - gap, 0)
    window = occupied[y0:top + height + gap, x0:left + width + gap]
    if gap:
        window = cv2.dilate(window.astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0
        window = window[top - y0:top - y0 + height, left - x0:left - x0 + width]
    return not (window & solid).any()


def slide_to_centre(occupied, solid, top, left):
    """Move a pill toward the centre, one small step at a time, until it would collide."""
    target = np.array(occupied.shape) / 2 - np.array(solid.shape) / 2
    position = np.array([top, left], float)
    step = target - position
    distance = np.linalg.norm(step)
    if distance < 1:
        return top, left
    step = step / distance * 2
    for _ in range(int(distance / 2)):
        following = position + step
        if not free(occupied, solid, int(following[0]), int(following[1]), 0):
            break
        position = following
    return int(position[0]), int(position[1])
