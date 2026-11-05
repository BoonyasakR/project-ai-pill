"""Find separate pills on a dark background and align each one like the training images."""
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

CANVAS = 800  # Training images are 800 x 800.
PILL_LENGTH = 693  # Median long-axis length of a training pill, in pixels.
BACKGROUND_BGR = (26, 15, 15)  # Median training background.
MIN_RELATIVE_AREA = 0.2  # Smaller blobs are specks, not pills.
MASK_MARGIN = 0.02  # Keep this share of the pill length around the mask.


def find_pills(image):
    """Return one boolean mask per pill in an RGB array, largest first.

    ponytail: Otsu needs bright pills on a dark background. Use a trained
    instance segmenter if pills ever overlap or sit on a light surface.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    foreground = ndimage.binary_fill_holes(binary > 0)
    if not foreground.any():
        return []

    distance = ndimage.distance_transform_edt(foreground)
    # A pill's distance ridge is one peak. Touching pills give one peak each.
    smooth = cv2.GaussianBlur(distance.astype(np.float32), (0, 0), 3)
    blobs, _ = ndimage.label(foreground)
    widest = distance.max()
    peaks = peak_local_max(
        smooth, min_distance=max(int(widest * 0.8), 5), threshold_abs=widest * 0.4, labels=blobs
    )
    markers = np.zeros(foreground.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-smooth, markers, mask=foreground)

    masks = [labels == index for index in range(1, labels.max() + 1)]
    masks = sorted(masks, key=np.sum, reverse=True)
    return [mask for mask in masks if mask.sum() >= MIN_RELATIVE_AREA * masks[0].sum()]


def pill_pose(mask):
    """Return the centre, long-axis angle in degrees, and long-axis length of a mask."""
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    axis = vectors[:, 1]
    length = np.ptp((points - centre) @ axis)
    return centre, float(np.degrees(np.arctan2(axis[1], axis[0]))), float(length)


def align_pill(image, mask, others=None, flip=False):
    """Rotate, scale, and centre one pill on a training-style canvas.

    `others` marks neighbouring pills, which are removed. The long axis has no
    direction, so `flip` gives the other 180-degree option. Returns the aligned
    image and a boolean mask of the pill on it.
    """
    centre, angle, length = pill_pose(mask)
    matrix = cv2.getRotationMatrix2D(tuple(centre), angle + (180 if flip else 0), PILL_LENGTH / length)
    matrix[:, 2] += CANVAS / 2 - centre
    size = (CANVAS, CANVAS)
    pill = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_CUBIC)
    aligned_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
    margin = int(PILL_LENGTH * MASK_MARGIN) * 2 + 1
    keep = cv2.dilate(aligned_mask, np.ones((margin, margin), np.uint8))
    if others is not None:
        neighbours = cv2.warpAffine(others.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
        keep[cv2.dilate(neighbours, np.ones((5, 5), np.uint8)) > 0] = 0
    keep = cv2.GaussianBlur(keep.astype(np.float32), (9, 9), 0)[..., None]
    background = np.array(BACKGROUND_BGR[::-1], np.float32)
    aligned = (pill * keep + background * (1 - keep)).clip(0, 255).astype(np.uint8)
    return aligned, aligned_mask > 0
