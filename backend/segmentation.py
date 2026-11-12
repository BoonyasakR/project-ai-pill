"""Find separate pills on a dark background and align each one like the training images."""
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

CANVAS = 800  # Training images are 800 x 800.
PILL_LENGTH = 693  # Median long-axis length of a training pill, in pixels.
BACKGROUND_BGR = (26, 15, 15)  # Median training background.
MIN_RELATIVE_AREA = 0.2  # Smaller blobs are not inspected as pills.
MIN_WARNED_AREA = 0.01  # Blobs below this share of the largest are specks. No warning needed.
MIN_RELIABLE_LENGTH = 180  # Pixels. Shorter normal pills raise false alarms once enlarged.
MASK_MARGIN = 0.02  # Keep this share of the pill length around the mask.
# Widest pose that still counts as a training image. Every MVTec pill fits:
# at most 1.3 degrees of tilt, 1.9% size change, and 6 pixels off centre.
MAX_TILT = 2  # Degrees.
MAX_SIZE_CHANGE = 0.025
MAX_SHIFT = 10  # Pixels on the training canvas.


def find_pills(image):
    """Return one boolean mask per pill in an RGB array, largest first."""
    return keep_pills(find_blobs(image))


def keep_pills(blobs):
    """Drop blobs far smaller than the largest one. `blobs` must be largest first."""
    return [blob for blob in blobs if blob.sum() >= MIN_RELATIVE_AREA * blobs[0].sum()]


def find_blobs(image):
    """Return one boolean mask per separate bright object, largest first.

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
    # Objects too thin to give a peak get no watershed marker, so add them back.
    unmarked, count = ndimage.label(foreground & (labels == 0))
    masks += [unmarked == index for index in range(1, count + 1)]
    return sorted(masks, key=np.sum, reverse=True)


def pill_pose(mask):
    """Return the centre, long-axis angle in degrees, and long-axis length of a mask."""
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    axis = vectors[:, 1]
    length = np.ptp((points - centre) @ axis)
    return centre, float(np.degrees(np.arctan2(axis[1], axis[0]))), float(length)


def in_training_pose(mask):
    """Return whether a one-pill image already sits like a training image.

    The long axis has no direction, so an upside-down pill also passes.
    """
    height, width = mask.shape
    if height != width:
        return False
    centre, angle, length = pill_pose(mask)
    scale = CANVAS / width
    tilt = abs((angle + 90) % 180 - 90)
    shift = np.hypot(*(centre * scale - CANVAS / 2))
    return tilt <= MAX_TILT and abs(length * scale / PILL_LENGTH - 1) <= MAX_SIZE_CHANGE and shift <= MAX_SHIFT


def align_pill(image, mask, others=None, flip=False, canvas=CANVAS, pill_length=PILL_LENGTH, rotate=True):
    """Rotate, scale, and centre one pill on a training-style canvas.

    `others` marks neighbouring pills, which are removed. The long axis has no
    direction, so `flip` gives the other 180-degree option. Round pills pass
    `rotate=False` and keep their angle. Returns the aligned image and a boolean
    mask of the pill on it.
    """
    centre, angle, length = pill_pose(mask)
    turn = (angle if rotate else 0) + (180 if flip else 0)
    matrix = cv2.getRotationMatrix2D(tuple(centre), turn, pill_length / length)
    matrix[:, 2] += canvas / 2 - centre
    size = (canvas, canvas)
    pill = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_CUBIC)
    aligned_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
    margin = int(pill_length * MASK_MARGIN) * 2 + 1
    keep = cv2.dilate(aligned_mask, np.ones((margin, margin), np.uint8))
    if others is not None:
        neighbours = cv2.warpAffine(others.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
        keep[cv2.dilate(neighbours, np.ones((5, 5), np.uint8)) > 0] = 0
    keep = cv2.GaussianBlur(keep.astype(np.float32), (9, 9), 0)[..., None]
    background = np.array(BACKGROUND_BGR[::-1], np.float32)
    aligned = (pill * keep + background * (1 - keep)).clip(0, 255).astype(np.uint8)
    return aligned, aligned_mask > 0


def pink_pill_mask(image):
    """The pink pill in a pillQC crop. Dirt on the pill stays inside; dirt on the floor does not."""
    red, green = image[..., 0].astype(np.int16), image[..., 1].astype(np.int16)
    pink = ((red > 150) & (red > green + 10)).astype(np.uint8)
    contours, _ = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(pink)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return mask > 0


ROUND_PILL_GROWTH = 1.05  # Keeps the rim, where chips are, but not floor dirt touching the pill.


def round_pill_region(mask):
    """A disk over a round pill: the largest circle inside the mask, grown by 5%.

    White dirt on the floor can touch the pill and join its mask. The inscribed
    circle ignores such bumps, so only the pill itself is scored.
    """
    mask = mask.astype(np.uint8)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    disk = np.zeros_like(mask)
    cv2.circle(disk, (int(x), int(y)), int(distance.max() * ROUND_PILL_GROWTH), 1, -1)
    return disk > 0
