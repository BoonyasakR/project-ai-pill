"""Flag pill colours that never appear on normal training pills.

Normal pills are white with dull red specks. Colour defects are stains in new colours (bright red,
pink, yellow, blue), often small or on the rim, where PatchCore at 256 pixels misses them.
"""
import cv2
import numpy as np
from scipy import ndimage

BIN = 4  # Lab units per histogram bin, on OpenCV's 8-bit Lab scale.
BINS = 256 // BIN
TOLERANCE = 1  # Neighbouring bins that also count as seen.
MIN_AREA = 20  # Pixels on the 800 x 800 canvas. Smaller blobs are noise, whatever validation says.


def colour_bins(image):
    """Quantised Lab colour of each pixel in an RGB array, lightly blurred against noise."""
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_RGB2LAB)
    return lab.astype(np.int64) // BIN


def build_seen(pills):
    """Return a boolean table of colours seen on normal pills. `pills` yields (RGB array, mask)."""
    counts = np.zeros((BINS,) * 3, np.int64)
    for image, mask in pills:
        bins = colour_bins(image)[mask]
        np.add.at(counts, (bins[:, 0], bins[:, 1], bins[:, 2]), 1)
    return ndimage.maximum_filter(counts > 0, size=2 * TOLERANCE + 1)


def novel_colour(seen, image, mask):
    """Return the area of the largest unseen-colour blob inside the pill, and all unseen pixels."""
    bins = colour_bins(image)
    novel = ~seen[bins[..., 0], bins[..., 1], bins[..., 2]] & mask
    labels, count = ndimage.label(novel)
    return (int(np.bincount(labels.ravel())[1:].max()) if count else 0), novel
