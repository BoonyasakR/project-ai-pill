"""Evaluation metrics for anomaly detection. Model independent."""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def pick_threshold(validation_scores, quantile=0.99):
    """Image-score threshold taken from normal validation images only."""
    return float(np.quantile(np.asarray(validation_scores, dtype=np.float64), quantile))


def au_pro(masks, maps, fpr_limit=0.3, steps=200):
    """Area under the per-region-overlap curve up to fpr_limit, scaled to 0-1.

    masks: bool array (N, H, W). maps: float array (N, H, W).
    Each connected defect region counts equally, however large it is.
    """
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float32)
    if masks.shape != maps.shape:
        raise ValueError("masks and maps must have the same shape")
    if not masks.any():
        raise ValueError("au_pro needs at least one defect pixel")

    fprs = np.linspace(0.0, fpr_limit, steps + 1)
    thresholds = np.quantile(maps[~masks], 1.0 - fprs)

    overlaps = []
    for mask, score_map in zip(masks, maps):
        labels, count = ndimage.label(mask)
        for region in range(1, count + 1):
            scores = np.sort(score_map[labels == region])
            below = np.searchsorted(scores, thresholds, side="right")
            overlaps.append(1.0 - below / scores.size)
    pro = np.mean(overlaps, axis=0)

    area = np.sum((pro[1:] + pro[:-1]) / 2 * np.diff(fprs))
    return float(area / fpr_limit)


def evaluate(labels, scores, masks, maps, threshold):
    """Image and pixel metrics for one model. labels: 1 means defective."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores > threshold

    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    true_negative = int(np.sum(~predicted & (labels == 0)))

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    masks = np.asarray(masks, dtype=bool)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "pixel_auroc": float(roc_auc_score(masks.ravel(), np.asarray(maps).ravel())),
        "au_pro": au_pro(masks, maps),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def evaluate_by_defect(groups, scores, masks, maps, threshold):
    """Per-defect results. Each defect is compared against the good images."""
    groups = np.asarray(groups)
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps)
    good = groups == "good"

    rows = {}
    for defect in sorted(set(groups.tolist()) - {"good"}):
        selected = groups == defect
        both = selected | good
        rows[defect] = {
            "images": int(selected.sum()),
            "image_auroc": float(roc_auc_score(selected[both], scores[both])),
            "pixel_auroc": float(roc_auc_score(masks[both].ravel(), maps[both].ravel())),
            "au_pro": au_pro(masks[both], maps[both]),
            "recall": float(np.mean(scores[selected] > threshold)),
        }
    return rows
