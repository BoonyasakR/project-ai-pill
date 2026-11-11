"""Bootstrap confidence intervals for image-level anomaly detection metrics.

Images are resampled with replacement, normal and defective separately, so every
resample keeps both classes. The threshold is held fixed, so the intervals do not
include the uncertainty of the threshold itself.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import ad_metrics

METRICS = ("image_auroc", "precision", "recall", "f1")


def _resamples(labels, rounds, seed):
    normal, defect = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if normal.size == 0 or defect.size == 0:
        raise ValueError("bootstrap needs both normal and defective images")
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        yield np.concatenate([rng.choice(normal, normal.size), rng.choice(defect, defect.size)])


def _interval(value, draws, level):
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return {"value": float(value), "low": float(low), "high": float(high)}


def image_ci(labels, scores, threshold, rounds=2000, seed=42, level=0.95):
    """Percentile interval of each image metric. labels: 1 means defective."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    point = ad_metrics.image_metrics(labels, scores, threshold)
    draws = [
        ad_metrics.image_metrics(labels[index], scores[index], threshold)
        for index in _resamples(labels, rounds, seed)
    ]
    return {name: _interval(point[name], [draw[name] for draw in draws], level) for name in METRICS}


def paired_auroc_diff_ci(labels, scores_a, scores_b, rounds=2000, seed=42, level=0.95):
    """Interval of image AUROC(a) minus AUROC(b), both scored on the same resamples."""
    labels = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    point = roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b)
    draws = [
        roc_auc_score(labels[index], scores_a[index]) - roc_auc_score(labels[index], scores_b[index])
        for index in _resamples(labels, rounds, seed)
    ]
    return _interval(point, draws, level)
