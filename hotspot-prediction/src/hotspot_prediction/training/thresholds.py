"""Threshold selection utilities."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def matthews_corrcoef(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / math.sqrt(denom)


def best_mcc_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Pick the threshold with the best validation MCC."""
    if thresholds is None:
        thresholds = (i / 100 for i in range(1, 100))

    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
        score = matthews_corrcoef(y_true, y_pred)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score
