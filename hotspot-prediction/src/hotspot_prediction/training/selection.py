"""Validation metric based checkpoint selection helpers."""

from __future__ import annotations

from typing import Any


def should_save_checkpoint(metrics: dict[str, Any], best_score: float, metric: str = "pr_auc") -> bool:
    """Return true when the selected validation metric improves."""
    value = metrics.get(metric)
    if value is None:
        return False
    if best_score is None:
        return True
    return float(value) > float(best_score)
