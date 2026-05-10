"""CSV exports for independent-test evaluation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def export_prediction_table(
    path: str | Path,
    pdb_ids: Iterable[Any],
    labels: Iterable[Any],
    probs: Iterable[Any],
) -> None:
    """Write residue-level prediction records for Top-K analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "label", "prob"])
        writer.writeheader()
        for pdb_id, label, prob in zip(pdb_ids, labels, probs):
            writer.writerow(
                {
                    "pdb_id": str(pdb_id),
                    "label": int(label),
                    "prob": float(prob),
                }
            )


def export_metrics_table(
    path: str | Path,
    metrics: dict[str, Any],
    balanced_metrics: dict[str, Any] | None,
    threshold: float,
) -> None:
    """Write a one-row independent-test metrics table for report aggregation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {"threshold": float(threshold)}
    for key in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        if key in metrics:
            row[key] = metrics[key]

    if balanced_metrics:
        for key in ["roc_auc", "pr_auc", "precision", "recall", "specificity", "f1", "mcc"]:
            if key in balanced_metrics:
                row[f"balanced_{key}"] = balanced_metrics[key]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
