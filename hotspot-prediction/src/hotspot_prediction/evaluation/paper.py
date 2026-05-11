"""Paper-safe evaluation utilities."""

from __future__ import annotations

from typing import Any, Iterable


def resolve_test_threshold(
    source: str,
    fixed_threshold: float,
    cv_rows: Iterable[dict[str, Any]] | None = None,
) -> float:
    """Resolve an independent-test threshold without inspecting test labels."""
    if source == "fixed":
        return float(fixed_threshold)

    if source != "validation-mcc":
        raise ValueError("threshold source must be 'fixed' or 'validation-mcc'")

    values: list[float] = []
    for row in cv_rows or []:
        value = row.get("optimal_threshold", "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not values:
        return float(fixed_threshold)
    return sum(values) / len(values)
