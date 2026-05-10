"""Residue graph construction utilities."""

from __future__ import annotations

import math
from typing import Sequence


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def build_distance_edges(coords: Sequence[Sequence[float]], cutoff: float) -> tuple[list[int], list[int]]:
    """Build bidirectional residue edges for C-alpha pairs within cutoff."""
    src: list[int] = []
    dst: list[int] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _distance(coords[i], coords[j]) <= cutoff:
                src.extend([i, j])
                dst.extend([j, i])
    return src, dst
