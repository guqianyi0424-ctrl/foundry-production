"""Residue sampling helpers for imbalanced hotspot training."""

from __future__ import annotations

import random
from collections.abc import Sequence


def balanced_residue_indices(
    labels: Sequence[int],
    negative_ratio: int = 1,
    seed: int | None = None,
) -> list[int]:
    """Return valid residue indices with all positives and sampled negatives.

    Unknown residues must be labeled ``-1`` and are never returned. If a protein
    has only one observed class, return all observed residues so callers can
    decide whether to skip or keep that protein.
    """
    positive = [index for index, label in enumerate(labels) if int(label) == 1]
    negative = [index for index, label in enumerate(labels) if int(label) == 0]

    if not positive or not negative:
        return positive + negative

    n_negative = min(len(negative), len(positive) * max(int(negative_ratio), 1))
    rng = random.Random(seed)
    sampled_negative = rng.sample(negative, n_negative)
    return sorted(positive + sampled_negative)
