"""Simple rule-based hotspot baseline.

The baseline is intentionally lightweight. It provides a reproducible lower
bound for thesis tables when full PPI-hotspotID feature extraction is not
available locally.
"""

from __future__ import annotations

from typing import Any, Iterable


AROMATIC = {"F", "W", "Y"}
CHARGED = {"D", "E", "K", "R", "H"}
POLAR = {"N", "Q", "S", "T", "C"}


def score_residue(residue: str, residue_index: int, sequence_length: int) -> float:
    """Return a deterministic hotspot-likeness score in the range [0, 1]."""
    aa = residue.upper()
    score = 0.15

    if aa in AROMATIC:
        score += 0.35
    if aa in CHARGED:
        score += 0.25
    if aa in POLAR:
        score += 0.10

    if sequence_length > 1:
        rel_pos = residue_index / (sequence_length - 1)
        terminal_distance = min(rel_pos, 1.0 - rel_pos)
        score += 0.15 * terminal_distance

    return max(0.0, min(1.0, score))


def _labels_for_sample(sample: dict[str, Any], length: int) -> list[int]:
    labels = sample.get("labels", [])
    return [int(value) for value in labels[:length]]


def _residue_indices_for_sample(sample: dict[str, Any], length: int) -> list[int]:
    residue_indices = sample.get("residue_indices")
    if residue_indices is None:
        residue_indices = list(range(1, length + 1))
    return [int(value) for value in residue_indices[:length]]


def predict_rule_based_records(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return residue-level prediction records with ``pdb_id,label,prob`` fields."""
    records: list[dict[str, Any]] = []

    for sample in samples:
        sequence = str(sample.get("sequence", ""))
        labels = _labels_for_sample(sample, len(sequence))
        residue_indices = _residue_indices_for_sample(sample, len(sequence))
        pdb_id = str(sample.get("pdb_id", "unknown"))

        for idx, residue in enumerate(sequence):
            label = labels[idx] if idx < len(labels) else 0
            residue_id = residue_indices[idx] if idx < len(residue_indices) else idx + 1
            records.append(
                {
                    "pdb_id": pdb_id,
                    "residue_id": residue_id,
                    "aa": residue,
                    "label": int(label),
                    "prob": score_residue(residue, idx, len(sequence)),
                }
            )

    return records
