"""Dependency-light graph sample data structures and collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class GraphSample:
    pdb_id: str
    sequence: str
    node_features: list[list[float]]
    labels: list[int]
    edge_index: tuple[list[int], list[int]]
    residue_numbers: list[int]
    chain_id: str = ""
    coords: list[list[float]] | None = None


def _as_rows(values: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [list(row) for row in values]


def collate_samples(samples: Sequence[GraphSample]) -> dict[str, Any]:
    """Collate samples without padding labels, so nodes and labels stay aligned."""
    pdb_ids: list[str] = []
    sequences: list[str] = []
    node_features: list[list[float]] = []
    labels: list[int] = []
    residue_numbers: list[int] = []
    node_slices: list[tuple[int, int]] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []

    offset = 0
    for sample in samples:
        n_nodes = len(sample.node_features)
        start = offset
        end = start + n_nodes

        pdb_ids.append(sample.pdb_id)
        sequences.append(sample.sequence)
        node_features.extend(_as_rows(sample.node_features))
        labels.extend(int(label) for label in sample.labels)
        residue_numbers.extend(int(number) for number in sample.residue_numbers)
        node_slices.append((start, end))

        src, dst = sample.edge_index
        edge_src.extend(int(node) + offset for node in src)
        edge_dst.extend(int(node) + offset for node in dst)

        offset = end

    return {
        "pdb_ids": pdb_ids,
        "sequences": sequences,
        "node_features": node_features,
        "labels": labels,
        "residue_numbers": residue_numbers,
        "node_slices": node_slices,
        "edge_index": (edge_src, edge_dst),
    }
