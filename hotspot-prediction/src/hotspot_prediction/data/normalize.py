"""Normalize hotspot prediction source datasets.

The training source is data1, where labels are derived from mutation ddG:
hotspot if ddG > 2 kcal/mol. The independent test source is data2, where
hotspot lists are provided in UniProt numbering and mapped to PDB-relative
residue numbering through the structure range in the spreadsheet.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable


DDG_HOTSPOT_THRESHOLD = 2.0


@dataclass(frozen=True)
class ParsedPDB:
    pdb_id: str
    chain_id: str
    uniprot_start: int | None = None
    uniprot_end: int | None = None


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    normalized = {key.strip(): value for key, value in row.items()}
    for name in names:
        if name in normalized:
            return normalized[name]
    return ""


def label_from_ddg(ddg: Any, threshold: float = DDG_HOTSPOT_THRESHOLD) -> int:
    """Return 1 when ddG is strictly above the hotspot threshold."""
    parsed = parse_float(ddg)
    if parsed is None:
        return 0
    return int(parsed > threshold)


def parse_float(value: Any) -> float | None:
    text = _clean_cell(value)
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_hotspot_list(value: Any) -> list[int]:
    """Parse comma-separated residue positions, ignoring invalid tokens."""
    text = _clean_cell(value)
    if text == "":
        return []

    positions: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            positions.append(int(float(token)))
        except ValueError:
            continue
    return positions


def parse_pdb_chain_range(value: Any) -> ParsedPDB:
    """Parse strings such as ``5f18-A(374-620)`` or ``1c2b-A``."""
    text = _clean_cell(value)
    match = re.match(
        r"^\s*(?P<pdb>[A-Za-z0-9]{4})(?:[-_](?P<chain>[A-Za-z0-9]))?"
        r"(?:\((?P<start>\d+)\s*-\s*(?P<end>\d+)\))?",
        text,
    )
    if not match:
        return ParsedPDB(pdb_id=text[:4].lower(), chain_id="A")

    start = match.group("start")
    end = match.group("end")
    return ParsedPDB(
        pdb_id=match.group("pdb").lower(),
        chain_id=(match.group("chain") or "A").upper(),
        uniprot_start=int(start) if start else None,
        uniprot_end=int(end) if end else None,
    )


def uniprot_to_pdb_positions(positions: Iterable[int], uniprot_start: int | None) -> list[int]:
    """Convert UniProt positions to one-based PDB-relative residue positions."""
    if uniprot_start is None:
        return []
    converted: list[int] = []
    for pos in positions:
        pdb_pos = int(pos) - int(uniprot_start) + 1
        if pdb_pos > 0:
            converted.append(pdb_pos)
    return converted


def parse_mutation(value: Any) -> tuple[str, str]:
    text = _clean_cell(value)
    if "->" not in text:
        return "", ""
    wild_type, mutant = text.split("->", 1)
    return wild_type.strip(), mutant.strip()


def normalize_data1_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize data1 mutation rows into residue-level mutation records."""
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        pdb = parse_pdb_chain_range(_first_present(row, ["PDB ID", "pdb_id"]))
        pdb_positions = parse_hotspot_list(
            _first_present(row, ["PDB # of PPI-hot spots", "PDB # of PPI-hot spots "])
        )
        uniprot_positions = parse_hotspot_list(
            _first_present(row, ["Uniprot # of PPI-hot spots", "Uniprot # of PPI-hot spots "])
        )
        pdb_residue = pdb_positions[0] if pdb_positions else None
        uniprot_residue = uniprot_positions[0] if uniprot_positions else None
        wild_type, mutant = parse_mutation(_first_present(row, ["wild->mutant"]))
        ddg = parse_float(_first_present(row, ["Binding free energy change (kcal/mol)"]))
        if ddg is None or pdb_residue is None:
            continue

        normalized_rows.append(
            {
                "uniprot_id": _clean_cell(_first_present(row, ["Uniprot code", "uniprot_id"])),
                "partner_id": _clean_cell(
                    _first_present(row, ["Uniprot code of the binding partner", "partner_id"])
                ),
                "pdb_id": pdb.pdb_id,
                "chain_id": pdb.chain_id,
                "pdb_residue": pdb_residue,
                "uniprot_residue": uniprot_residue,
                "wild_type": wild_type,
                "mutant": mutant,
                "ddg": ddg,
                "label": label_from_ddg(ddg),
                "source": "data1",
            }
        )
    return normalized_rows


def normalize_data2_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize data2 rows into independent-test PDB-chain samples."""
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        pdb = parse_pdb_chain_range(
            _first_present(row, ["PDB ID of free protein A structure(length in Uniprot)", "pdb_id"])
        )
        hotspots_uniprot = parse_hotspot_list(
            _first_present(row, ["PPI-hot spots (Uniprot numbering)", "hotspot_list"])
        )
        hotspots_pdb = uniprot_to_pdb_positions(hotspots_uniprot, pdb.uniprot_start)

        normalized_rows.append(
            {
                "uniprot_id": _clean_cell(_first_present(row, ["Uniprot code of protein A", "uniprot_id"])),
                "partner_id": _clean_cell(
                    _first_present(
                        row,
                        ["Uniprot code of the binding partner, protein B", "partner_id"],
                    )
                ),
                "pdb_id": pdb.pdb_id,
                "chain_id": pdb.chain_id,
                "uniprot_start": pdb.uniprot_start,
                "uniprot_end": pdb.uniprot_end,
                "hotspots_uniprot": hotspots_uniprot,
                "hotspots_pdb": hotspots_pdb,
                "source": "data2",
            }
        )
    return normalized_rows


def build_data1_train_samples(mutation_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate data1 mutation rows into legacy PDB-chain sample rows."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in mutation_rows:
        pdb_id = _clean_cell(row.get("pdb_id")).lower()
        chain_id = (_clean_cell(row.get("chain_id")) or "A").upper()
        key = (pdb_id, chain_id)
        if key not in grouped:
            grouped[key] = {
                "uniprot_id": _clean_cell(row.get("uniprot_id")),
                "partner_id": _clean_cell(row.get("partner_id")),
                "pdb_id": f"{pdb_id}-{chain_id}",
                "hotspots": set(),
                "source": "data1",
            }

        if int(row.get("label", 0)) == 1 and row.get("pdb_residue") is not None:
            grouped[key]["hotspots"].add(int(row["pdb_residue"]))

    samples: list[dict[str, Any]] = []
    for value in grouped.values():
        hotspots = sorted(value.pop("hotspots"))
        samples.append(
            {
                **value,
                "hotspot_list": ",".join(str(pos) for pos in hotspots),
                "n_hotspots": len(hotspots),
            }
        )
    return samples


def build_data2_test_samples(test_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export normalized data2 rows into legacy PDB-chain sample rows."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in test_rows:
        pdb_id = _clean_cell(row.get("pdb_id")).lower()
        chain_id = (_clean_cell(row.get("chain_id")) or "A").upper()
        key = (pdb_id, chain_id)
        if key not in grouped:
            grouped[key] = {
                "uniprot_id": _clean_cell(row.get("uniprot_id")),
                "partner_id": _clean_cell(row.get("partner_id")),
                "pdb_id": f"{pdb_id}-{chain_id}",
                "hotspots": set(),
                "source": "data2",
            }
        for pos in row.get("hotspots_pdb", []):
            grouped[key]["hotspots"].add(int(pos))

    samples: list[dict[str, Any]] = []
    for value in grouped.values():
        hotspots = sorted(value.pop("hotspots"))
        samples.append(
            {
                **value,
                "hotspot_list": ",".join(str(pos) for pos in hotspots),
                "n_hotspots": len(hotspots),
            }
        )
    return samples
