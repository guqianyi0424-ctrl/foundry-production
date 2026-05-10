"""Small I/O helpers for normalized hotspot data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile


def read_excel_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read Excel rows through pandas, falling back to simple xlsx XML parsing."""
    try:
        import pandas as pd
    except ImportError:
        return _read_xlsx_rows_stdlib(path)

    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df.to_dict(orient="records")


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter.upper()) - 64
    return index - 1


def _read_xlsx_rows_stdlib(path: str | Path) -> list[dict[str, Any]]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        first_sheet = workbook.find(".//a:sheet", ns)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        sheet_path = "xl/" + relmap[rel_id].lstrip("/")
        sheet = ET.fromstring(archive.read(sheet_path))

    table: list[list[Any]] = []
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        values: list[Any] = []
        for cell in row.findall("a:c", ns):
            idx = _column_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            value_node = cell.find("a:v", ns)
            value: Any = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value != "":
                value = shared_strings[int(value)]
            values[idx] = value
        table.append(values)

    if not table:
        return []
    headers = [str(value).strip() for value in table[0]]
    rows: list[dict[str, Any]] = []
    for raw_row in table[1:]:
        row = {}
        for idx, header in enumerate(headers):
            if header:
                row[header] = raw_row[idx] if idx < len(raw_row) else ""
        rows.append(row)
    return rows


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
