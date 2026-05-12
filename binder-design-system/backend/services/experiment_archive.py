import csv
import json
import re
from pathlib import Path
from typing import Any


class ExperimentArchiveService:
    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)

    def write_archive(
        self,
        experiment: dict[str, Any],
        results: dict[str, Any],
        designs: list[dict[str, Any]],
    ) -> dict[str, str]:
        experiment_id = str(experiment["id"])
        archive_dir = self.output_root / "experiments" / experiment_id
        structures_dir = archive_dir / "structures"
        raw_results_dir = archive_dir / "raw_results"
        archive_dir.mkdir(parents=True, exist_ok=True)
        structures_dir.mkdir(parents=True, exist_ok=True)
        raw_results_dir.mkdir(parents=True, exist_ok=True)

        design_rows = []
        candidate_payload = []
        for index, design in enumerate(designs, start=1):
            design_name = design.get("design_name") or design.get("name") or f"candidate_{index:03d}"
            candidate_id = design.get("id") or design_name
            pdb_file = ""
            pdb_content = design.get("pdb_content") or ""
            if pdb_content:
                pdb_filename = f"{self._safe_filename(design_name)}.pdb"
                pdb_path = structures_dir / pdb_filename
                pdb_path.write_text(pdb_content)
                pdb_file = f"structures/{pdb_filename}"

            row = {
                "experiment_id": experiment_id,
                "candidate_id": str(candidate_id),
                "design_name": str(design_name),
                "sequence": design.get("sequence") or "",
                "plddt": self._stringify_metric(design.get("plddt")),
                "rmsd": self._stringify_metric(design.get("rmsd")),
                "ranking_score": self._stringify_metric(design.get("ranking_score")),
                "passed_validation": "true" if design.get("passed_validation") else "false",
                "pdb_file": pdb_file,
                "created_at": experiment.get("created_at") or "",
            }
            design_rows.append(row)
            candidate_payload.append({**row, "pdb_content": pdb_content})

        self._write_manifest(archive_dir, experiment, results)
        self._write_candidates_csv(archive_dir / "candidates.csv", design_rows)
        (archive_dir / "candidates.json").write_text(json.dumps(candidate_payload, indent=2))
        self._write_fasta(archive_dir / "sequences.fasta", design_rows)
        self._write_raw_results(raw_results_dir, results)

        return {
            "archive_dir": str(archive_dir),
            "manifest": str(archive_dir / "manifest.json"),
            "candidates_csv": str(archive_dir / "candidates.csv"),
            "candidates_json": str(archive_dir / "candidates.json"),
            "sequences_fasta": str(archive_dir / "sequences.fasta"),
        }

    def _write_manifest(
        self,
        archive_dir: Path,
        experiment: dict[str, Any],
        results: dict[str, Any],
    ) -> None:
        manifest = {
            "experiment": experiment,
            "results_summary": {
                key: self._summarize_result(value) for key, value in results.items()
            },
            "artifact_files": {
                "candidates_csv": "candidates.csv",
                "candidates_json": "candidates.json",
                "sequences_fasta": "sequences.fasta",
                "structures_dir": "structures/",
                "raw_results_dir": "raw_results/",
            },
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _write_candidates_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        fieldnames = [
            "experiment_id",
            "candidate_id",
            "design_name",
            "sequence",
            "plddt",
            "rmsd",
            "ranking_score",
            "passed_validation",
            "pdb_file",
            "created_at",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_fasta(self, path: Path, rows: list[dict[str, str]]) -> None:
        lines = []
        for row in rows:
            if not row["sequence"]:
                continue
            lines.append(f">{row['design_name']}")
            lines.append(row["sequence"])
        content = "\n".join(lines)
        path.write_text(f"{content}\n" if content else "")

    def _write_raw_results(self, raw_results_dir: Path, results: dict[str, Any]) -> None:
        for step in ("rfd3", "mpnn", "rf3"):
            payload = results.get(step)
            if payload is not None:
                (raw_results_dir / f"{step}_results.json").write_text(
                    json.dumps(payload, indent=2)
                )

    def _summarize_result(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        summary = {"success": result.get("success")}
        for key in ("num_designs", "num_sequences", "avg_plddt", "rmsd", "passed"):
            if key in result:
                summary[key] = result[key]
        return summary

    def _safe_filename(self, value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        return safe or "candidate"

    def _stringify_metric(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)
