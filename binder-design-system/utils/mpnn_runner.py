"""
ProteinMPNN 调用模块
跨 conda 环境调用: binder(Python3.10) -> foundry(Python3.12) via conda run
"""
import os
import subprocess
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any


FOUNDRY_ENV = "foundry"


class MPNNRunner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "mpnn"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._foundry_available = None

    def _check_foundry(self) -> bool:
        if self._foundry_available is not None:
            return self._foundry_available
        try:
            result = subprocess.run(
                ["conda", "run", "-n", FOUNDRY_ENV, "--no-banner", "mpnn", "--help"],
                capture_output=True, text=True, timeout=30
            )
            self._foundry_available = result.returncode == 0
        except Exception:
            self._foundry_available = False
        return self._foundry_available

    def is_available(self) -> bool:
        return self._check_foundry()

    def run(
        self,
        backbone_pdb: str,
        num_sequences: int = 3,
        sampling_temp: float = 0.1,
        job_id: str = "default",
        top_k: int = 3
    ) -> Dict[str, Any]:
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            return self._mock_run(backbone_pdb, num_sequences, job_dir, top_k)

        cmd = [
            "conda", "run", "-n", FOUNDRY_ENV, "--no-banner",
            "mpnn",
            "--structure_path", backbone_pdb,
            "--out_directory", str(job_dir),
            "--model_type", "ligand_mpnn",
            "--batch_size", str(num_sequences),
            "--number_of_batches", "1",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr[-500:] if result.stderr else "Unknown error",
                    "sequences": []
                }

            sequences = self._parse_results(job_dir)
            sequences = self._rank_and_select_top_k(sequences, top_k)

            return {
                "success": True,
                "sequences": sequences,
                "num_sequences": len(sequences),
                "output_dir": str(job_dir)
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "MPNN运行超时(10min)", "sequences": []}
        except Exception as e:
            return {"success": False, "error": str(e), "sequences": []}

    def _parse_results(self, job_dir: Path) -> List[Dict]:
        sequences = []

        fasta_files = sorted(glob.glob(str(job_dir / "**/*.fa"), recursive=True))
        if not fasta_files:
            fasta_files = sorted(glob.glob(str(job_dir / "*.fa")))
        for fasta_path in fasta_files:
            with open(fasta_path, "r") as f:
                content = f.read()

            entries = content.split(">")
            for entry in entries:
                if not entry.strip():
                    continue
                lines = entry.strip().split("\n")
                header = lines[0]
                seq = "".join(lines[1:])

                score = 0.0
                if "score=" in header:
                    try:
                        score = float(header.split("score=")[1].split(",")[0].strip())
                    except (ValueError, IndexError):
                        pass

                sequences.append({
                    "header": header,
                    "sequence": seq,
                    "score": score,
                    "fasta_path": fasta_path
                })

        json_files = sorted(glob.glob(str(job_dir / "**/*.json"), recursive=True))
        if not json_files:
            json_files = sorted(glob.glob(str(job_dir / "*.json")))
        for json_path in json_files:
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "sequence" in data:
                    sequences.append({
                        "header": data.get("header", ""),
                        "sequence": data["sequence"],
                        "score": data.get("score", 0.0),
                        "json_path": json_path
                    })
            except Exception:
                pass

        return sequences

    def _rank_and_select_top_k(self, sequences: List[Dict], top_k: int) -> List[Dict]:
        sequences.sort(key=lambda x: x.get("score", 0))
        selected = sequences[:top_k]
        for i, s in enumerate(selected):
            s["rank"] = i + 1
        return selected

    def _mock_run(
        self,
        backbone_pdb: str,
        num_sequences: int,
        job_dir: Path,
        top_k: int = 3
    ) -> Dict[str, Any]:
        import random

        aa_list = "ACDEFGHIKLMNPQRSTVWY"

        try:
            with open(backbone_pdb, "r") as f:
                pdb_content = f.read()
            n_residues = pdb_content.count("ATOM")
        except Exception:
            n_residues = 60

        sequences = []
        for i in range(num_sequences):
            seq = "".join(random.choices(aa_list, k=n_residues))
            score = round(random.uniform(-30, -15), 2)

            sequences.append({
                "header": f"design_{i},score={score}",
                "sequence": seq,
                "score": score,
                "mock": True,
                "rank": i + 1
            })

            fa_path = job_dir / f"design_{i}.fa"
            with open(fa_path, "w") as f:
                f.write(f">design_{i},score={score}\n{seq}\n")

        sequences.sort(key=lambda x: x.get("score", 0))
        selected = sequences[:top_k]
        for i, s in enumerate(selected):
            s["rank"] = i + 1

        return {
            "success": True,
            "sequences": selected,
            "num_sequences": len(selected),
            "output_dir": str(job_dir),
            "mock": True
        }
