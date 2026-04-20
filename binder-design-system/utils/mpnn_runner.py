"""
ProteinMPNN / LigandMPNN 调用模块
优先使用 foundry Python API (MPNNInferenceEngine)
回退到 conda_bridge CLI 调用
最终回退到 mock 模式
"""
import os
import subprocess
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.conda_bridge import is_foundry_available, run_foundry_cli


class MPNNRunner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "mpnn"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        try:
            from mpnn.inference_engines.mpnn import MPNNInferenceEngine
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._api_available or is_foundry_available()

    def run(
        self,
        backbone_pdb: str,
        num_sequences: int = 3,
        sampling_temp: float = 0.1,
        job_id: str = "default",
        top_k: int = 3
    ) -> Dict[str, Any]:
        if self._api_available:
            try:
                return self._run_api(backbone_pdb, num_sequences, sampling_temp, job_id, top_k)
            except Exception as e:
                print(f"[MPNN] API调用失败，回退CLI: {e}")

        if is_foundry_available():
            try:
                return self._run_cli(backbone_pdb, num_sequences, sampling_temp, job_id, top_k)
            except Exception as e:
                print(f"[MPNN] CLI调用失败，回退mock: {e}")

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return self._mock_run(backbone_pdb, num_sequences, job_dir, top_k)

    def _run_api(
        self,
        backbone_pdb: str,
        num_sequences: int,
        sampling_temp: float,
        job_id: str,
        top_k: int
    ) -> Dict[str, Any]:
        from mpnn.inference_engines.mpnn import MPNNInferenceEngine
        from biotite.structure import from_pdb

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        atom_array = from_pdb(backbone_pdb)

        engine_config = {
            "model_type": "ligand_mpnn",
            "is_legacy_weights": True,
            "out_directory": str(job_dir),
            "write_structures": True,
            "write_fasta": True,
        }

        input_configs = [
            {
                "batch_size": num_sequences,
                "remove_waters": True,
            }
        ]

        model = MPNNInferenceEngine(**engine_config)
        mpnn_outputs = model.run(input_dicts=input_configs, atom_arrays=[atom_array])

        sequences = []
        for i, item in enumerate(mpnn_outputs):
            seq_1letter = self._extract_sequence_from_atom_array(item.atom_array)
            score = 0.0
            sequences.append({
                "header": f"design_{i},score={score}",
                "sequence": seq_1letter,
                "score": score,
                "rank": i + 1
            })

        sequences = self._rank_and_select_top_k(sequences, top_k)

        return {
            "success": True,
            "sequences": sequences,
            "num_sequences": len(sequences),
            "output_dir": str(job_dir)
        }

    def _extract_sequence_from_atom_array(self, atom_array) -> str:
        try:
            from biotite.structure import get_residue_starts
            from biotite.sequence import ProteinSequence

            res_starts = get_residue_starts(atom_array)
            seq_1letter = ''.join(
                ProteinSequence.convert_letter_3to1(res_name)
                for res_name in atom_array.res_name[res_starts]
            )
            return seq_1letter
        except Exception:
            return ""

    def _run_cli(
        self,
        backbone_pdb: str,
        num_sequences: int,
        sampling_temp: float,
        job_id: str,
        top_k: int
    ) -> Dict[str, Any]:
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        cmd_args = [
            "mpnn",
            "--structure_path", backbone_pdb,
            "--out_directory", str(job_dir),
            "--model_type", "ligand_mpnn",
            "--batch_size", str(num_sequences),
            "--number_of_batches", "1",
        ]

        result = run_foundry_cli(cmd_args, timeout=600)

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
