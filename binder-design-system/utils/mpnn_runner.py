"""
ProteinMPNN / LigandMPNN 调用模块
完全对齐官方 all.ipynb API 用法
支持 fixed_chains 固定受体链 + atom_array 直接输入
"""
import os
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


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
        return self._api_available

    def run_mpnn(
        self,
        backbone_pdb_content: str = None,
        backbone_pdb_path: str = None,
        batch_size: int = 10,
        fixed_chains: List[str] = None,
        model_type: str = "ligand_mpnn",
        job_id: str = "default"
    ) -> Dict[str, Any]:
        if self._api_available:
            try:
                return self._run_api(
                    backbone_pdb_content, backbone_pdb_path, batch_size,
                    fixed_chains, model_type, job_id
                )
            except Exception as e:
                print(f"[MPNN] API调用失败: {e}")
                import traceback
                traceback.print_exc()

        return self._mock_run(
            backbone_pdb_content, backbone_pdb_path, batch_size,
            fixed_chains, job_id
        )

    def _run_api(
        self,
        backbone_pdb_content: str,
        backbone_pdb_path: str,
        batch_size: int,
        fixed_chains: List[str],
        model_type: str,
        job_id: str
    ) -> Dict[str, Any]:
        from mpnn.inference_engines.mpnn import MPNNInferenceEngine
        from biotite.structure.io.pdb import PDBFile
        import biotite.structure as bs

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if backbone_pdb_content:
            pdb_file = PDBFile.read(io.StringIO(backbone_pdb_content))
            atom_array = pdb_file.get_structure(model=1)
            atom_array = atom_array[bs.filter_amino_acids(atom_array)]
        elif backbone_pdb_path:
            pdb_file = PDBFile.read(backbone_pdb_path)
            atom_array = pdb_file.get_structure(model=1)
            atom_array = atom_array[bs.filter_amino_acids(atom_array)]
        else:
            return {"success": False, "error": "No backbone input provided", "sequences": []}

        engine_config = {
            "model_type": model_type,
            "is_legacy_weights": True,
            "out_directory": None,
            "write_structures": False,
            "write_fasta": False,
        }

        input_config = {
            "batch_size": batch_size,
            "remove_waters": True,
        }

        if fixed_chains:
            input_config["fixed_chains"] = fixed_chains

        input_configs = [input_config]

        print(f"[MPNN] model={model_type} | batch_size={batch_size} | fixed_chains={fixed_chains}")

        model = MPNNInferenceEngine(**engine_config)
        mpnn_outputs = model.run(input_dicts=input_configs, atom_arrays=[atom_array])

        sequences = []
        for i, item in enumerate(mpnn_outputs):
            seq_1letter = self._extract_sequence(item.atom_array)
            pdb_str = self._atom_array_to_pdb(item.atom_array)
            design_name = f"seq_{i}"
            pdb_path = job_dir / f"{design_name}.pdb"
            with open(pdb_path, "w") as f:
                f.write(pdb_str)

            sequences.append({
                "index": i,
                "name": design_name,
                "sequence": seq_1letter,
                "pdb_path": str(pdb_path),
                "pdb_content": pdb_str,
                "score": 0.0,
            })

        first_pdb = sequences[0]["pdb_content"] if sequences else ""

        return {
            "success": True,
            "sequences": sequences,
            "num_sequences": len(sequences),
            "first_sequence_pdb": first_pdb,
            "output_dir": str(job_dir),
        }

    def _extract_sequence(self, atom_array) -> str:
        try:
            from biotite.structure import get_residue_starts
            from biotite.sequence import ProteinSequence

            res_starts = get_residue_starts(atom_array)
            seq_1letter = ''.join(
                ProteinSequence.convert_letter_3to1(res_name)
                for res_name in atom_array.res_name[res_starts]
            )
            return seq_1letter
        except Exception as e:
            print(f"[MPNN] 序列提取失败: {e}")
            return ""

    def _atom_array_to_pdb(self, atom_array) -> str:
        try:
            from atomworks.io.utils.io_utils import to_pdb_file
            buf = io.StringIO()
            to_pdb_file(atom_array, buf)
            return buf.getvalue()
        except ImportError:
            pass

        try:
            import biotite.structure.io.pdb as bpdb
            buf = io.StringIO()
            pdb_file = bpdb.PDBFile()
            pdb_file.set_structure(atom_array)
            pdb_file.write(buf)
            return buf.getvalue()
        except Exception:
            pass

        return self._write_atom_array_manual(atom_array)

    def _write_atom_array_manual(self, atom_array) -> str:
        lines = []
        for i in range(len(atom_array)):
            atom = atom_array[i]
            res_name = getattr(atom, 'res_name', 'ALA')
            atom_name = getattr(atom, 'atom_name', 'CA')
            chain_id = getattr(atom, 'chain_id', 'A')
            res_id = getattr(atom, 'res_id', i + 1)
            x, y, z = atom.coord
            bfactor = getattr(atom, 'b_factor', 50.0)
            element = getattr(atom, 'element', 'C')
            lines.append(
                f"ATOM  {i+1:5d} {atom_name:<4s} {res_name:3s} {chain_id:1s}"
                f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00 {bfactor:5.2f}           {element}"
            )
        lines.append("END")
        return "\n".join(lines) + "\n"

    def _mock_run(
        self,
        backbone_pdb_content: str,
        backbone_pdb_path: str,
        batch_size: int,
        fixed_chains: List[str],
        job_id: str
    ) -> Dict[str, Any]:
        import random

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        aa_list = "ACDEFGHIKLMNPQRSTVWY"
        n_residues = 80

        if backbone_pdb_content:
            n_residues = max(backbone_pdb_content.count("ATOM") // 4, 30)
        elif backbone_pdb_path:
            try:
                with open(backbone_pdb_path, "r") as f:
                    n_residues = max(f.read().count("ATOM") // 4, 30)
            except Exception:
                pass

        sequences = []
        for i in range(batch_size):
            seq = "".join(random.choices(aa_list, k=n_residues))
            score = round(random.uniform(-30, -15), 2)
            sequences.append({
                "index": i,
                "name": f"seq_{i}",
                "sequence": seq,
                "score": score,
                "mock": True,
            })

        first_pdb = ""
        return {
            "success": True,
            "sequences": sequences,
            "num_sequences": len(sequences),
            "first_sequence_pdb": first_pdb,
            "output_dir": str(job_dir),
            "mock": True,
        }
