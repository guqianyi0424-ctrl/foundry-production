"""
RoseTTAFold3 调用模块
完全对齐官方 all.ipynb API 用法
支持 from_atom_array 输入 + RMSD 骨架对比 + 置信度指标
"""
import os
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


class RF3Runner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rf3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        try:
            from rf3.inference_engines.rf3 import RF3InferenceEngine
            print("[RF3] ✅ API可用 (foundry)")
            return True
        except ImportError as e:
            print(f"[RF3] ❌ API不可用，将使用Mock模式: {e}")
            return False

    def is_available(self) -> bool:
        return self._api_available

    def run_rf3(
        self,
        mpnn_pdb_content: str = None,
        rfd3_pdb_content: str = None,
        example_id: str = "binder_design",
        job_id: str = "default"
    ) -> Dict[str, Any]:
        if self._api_available:
            try:
                return self._run_api(
                    mpnn_pdb_content, rfd3_pdb_content, example_id, job_id
                )
            except Exception as e:
                print(f"[RF3] API调用失败: {e}")
                import traceback
                traceback.print_exc()

        return self._mock_run(mpnn_pdb_content, rfd3_pdb_content, example_id, job_id)

    def _run_api(
        self,
        mpnn_pdb_content: str,
        rfd3_pdb_content: str,
        example_id: str,
        job_id: str
    ) -> Dict[str, Any]:
        from rf3.inference_engines.rf3 import RF3InferenceEngine
        from rf3.utils.inference import InferenceInput

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        inference_engine = RF3InferenceEngine(ckpt_path='rf3', verbose=False)

        if mpnn_pdb_content:
            atom_array = self._pdb_to_atom_array(mpnn_pdb_content)
            if atom_array is not None:
                input_structure = InferenceInput.from_atom_array(
                    atom_array, example_id=example_id
                )
            else:
                return {"success": False, "error": "无法解析MPNN PDB"}
        else:
            return {"success": False, "error": "需要MPNN结构作为输入"}

        print(f"[RF3] example_id={example_id} | 输入: MPNN atom_array")

        rf3_outputs = inference_engine.run(inputs=input_structure)

        output_key = next(iter(rf3_outputs.keys()))
        num_models = len(rf3_outputs[output_key])
        rf3_output = rf3_outputs[output_key][0]

        predicted_pdb = self._atom_array_to_pdb(rf3_output.atom_array)
        predicted_path = job_dir / "rf3_predicted.pdb"
        with open(predicted_path, "w") as f:
            f.write(predicted_pdb)

        summary = rf3_output.summary_confidences
        conf = rf3_output.confidences

        plddt_list = []
        if conf and 'atom_plddts' in conf:
            plddt_list = np.round(conf['atom_plddts'], 2).tolist()

        pae_data = None
        if conf and 'pae' in conf:
            pae_data = conf['pae']

        rmsd_value = -1.0
        rmsd_interpretation = "N/A"
        per_res_rmsd = []
        if rfd3_pdb_content:
            rmsd_value = self._calculate_rmsd_pdb(rfd3_pdb_content, predicted_pdb)
            if rmsd_value >= 0:
                if rmsd_value < 1.0:
                    rmsd_interpretation = "Excellent"
                elif rmsd_value < 2.0:
                    rmsd_interpretation = "Good"
                else:
                    rmsd_interpretation = "Moderate"
                per_res_rmsd = self._calculate_per_res_rmsd_pdb(rfd3_pdb_content, predicted_pdb)

        try:
            from atomworks.io.utils.io_utils import to_cif_file
            generated_cif = job_dir / "generated.cif"
            refolded_cif = job_dir / "refolded.cif"
            if rfd3_pdb_content:
                rfd3_aa = self._pdb_to_atom_array(rfd3_pdb_content)
                if rfd3_aa is not None:
                    to_cif_file(rfd3_aa, str(generated_cif))
            to_cif_file(rf3_output.atom_array, str(refolded_cif))
        except Exception:
            pass

        return {
            "success": True,
            "predicted_pdb": predicted_pdb,
            "predicted_pdb_path": str(predicted_path),
            "num_models": num_models,
            "summary": {
                "chain_ptm": summary.get("chain_ptm", []),
                "overall_plddt": summary.get("overall_plddt", 0.0),
                "overall_pde": summary.get("overall_pde", 0.0),
                "overall_pae": summary.get("overall_pae", 0.0),
                "ptm": summary.get("ptm", 0.0),
                "iptm": summary.get("iptm", 0.0),
                "has_clash": summary.get("has_clash", False),
                "ranking_score": summary.get("ranking_score", 0.0),
            },
            "pae": pae_data,
            "plddt": plddt_list,
            "avg_plddt": round(float(summary.get("overall_plddt", 0.0)), 1),
            "rmsd": round(rmsd_value, 2),
            "rmsd_interpretation": rmsd_interpretation,
            "per_res_rmsd": per_res_rmsd,
            "passed": rmsd_value >= 0 and rmsd_value < 2.0,
            "output_dir": str(job_dir),
        }

    def _pdb_to_atom_array(self, pdb_content: str):
        try:
            import biotite.structure.io.pdb as bpdb
            import biotite.structure as bs
            pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_content))
            atom_array = pdb_file.get_structure(model=1)
            atom_array = atom_array[bs.filter_amino_acids(atom_array)]
            return atom_array
        except Exception as e:
            print(f"[RF3] PDB解析失败: {e}")
            return None

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

    def _calculate_rmsd_pdb(self, pdb1_content: str, pdb2_content: str) -> float:
        try:
            coords1 = self._read_ca_coords(pdb1_content)
            coords2 = self._read_ca_coords(pdb2_content)
            if len(coords1) == 0 or len(coords2) == 0:
                return -1.0
            min_len = min(len(coords1), len(coords2))
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]

            centroid1 = coords1.mean(axis=0)
            centroid2 = coords2.mean(axis=0)
            coords1 -= centroid1
            coords2 -= centroid2

            H = coords1.T @ coords2
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            coords2_aligned = (R @ coords2.T).T
            diff = coords1 - coords2_aligned
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            return round(float(rmsd), 3)
        except Exception as e:
            print(f"[RF3] RMSD计算失败: {e}")
            return -1.0

    def _calculate_per_res_rmsd_pdb(self, pdb1_content: str, pdb2_content: str) -> List[float]:
        try:
            coords1 = self._read_ca_coords(pdb1_content)
            coords2 = self._read_ca_coords(pdb2_content)
            min_len = min(len(coords1), len(coords2))
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]

            centroid1 = coords1.mean(axis=0)
            centroid2 = coords2.mean(axis=0)
            coords1 -= centroid1
            coords2 -= centroid2

            H = coords1.T @ coords2
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            coords2_aligned = (R @ coords2.T).T
            diff = coords1 - coords2_aligned
            per_res = np.sqrt(np.sum(diff ** 2, axis=1))
            return [round(float(r), 3) for r in per_res]
        except Exception:
            return []

    def _read_ca_coords(self, pdb_content: str) -> np.ndarray:
        coords = []
        for line in pdb_content.split('\n'):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
        return np.array(coords)

    def _mock_run(
        self,
        mpnn_pdb_content: str,
        rfd3_pdb_content: str,
        example_id: str,
        job_id: str
    ) -> Dict[str, Any]:
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        n_res = 80
        if mpnn_pdb_content:
            ca_count = sum(1 for line in mpnn_pdb_content.split('\n')
                          if line.startswith("ATOM") and "CA" in line[12:16])
            if ca_count > 0:
                n_res = ca_count

        np.random.seed(42)
        coords = np.zeros((n_res, 3))
        for i in range(1, n_res):
            angle = np.random.uniform(-0.2, 0.2)
            step = 3.8
            coords[i] = coords[i - 1] + [
                step * np.cos(angle),
                step * np.sin(angle),
                np.random.uniform(-0.3, 0.3)
            ]

        plddt = [round(np.random.uniform(70, 95), 1) for _ in range(n_res)]
        avg_plddt = round(sum(plddt) / len(plddt), 1)

        predicted_pdb_lines = []
        for i in range(n_res):
            x, y, z = coords[i]
            bfactor = plddt[i]
            predicted_pdb_lines.append(
                f"ATOM  {i*3+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {bfactor:5.1f}           C"
            )
        predicted_pdb_lines.append("END")
        predicted_pdb = "\n".join(predicted_pdb_lines) + "\n"

        predicted_path = job_dir / "rf3_predicted.pdb"
        with open(predicted_path, "w") as f:
            f.write(predicted_pdb)

        rmsd_value = round(np.random.uniform(1.5, 5.0), 2)
        if rmsd_value < 1.0:
            rmsd_interpretation = "Excellent"
        elif rmsd_value < 2.0:
            rmsd_interpretation = "Good"
        else:
            rmsd_interpretation = "Moderate"

        per_res_rmsd = [round(np.random.uniform(0.5, 4.0), 2) for _ in range(n_res)]

        return {
            "success": True,
            "predicted_pdb": predicted_pdb,
            "predicted_pdb_path": str(predicted_path),
            "num_models": 1,
            "summary": {
                "chain_ptm": [0.8, 0.69],
                "overall_plddt": avg_plddt / 100.0,
                "overall_pde": round(np.random.uniform(5, 10), 2),
                "overall_pae": round(np.random.uniform(10, 25), 2),
                "ptm": round(np.random.uniform(0.3, 0.6), 4),
                "iptm": round(np.random.uniform(0.1, 0.4), 4),
                "has_clash": False,
                "ranking_score": round(np.random.uniform(0.2, 0.5), 4),
            },
            "pae": None,
            "plddt": plddt,
            "avg_plddt": avg_plddt,
            "rmsd": rmsd_value,
            "rmsd_interpretation": rmsd_interpretation,
            "per_res_rmsd": per_res_rmsd,
            "passed": rmsd_value < 2.0,
            "output_dir": str(job_dir),
            "mock": True,
        }
