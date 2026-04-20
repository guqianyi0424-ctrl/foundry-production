"""
RoseTTAFold3 调用模块
优先使用 foundry Python API (RF3InferenceEngine)
回退到 conda_bridge CLI 调用
最终回退到 mock 模式
"""
import os
import subprocess
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.conda_bridge import is_foundry_available, run_foundry_cli


class RF3Runner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rf3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        try:
            from rf3.inference_engines.rf3 import RF3InferenceEngine
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._api_available or is_foundry_available()

    def run(
        self,
        sequence: str,
        job_id: str = "default"
    ) -> Dict[str, Any]:
        if self._api_available:
            try:
                return self._run_api(sequence, job_id)
            except Exception as e:
                print(f"[RF3] API调用失败，回退CLI: {e}")

        if is_foundry_available():
            try:
                return self._run_cli(sequence, job_id)
            except Exception as e:
                print(f"[RF3] CLI调用失败，回退mock: {e}")

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return self._mock_run(sequence, job_dir)

    def _run_api(
        self,
        sequence: str,
        job_id: str
    ) -> Dict[str, Any]:
        from rf3.inference_engines.rf3 import RF3InferenceEngine
        from rf3.utils.inference import InferenceInput

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        inference_engine = RF3InferenceEngine(ckpt_path='rf3', verbose=False)

        input_structure = InferenceInput.from_sequence(
            sequence=sequence,
            example_id=f"binder_{job_id}"
        )
        rf3_outputs = inference_engine.run(inputs=input_structure)

        output_key = next(iter(rf3_outputs.keys()))
        rf3_output = rf3_outputs[output_key][0]

        atom_array = rf3_output.atom_array
        pdb_path = job_dir / "rf3_predicted.pdb"
        self._save_atom_array(atom_array, str(pdb_path))

        summary = rf3_output.summary_confidences
        conf = rf3_output.confidences

        avg_plddt = summary.get('overall_plddt', 0.0)
        plddt_list = []
        if conf and 'atom_plddts' in conf:
            import numpy as np
            plddt_list = np.round(conf['atom_plddts'], 2).tolist()

        pae_data = None
        if conf and 'pae' in conf:
            pae_data = conf['pae']

        return {
            "success": True,
            "pdb_path": str(pdb_path),
            "pae": pae_data,
            "plddt": plddt_list,
            "avg_plddt": round(float(avg_plddt), 1),
            "ptm": summary.get('ptm', 0.0),
            "ranking_score": summary.get('ranking_score', 0.0),
            "output_dir": str(job_dir)
        }

    def _run_cli(
        self,
        sequence: str,
        job_id: str
    ) -> Dict[str, Any]:
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = job_dir / "input.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">binder\n{sequence}\n")

        result = run_foundry_cli(
            ["rf3", "fold", f"inputs={fasta_path}", f"out_dir={job_dir}"],
            timeout=3600
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
                "pdb_path": None
            }

        pdb_files = sorted(glob.glob(str(job_dir / "**/*.pdb"), recursive=True))
        if not pdb_files:
            pdb_files = sorted(glob.glob(str(job_dir / "*.pdb")))
        if not pdb_files:
            return {"success": False, "error": "未生成PDB文件", "pdb_path": None}

        pae_data = self._parse_pae(job_dir)
        plddt = self._parse_plddt(job_dir)
        avg_plddt = sum(plddt) / len(plddt) if plddt else 0.0

        return {
            "success": True,
            "pdb_path": pdb_files[0],
            "pae": pae_data,
            "plddt": plddt,
            "avg_plddt": round(avg_plddt, 1),
            "output_dir": str(job_dir)
        }

    def _save_atom_array(self, atom_array, pdb_path: str):
        try:
            from atomworks.io.utils.io_utils import to_pdb_file
            to_pdb_file(atom_array, pdb_path)
        except ImportError:
            try:
                from atomworks.io.utils.io_utils import to_cif_file
                cif_path = pdb_path.replace('.pdb', '.cif')
                to_cif_file(atom_array, cif_path)
            except ImportError:
                self._write_atom_array_manual(atom_array, pdb_path)

    def _write_atom_array_manual(self, atom_array, pdb_path: str):
        with open(pdb_path, "w") as f:
            for i in range(len(atom_array)):
                atom = atom_array[i]
                res_name = atom.res_name if hasattr(atom, 'res_name') else 'ALA'
                atom_name = atom.atom_name if hasattr(atom, 'atom_name') else 'CA'
                chain_id = atom.chain_id if hasattr(atom, 'chain_id') else 'A'
                res_id = atom.res_id if hasattr(atom, 'res_id') else i + 1
                x, y, z = atom.coord
                bfactor = atom.b_factor if hasattr(atom, 'b_factor') else 50.0
                element = atom.element if hasattr(atom, 'element') else 'C'
                f.write(
                    f"ATOM  {i+1:5d} {atom_name:<4s} {res_name:3s} {chain_id:1s}"
                    f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00 {bfactor:5.2f}           {element}\n"
                )
            f.write("END\n")

    def _parse_pae(self, job_dir: Path) -> Optional[List[List[float]]]:
        json_files = glob.glob(str(job_dir / "**/*pae*.json"), recursive=True)
        if not json_files:
            json_files = glob.glob(str(job_dir / "*pae*.json"))
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _parse_plddt(self, job_dir: Path) -> Optional[List[float]]:
        json_files = glob.glob(str(job_dir / "**/*plddt*.json"), recursive=True)
        if not json_files:
            json_files = glob.glob(str(job_dir / "*plddt*.json"))
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _mock_run(self, sequence: str, job_dir: Path) -> Dict[str, Any]:
        import numpy as np

        n_res = len(sequence)
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

        pdb_path = job_dir / "rf3_predicted.pdb"
        with open(pdb_path, "w") as f:
            for i, (aa, (x, y, z)) in enumerate(zip(sequence, coords)):
                res_name = self._aa1to3(aa)
                bfactor = round(np.random.uniform(70, 95), 1)
                f.write(f"ATOM  {i * 3 + 1:5d}  CA  {res_name:3s} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {bfactor:5.1f}           C\n")
            f.write("END\n")

        plddt = [round(np.random.uniform(70, 95), 1) for _ in range(n_res)]
        avg_plddt = round(sum(plddt) / len(plddt), 1)
        pae = [[round(np.random.uniform(1, 15), 1) for _ in range(n_res)] for _ in range(n_res)]

        return {
            "success": True,
            "pdb_path": str(pdb_path),
            "pae": pae,
            "plddt": plddt,
            "avg_plddt": avg_plddt,
            "output_dir": str(job_dir),
            "mock": True
        }

    @staticmethod
    def _aa1to3(aa: str) -> str:
        mapping = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        return mapping.get(aa, 'ALA')

    @staticmethod
    def calculate_rmsd(pdb1_path: str, pdb2_path: str) -> float:
        try:
            import numpy as np

            def read_ca_coords(pdb_path):
                coords = []
                with open(pdb_path, "r") as f:
                    for line in f:
                        if line.startswith("ATOM") and "CA" in line[12:16]:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                return np.array(coords)

            coords1 = read_ca_coords(pdb1_path)
            coords2 = read_ca_coords(pdb2_path)

            if len(coords1) != len(coords2):
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
            print(f"RMSD计算失败: {e}")
            return -1.0

    @staticmethod
    def calculate_per_residue_rmsd(pdb1_path: str, pdb2_path: str) -> List[float]:
        try:
            import numpy as np

            def read_ca_coords(pdb_path):
                coords = []
                with open(pdb_path, "r") as f:
                    for line in f:
                        if line.startswith("ATOM") and "CA" in line[12:16]:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                return np.array(coords)

            coords1 = read_ca_coords(pdb1_path)
            coords2 = read_ca_coords(pdb2_path)

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
            per_res_rmsd = np.sqrt(np.sum(diff ** 2, axis=1))

            return [round(float(r), 3) for r in per_res_rmsd]

        except Exception:
            return []

    def validate_design(
        self,
        backbone_pdb: str,
        sequence: str,
        job_id: str = "default",
        rmsd_threshold: float = 2.0
    ) -> Dict[str, Any]:
        rf3_result = self.run(sequence, job_id)

        if not rf3_result["success"]:
            return {
                "success": False,
                "error": rf3_result.get("error", "RF3预测失败"),
                "rmsd": -1.0,
                "passed": False
            }

        rmsd_val = -1.0
        per_res_rmsd = []
        if rf3_result.get("pdb_path") and os.path.exists(rf3_result["pdb_path"]):
            rmsd_val = self.calculate_rmsd(backbone_pdb, rf3_result["pdb_path"])
            per_res_rmsd = self.calculate_per_residue_rmsd(backbone_pdb, rf3_result["pdb_path"])

        return {
            "success": True,
            "rf3_pdb": rf3_result.get("pdb_path"),
            "rmsd": rmsd_val,
            "per_res_rmsd": per_res_rmsd,
            "plddt": rf3_result.get("plddt"),
            "avg_plddt": rf3_result.get("avg_plddt", 0),
            "pae": rf3_result.get("pae"),
            "passed": rmsd_val >= 0 and rmsd_val < rmsd_threshold,
            "mock": rf3_result.get("mock", False)
        }
