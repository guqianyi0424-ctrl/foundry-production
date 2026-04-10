"""
RoseTTAFold3 调用模块
支持结构预测和RMSD验证
"""
import os
import subprocess
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class RF3Runner:
    """RoseTTAFold3 运行器"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.rf3_path = self._find_rf3()
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rf3"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_rf3(self) -> Optional[Path]:
        possible = [
            self.base_path / "RoseTTAFold3",
            self.base_path / "rosettafold3",
            self.base_path / "RF3",
            Path("/workspace/RoseTTAFold3"),
            Path("/opt/RoseTTAFold3"),
            Path(os.getenv("RF3_PATH", "/nonexistent")),
        ]
        for p in possible:
            if p.exists():
                return p
        return None

    def is_available(self) -> bool:
        if self.rf3_path is None:
            return False
        return (self.rf3_path / "run_rf3.py").exists() or \
               (self.rf3_path / "predict.py").exists()

    def run(
        self,
        sequence: str,
        job_id: str = "default"
    ) -> Dict[str, Any]:
        """
        运行RoseTTAFold3预测结构

        Args:
            sequence: 氨基酸序列
            job_id: 任务ID

        Returns:
            结果字典
        """
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = job_dir / "input.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">binder\n{sequence}\n")

        if not self.is_available():
            return self._mock_run(sequence, job_dir)

        run_script = self.rf3_path / "run_rf3.py"
        if not run_script.exists():
            run_script = self.rf3_path / "predict.py"

        cmd = [
            "python", str(run_script),
            "--input", str(fasta_path),
            "--output_dir", str(job_dir),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                env={**os.environ, "PYTHONPATH": str(self.rf3_path)}
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr[-500:] if result.stderr else "Unknown error",
                    "pdb_path": None
                }

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

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "RF3运行超时(60min)", "pdb_path": None}
        except Exception as e:
            return {"success": False, "error": str(e), "pdb_path": None}

    def _parse_pae(self, job_dir: Path) -> Optional[List[List[float]]]:
        json_files = glob.glob(str(job_dir / "*pae*.json"))
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _parse_plddt(self, job_dir: Path) -> Optional[List[float]]:
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
        """计算两个PDB结构的RMSD (Kabsch算法)"""
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
        """计算每个残基的RMSD"""
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
        """
        完整验证流程: RF3预测 + RMSD计算

        Args:
            backbone_pdb: RFD3生成的主链PDB
            sequence: MPNN设计的序列
            job_id: 任务ID
            rmsd_threshold: RMSD阈值

        Returns:
            验证结果字典
        """
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
