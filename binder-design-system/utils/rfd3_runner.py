"""
RFDiffusion3 调用模块
支持 Top-K 设计选择和云服务器路径
"""
import os
import subprocess
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class RFD3Runner:
    """RFDiffusion3 运行器"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.rfd3_path = self._find_rfd3()
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rfd3"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_rfd3(self) -> Optional[Path]:
        possible = [
            self.base_path / "RFdiffusion",
            self.base_path / "rfdiffusion",
            self.base_path / "RFdiffusion-main",
            Path("/workspace/RFdiffusion"),
            Path("/opt/RFdiffusion"),
            Path(os.getenv("RFD3_PATH", "/nonexistent")),
        ]
        for p in possible:
            if p.exists():
                return p
        return None

    def is_available(self) -> bool:
        if self.rfd3_path is None:
            return False
        return (self.rfd3_path / "run_inference.py").exists()

    def run(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int = 60,
        num_designs: int = 3,
        job_id: str = "default"
    ) -> Dict[str, Any]:
        """
        运行RFDiffusion3生成Binder主链

        Args:
            target_pdb: 目标蛋白PDB文件路径
            hotspot_residues: 热点残基列表 (如 ["A45", "A67"])
            binder_length: Binder长度 (aa)
            num_designs: 生成数量 (Top-K)
            job_id: 任务ID

        Returns:
            结果字典，包含Top-K设计
        """
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            return self._mock_run(target_pdb, hotspot_residues, binder_length, num_designs, job_dir)

        contig = f"A{binder_length}"
        hotspot_str = ",".join(hotspot_residues) if hotspot_residues else ""

        cmd = [
            "python", str(self.rfd3_path / "run_inference.py"),
            "--pdb", target_pdb,
            "--contigmap.contigs", contig,
            "--inference.num_designs", str(num_designs),
            "--output_dir", str(job_dir),
        ]

        if hotspot_str:
            cmd.extend(["--hotspot_residues", hotspot_str])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env={**os.environ, "PYTHONPATH": str(self.rfd3_path)}
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr[-500:] if result.stderr else "Unknown error",
                    "designs": []
                }

            designs = self._collect_results(job_dir)
            designs = self._rank_and_select_top_k(designs, num_designs)

            return {
                "success": True,
                "designs": designs,
                "num_designs": len(designs),
                "output_dir": str(job_dir)
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "RFD3运行超时(30min)", "designs": []}
        except Exception as e:
            return {"success": False, "error": str(e), "designs": []}

    def _collect_results(self, job_dir: Path) -> List[Dict]:
        design_files = sorted(glob.glob(str(job_dir / "*.pdb")))
        designs = []
        for i, pdb_path in enumerate(design_files):
            plddt = self._parse_plddt_from_pdb(pdb_path)
            designs.append({
                "index": i,
                "pdb_path": pdb_path,
                "pdb_name": Path(pdb_path).name,
                "plddt": plddt
            })
        return designs

    def _rank_and_select_top_k(self, designs: List[Dict], top_k: int) -> List[Dict]:
        """按pLDDT排序并选取Top-K"""
        designs.sort(key=lambda x: x.get("plddt", 0), reverse=True)
        selected = designs[:top_k]
        for i, d in enumerate(selected):
            d["rank"] = i + 1
        return selected

    def _parse_plddt_from_pdb(self, pdb_path: str) -> float:
        """从PDB文件中解析pLDDT分数"""
        try:
            scores = []
            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try:
                            bfactor = float(line[60:66].strip())
                            scores.append(bfactor)
                        except (ValueError, IndexError):
                            pass
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _mock_run(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int,
        num_designs: int,
        job_dir: Path
    ) -> Dict[str, Any]:
        import numpy as np

        designs = []
        for i in range(num_designs):
            mock_pdb = self._generate_mock_backbone(binder_length, i, job_dir)
            plddt = round(np.random.uniform(75, 95), 1)
            designs.append({
                "index": i,
                "pdb_path": str(mock_pdb),
                "pdb_name": mock_pdb.name,
                "mock": True,
                "plddt": plddt,
                "rank": i + 1
            })

        designs.sort(key=lambda x: x.get("plddt", 0), reverse=True)
        for i, d in enumerate(designs):
            d["rank"] = i + 1

        return {
            "success": True,
            "designs": designs,
            "num_designs": num_designs,
            "output_dir": str(job_dir),
            "mock": True
        }

    def _generate_mock_backbone(self, length: int, design_idx: int, job_dir: Path) -> Path:
        import numpy as np

        np.random.seed(42 + design_idx)

        coords = np.zeros((length, 3))
        for i in range(1, length):
            angle = np.random.uniform(-0.3, 0.3)
            step = 3.8
            coords[i] = coords[i - 1] + [
                step * np.cos(angle),
                step * np.sin(angle),
                np.random.uniform(-0.5, 0.5)
            ]

        pdb_path = job_dir / f"design_{design_idx}_backbone.pdb"
        with open(pdb_path, "w") as f:
            for i in range(length):
                x, y, z = coords[i]
                bfactor = 50.0 + np.random.uniform(0, 40)
                f.write(f"ATOM  {i * 3 + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {bfactor:5.2f}           C\n")
            f.write("END\n")

        return pdb_path
