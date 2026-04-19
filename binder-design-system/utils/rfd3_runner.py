"""
RFDiffusion3 调用模块
跨 conda 环境调用: binder(Python3.10) -> foundry(Python3.12)
"""
import os
import subprocess
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.conda_bridge import is_foundry_available, run_foundry_cli


class RFD3Runner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rfd3"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        return is_foundry_available()

    def run(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int = 60,
        num_designs: int = 3,
        job_id: str = "default"
    ) -> Dict[str, Any]:
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            return self._mock_run(target_pdb, hotspot_residues, binder_length, num_designs, job_dir)

        contig = f"A{binder_length}"
        hotspot_str = ",".join(hotspot_residues) if hotspot_residues else ""

        overrides = [
            f"inputs={target_pdb}",
            f"out_dir={job_dir}",
            f"diffusion_batch_size={num_designs}",
            "n_batches=1",
        ]

        if hotspot_str:
            overrides.append(f"+specification.hotspot_res=[{hotspot_str}]")
        if contig:
            overrides.append(f"+specification.contigmap.contigs=[{contig}]")

        try:
            result = run_foundry_cli(
                ["rfd3", "design"] + overrides,
                timeout=1800
            )

            if result.returncode != 0:
                stderr = result.stderr or ""
                if "Invalid checkpoint" in stderr or "could not find checkpoint" in stderr:
                    return {
                        "success": False,
                        "error": "RFD3模型权重未下载。请在foundry环境下运行:\n  conda activate foundry\n  foundry install rfd3 --checkpoint-dir ./checkpoints",
                        "designs": []
                    }
                return {
                    "success": False,
                    "error": stderr[-500:] if stderr else "Unknown error",
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
        except RuntimeError:
            return self._mock_run(target_pdb, hotspot_residues, binder_length, num_designs, job_dir)
        except Exception as e:
            return {"success": False, "error": str(e), "designs": []}

    def _collect_results(self, job_dir: Path) -> List[Dict]:
        design_files = sorted(glob.glob(str(job_dir / "**/*.pdb"), recursive=True))
        if not design_files:
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
        designs.sort(key=lambda x: x.get("plddt", 0), reverse=True)
        selected = designs[:top_k]
        for i, d in enumerate(selected):
            d["rank"] = i + 1
        return selected

    def _parse_plddt_from_pdb(self, pdb_path: str) -> float:
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
