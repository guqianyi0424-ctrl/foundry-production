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
        self._ckpt_path = self._find_checkpoint()

    def _find_checkpoint(self) -> Optional[str]:
        search_paths = [
            self.base_path / "foundry-production" / "checkpoints" / "rfd3_latest.ckpt",
            self.base_path / "checkpoints" / "rfd3_latest.ckpt",
            Path(os.path.expanduser("~/.foundry/checkpoints/rfd3_latest.ckpt")),
            Path("/root/.foundry/checkpoints/rfd3_latest.ckpt"),
        ]
        for p in search_paths:
            if p.exists():
                return str(p)
        ckpt_env = os.environ.get("FOUNDRY_CHECKPOINT_DIRS", "")
        if ckpt_env:
            for d in ckpt_env.split(os.pathsep):
                p = Path(d) / "rfd3_latest.ckpt"
                if p.exists():
                    return str(p)
        return None

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

        yaml_path = self._generate_input_yaml(target_pdb, hotspot_residues, binder_length, job_dir)

        overrides = [
            f"inputs={yaml_path}",
            f"out_dir={job_dir}",
            f"diffusion_batch_size={num_designs}",
            "n_batches=1",
        ]

        if self._ckpt_path:
            overrides.append(f"ckpt_path={self._ckpt_path}")

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

    def _parse_pdb_chains(self, pdb_path: str) -> Dict[str, tuple]:
        chains = {}
        try:
            with open(pdb_path, "r") as f:
                for line in f:
                    if not line.startswith("ATOM") and not line.startswith("HETATM"):
                        continue
                    if len(line) < 22:
                        continue
                    chain_id = line[21].strip()
                    if not chain_id:
                        continue
                    try:
                        res_id = int(line[22:26].strip())
                    except (ValueError, IndexError):
                        continue
                    if chain_id not in chains:
                        chains[chain_id] = [res_id, res_id]
                    else:
                        if res_id < chains[chain_id][0]:
                            chains[chain_id][0] = res_id
                        if res_id > chains[chain_id][1]:
                            chains[chain_id][1] = res_id
        except Exception as e:
            print(f"[RFD3] PDB解析失败: {e}")
        return chains

    def _generate_input_yaml(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int,
        job_dir: Path
    ) -> str:
        chains = self._parse_pdb_chains(target_pdb)

        if not chains:
            print("[RFD3] ⚠️ 无法解析PDB链信息，使用默认contig")
            contig = f"{binder_length},/0,A1-999"
        else:
            chain_parts = []
            for chain_id in sorted(chains.keys()):
                res_start, res_end = chains[chain_id]
                chain_parts.append(f"{chain_id}{res_start}-{res_end}")
            contig = f"{binder_length},/0," + ",".join(chain_parts)

        validated_hotspots = []
        if hotspot_residues and chains:
            for hs in hotspot_residues:
                hs_chain = None
                hs_resid = None
                for c in hs:
                    if c.isalpha():
                        hs_chain = c
                    elif c.isdigit():
                        idx = hs.index(c)
                        hs_chain = hs[:idx]
                        hs_resid = hs[idx:]
                        break
                if hs_chain and hs_resid and hs_chain in chains:
                    res_start, res_end = chains[hs_chain]
                    if res_start <= int(hs_resid) <= res_end:
                        validated_hotspots.append(hs)
                    else:
                        print(f"[RFD3] ⚠️ 热点 {hs} 不在链 {hs_chain} 范围({res_start}-{res_end})内，已跳过")
                elif hs_chain and hs_chain not in chains:
                    print(f"[RFD3] ⚠️ 热点 {hs} 的链 {hs_chain} 不在PDB中，已跳过")
                else:
                    validated_hotspots.append(hs)

        yaml_path = job_dir / "rfd3_input.yaml"
        with open(yaml_path, "w") as f:
            f.write("binder_design:\n")
            f.write(f"  input: {target_pdb}\n")
            f.write(f"  contig: {contig}\n")
            f.write(f"  is_non_loopy: true\n")
            if validated_hotspots:
                f.write("  select_hotspots:\n")
                for res in validated_hotspots:
                    f.write(f"    {res}: ALL\n")
                f.write("  infer_ori_strategy: hotspots\n")

        return str(yaml_path)

    def _collect_results(self, job_dir: Path) -> List[Dict]:
        design_files = sorted(glob.glob(str(job_dir / "**/*.cif.gz"), recursive=True))
        if not design_files:
            design_files = sorted(glob.glob(str(job_dir / "**/*.cif"), recursive=True))
        if not design_files:
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
