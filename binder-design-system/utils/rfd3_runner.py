"""
RFDiffusion3 调用模块
优先使用 foundry Python API (RFD3InferenceEngine)
回退到 conda_bridge CLI 调用
最终回退到 mock 模式
"""
import os
import glob
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.conda_bridge import is_foundry_available, run_foundry_cli


class RFD3Runner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rfd3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        try:
            from rfd3.engine import RFD3InferenceEngine
            return True
        except ImportError:
            return False

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
        return None

    def is_available(self) -> bool:
        return self._api_available or is_foundry_available()

    def run(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int = 60,
        num_designs: int = 3,
        job_id: str = "default"
    ) -> Dict[str, Any]:
        if self._api_available:
            try:
                return self._run_api(target_pdb, hotspot_residues, binder_length, num_designs, job_id)
            except Exception as e:
                print(f"[RFD3] API调用失败，回退CLI: {e}")

        if is_foundry_available():
            try:
                return self._run_cli(target_pdb, hotspot_residues, binder_length, num_designs, job_id)
            except Exception as e:
                print(f"[RFD3] CLI调用失败，回退mock: {e}")

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return self._mock_run(target_pdb, hotspot_residues, binder_length, num_designs, job_dir)

    def _run_api(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int,
        num_designs: int,
        job_id: str
    ) -> Dict[str, Any]:
        from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
        from lightning.fabric import seed_everything

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(42)

        chains = self._parse_pdb_chains(target_pdb)
        contig = self._build_contig(chains, binder_length)

        specification = {
            'length': binder_length,
            'extra': {
                'input': target_pdb,
                'contig': contig,
            }
        }

        if hotspot_residues and chains:
            hotspot_dict = {}
            for hs in hotspot_residues:
                hs_chain, hs_resid = self._parse_hotspot(hs)
                if hs_chain and hs_resid and hs_chain in chains:
                    hotspot_dict[f"{hs_chain}{hs_resid}"] = "ALL"
            if hotspot_dict:
                specification['extra']['select_hotspots'] = hotspot_dict
                specification['extra']['infer_ori_strategy'] = 'hotspots'

        config = RFD3InferenceConfig(
            specification=specification,
            diffusion_batch_size=num_designs,
        )

        model = RFD3InferenceEngine(**config)
        outputs = model.run(
            inputs=None,
            out_dir=str(job_dir),
            n_batches=1,
        )

        designs = []
        for idx, data in outputs.items():
            for i, item in enumerate(data):
                atom_array = item.atom_array
                design_name = f"design_{idx}_{i}"
                pdb_path = job_dir / f"{design_name}.pdb"
                self._save_atom_array(atom_array, str(pdb_path))

                plddt = self._extract_plddt_from_atom_array(atom_array)
                designs.append({
                    "index": len(designs),
                    "pdb_path": str(pdb_path),
                    "pdb_name": pdb_path.name,
                    "plddt": plddt
                })

        designs = self._rank_and_select_top_k(designs, num_designs)

        return {
            "success": True,
            "designs": designs,
            "num_designs": len(designs),
            "output_dir": str(job_dir)
        }

    def _run_cli(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int,
        num_designs: int,
        job_id: str
    ) -> Dict[str, Any]:
        import subprocess

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = self._generate_input_yaml(target_pdb, hotspot_residues, binder_length, job_dir)

        overrides = [
            f"inputs={yaml_path}",
            f"out_dir={job_dir}",
            f"diffusion_batch_size={num_designs}",
            "n_batches=1",
        ]

        result = run_foundry_cli(["rfd3", "design"] + overrides, timeout=1800)

        if result.returncode != 0:
            stderr = result.stderr or ""
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

    def _build_contig(self, chains: Dict[str, tuple], binder_length: int) -> str:
        if not chains:
            return f"{binder_length},/0,A1-999"
        chain_parts = []
        for chain_id in sorted(chains.keys()):
            res_start, res_end = chains[chain_id]
            chain_parts.append(f"{chain_id}{res_start}-{res_end}")
        return f"{binder_length},/0," + ",".join(chain_parts)

    def _parse_hotspot(self, hs: str):
        hs_chain = None
        hs_resid = None
        for i, c in enumerate(hs):
            if c.isdigit():
                hs_chain = hs[:i]
                hs_resid = hs[i:]
                break
        return hs_chain, hs_resid

    def _save_atom_array(self, atom_array, pdb_path: str):
        try:
            from atomworks.io.utils.io_utils import to_pdb_file
            to_pdb_file(atom_array, pdb_path)
        except ImportError:
            try:
                from atomworks.io import to_pdb_file
                to_pdb_file(atom_array, pdb_path)
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

    def _extract_plddt_from_atom_array(self, atom_array) -> float:
        try:
            if hasattr(atom_array, 'b_factor'):
                b_factors = atom_array.b_factor
                ca_mask = (atom_array.atom_name == 'CA') if hasattr(atom_array, 'atom_name') else None
                if ca_mask is not None and ca_mask.any():
                    return float(b_factors[ca_mask].mean())
                return float(b_factors.mean())
        except Exception:
            pass
        return 0.0

    def _generate_input_yaml(
        self,
        target_pdb: str,
        hotspot_residues: List[str],
        binder_length: int,
        job_dir: Path
    ) -> str:
        chains = self._parse_pdb_chains(target_pdb)
        contig = self._build_contig(chains, binder_length)

        validated_hotspots = []
        if hotspot_residues and chains:
            for hs in hotspot_residues:
                hs_chain, hs_resid = self._parse_hotspot(hs)
                if hs_chain and hs_resid and hs_chain in chains:
                    res_start, res_end = chains[hs_chain]
                    if res_start <= int(hs_resid) <= res_end:
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
