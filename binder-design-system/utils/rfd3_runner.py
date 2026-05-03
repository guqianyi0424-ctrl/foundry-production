"""
RFDiffusion3 调用模块
完全对齐官方 all.ipynb API 用法
支持 binder 设计: target + hotspots + contig
"""
import os
import json
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


class RFD3Runner:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "rfd3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        try:
            from rfd3.engine import RFD3InferenceEngine
            print("[RFD3] ✅ API可用 (foundry)")
            return True
        except ImportError as e:
            print(f"[RFD3] ❌ API不可用，将使用Mock模式: {e}")
            return False

    def is_available(self) -> bool:
        return self._api_available

    def run_rfd3(
        self,
        pdb_content: str = None,
        target: str = None,
        hotspots: List[str] = None,
        binder_length: int = 80,
        length_min: int = 40,
        length_max: int = 120,
        diffusion_batch_size: int = 2,
        n_batches: int = 2,
        job_id: str = "default"
    ) -> Dict[str, Any]:
        if self._api_available:
            try:
                return self._run_api(
                    pdb_content, target, hotspots, binder_length,
                    length_min, length_max, diffusion_batch_size, n_batches, job_id
                )
            except Exception as e:
                print(f"[RFD3] API调用失败: {e}")
                import traceback
                traceback.print_exc()

        return self._mock_run(
            pdb_content, target, hotspots, binder_length,
            length_min, length_max, diffusion_batch_size, n_batches, job_id
        )

    def _run_api(
        self,
        pdb_content: str,
        target: str,
        hotspots: List[str],
        binder_length: int,
        length_min: int,
        length_max: int,
        diffusion_batch_size: int,
        n_batches: int,
        job_id: str
    ) -> Dict[str, Any]:
        from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
        from lightning.fabric import seed_everything

        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(42)

        specification = {
            'length': binder_length,
            'extra': {}
        }

        if pdb_content and target:
            input_pdb_path = job_dir / "input_target.pdb"
            with open(input_pdb_path, "w") as f:
                f.write(pdb_content)

            chains_info = self._parse_target(target)
            contig = self._build_binder_contig(chains_info, binder_length)

            specification['extra']['input'] = str(input_pdb_path)
            specification['extra']['contig'] = contig

            if hotspots:
                hotspot_dict = {}
                for hs in hotspots:
                    hs_clean = hs.replace("/", "").strip()
                    hotspot_dict[hs_clean] = "ALL"
                if hotspot_dict:
                    specification['extra']['select_hotspots'] = hotspot_dict
                    specification['extra']['infer_ori_strategy'] = 'hotspots'

        config = RFD3InferenceConfig(
            specification=specification,
            diffusion_batch_size=diffusion_batch_size,
        )

        print(f"[RFD3] target={target} | hotspots={hotspots} | length={binder_length} | batches={n_batches}x{diffusion_batch_size}")

        model = RFD3InferenceEngine(**config)
        outputs = model.run(
            inputs=None,
            out_dir=None,
            n_batches=n_batches,
        )

        designs = []
        batch_info = []
        for idx, data in outputs.items():
            batch_designs = []
            for i, item in enumerate(data):
                atom_array = item.atom_array
                pdb_str = self._atom_array_to_pdb(atom_array)
                design_name = f"batch{idx}_design{i}"
                pdb_path = job_dir / f"{design_name}.pdb"
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)

                plddt = self._extract_plddt(atom_array)
                design = {
                    "index": len(designs),
                    "batch": idx,
                    "design_in_batch": i,
                    "name": design_name,
                    "pdb_path": str(pdb_path),
                    "pdb_content": pdb_str,
                    "plddt": plddt,
                }
                designs.append(design)
                batch_designs.append(design)

            batch_info.append({
                "batch_idx": idx,
                "num_structures": len(batch_designs),
                "designs": batch_designs,
            })

        first_key = next(iter(outputs.keys()))
        first_atom_array = outputs[first_key][0].atom_array
        first_pdb = self._atom_array_to_pdb(first_atom_array)

        return {
            "success": True,
            "designs": designs,
            "batches": batch_info,
            "num_batches": len(batch_info),
            "num_designs": len(designs),
            "first_backbone_pdb": first_pdb,
            "output_dir": str(job_dir),
        }

    def _parse_target(self, target: str) -> Dict[str, tuple]:
        chains = {}
        if not target:
            return chains
        parts = target.split(",")
        for part in parts:
            part = part.strip()
            if "/" in part:
                chain_id, res_range = part.split("/", 1)
            else:
                chain_id = part[0]
                res_range = part[1:]
            if "-" in res_range:
                start, end = res_range.split("-")
                chains[chain_id] = (int(start), int(end))
            else:
                chains[chain_id] = (int(res_range), int(res_range))
        return chains

    def _build_binder_contig(self, chains_info: Dict[str, tuple], binder_length: int) -> str:
        if not chains_info:
            return f"{binder_length}"
        chain_parts = []
        for chain_id in sorted(chains_info.keys()):
            res_start, res_end = chains_info[chain_id]
            chain_parts.append(f"{chain_id}{res_start}-{res_end}")
        return f"{binder_length},/0," + ",".join(chain_parts)

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
            import biotite.structure as bs
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

    def _extract_plddt(self, atom_array) -> float:
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

    def _mock_run(
        self,
        pdb_content: str,
        target: str,
        hotspots: List[str],
        binder_length: int,
        length_min: int,
        length_max: int,
        diffusion_batch_size: int,
        n_batches: int,
        job_id: str
    ) -> Dict[str, Any]:
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        designs = []
        batch_info = []
        for batch_idx in range(n_batches):
            batch_designs = []
            for design_idx in range(diffusion_batch_size):
                mock_pdb = self._generate_mock_backbone(binder_length, batch_idx, design_idx, job_dir)
                with open(mock_pdb, "r") as f:
                    pdb_content_str = f.read()
                plddt = round(np.random.uniform(75, 95), 1)
                design = {
                    "index": len(designs),
                    "batch": batch_idx,
                    "design_in_batch": design_idx,
                    "name": f"batch{batch_idx}_design{design_idx}",
                    "pdb_path": str(mock_pdb),
                    "pdb_content": pdb_content_str,
                    "plddt": plddt,
                    "mock": True,
                }
                designs.append(design)
                batch_designs.append(design)
            batch_info.append({
                "batch_idx": batch_idx,
                "num_structures": len(batch_designs),
                "designs": batch_designs,
            })

        first_pdb = designs[0]["pdb_content"] if designs else ""

        return {
            "success": True,
            "designs": designs,
            "batches": batch_info,
            "num_batches": len(batch_info),
            "num_designs": len(designs),
            "first_backbone_pdb": first_pdb,
            "output_dir": str(job_dir),
            "mock": True,
        }

    def _generate_mock_backbone(self, length: int, batch_idx: int, design_idx: int, job_dir: Path) -> Path:
        np.random.seed(42 + batch_idx * 100 + design_idx)
        coords = np.zeros((length, 3))
        for i in range(1, length):
            angle = np.random.uniform(-0.3, 0.3)
            step = 3.8
            coords[i] = coords[i - 1] + [
                step * np.cos(angle),
                step * np.sin(angle),
                np.random.uniform(-0.5, 0.5)
            ]
        pdb_path = job_dir / f"batch{batch_idx}_design{design_idx}_backbone.pdb"
        with open(pdb_path, "w") as f:
            for i in range(length):
                x, y, z = coords[i]
                bfactor = 50.0 + np.random.uniform(0, 40)
                f.write(f"ATOM  {i * 3 + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {bfactor:5.2f}           C\n")
            f.write("END\n")
        return pdb_path
