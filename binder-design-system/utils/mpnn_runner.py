"""
ProteinMPNN 调用模块
支持 Top-K 序列选择
"""
import os
import subprocess
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any


class MPNNRunner:
    """ProteinMPNN 运行器"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.mpnn_path = self._find_mpnn()
        self.output_dir = self.base_path / "binder-design-system" / "outputs" / "mpnn"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_mpnn(self) -> Optional[Path]:
        possible = [
            self.base_path / "ProteinMPNN",
            self.base_path / "protein_mpnn",
            self.base_path / "ProteinMPNN-main",
            Path("/workspace/ProteinMPNN"),
            Path("/opt/ProteinMPNN"),
            Path(os.getenv("MPNN_PATH", "/nonexistent")),
        ]
        for p in possible:
            if p.exists():
                return p
        return None

    def is_available(self) -> bool:
        if self.mpnn_path is None:
            return False
        return (self.mpnn_path / "protein_mpnn_run.py").exists() or \
               (self.mpnn_path / "run.py").exists()

    def run(
        self,
        backbone_pdb: str,
        num_sequences: int = 3,
        sampling_temp: float = 0.1,
        job_id: str = "default",
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        运行ProteinMPNN设计序列

        Args:
            backbone_pdb: 主链PDB文件路径
            num_sequences: 每个设计生成的序列数
            sampling_temp: 采样温度
            job_id: 任务ID
            top_k: 选取Top-K最优序列

        Returns:
            结果字典，包含Top-K序列
        """
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            return self._mock_run(backbone_pdb, num_sequences, job_dir, top_k)

        run_script = self.mpnn_path / "protein_mpnn_run.py"
        if not run_script.exists():
            run_script = self.mpnn_path / "run.py"

        cmd = [
            "python", str(run_script),
            "--pdb_path", backbone_pdb,
            "--out_folder", str(job_dir),
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", str(sampling_temp),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, "PYTHONPATH": str(self.mpnn_path)}
            )

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

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "MPNN运行超时(10min)", "sequences": []}
        except Exception as e:
            return {"success": False, "error": str(e), "sequences": []}

    def _parse_results(self, job_dir: Path) -> List[Dict]:
        sequences = []

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

        json_files = sorted(glob.glob(str(job_dir / "*.json")))
        for json_path in json_files:
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "sequence" in data:
                    sequences.append({
                        "header": data.get("header", ""),
                        "sequence": data["sequence"],
                        "score": data.get("score", 0.0),
                        "json_path": json_path
                    })
            except Exception:
                pass

        return sequences

    def _rank_and_select_top_k(self, sequences: List[Dict], top_k: int) -> List[Dict]:
        """按MPNN得分排序并选取Top-K (得分越低越好)"""
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
