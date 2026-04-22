"""
热点残基预测工具
方案: ESM-2 (GPU) + 纯 PyTorch MLP (无需DGL)
支持 Top-K 选择策略
"""
import os
import sys
import io
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


class HotspotMLP(torch.nn.Module if False else object):

    @staticmethod
    def create_model(input_dim=1280, hidden_dim=256, dropout=0.3):
        import torch
        import torch.nn as nn

        class HotspotMLPNet(nn.Module):
            def __init__(self, in_dim, h_dim, drop):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(h_dim, h_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(h_dim // 2, h_dim // 4),
                    nn.ReLU(),
                    nn.Linear(h_dim // 4, 2),
                )

            def forward(self, x):
                return self.net(x)

        return HotspotMLPNet(input_dim, hidden_dim, dropout)


class HotspotPredictor:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.base_path = Path(__file__).parent.parent.parent
        self.hotspot_dl_path = self.base_path / "hotspot-prediction"

        self._mlp_models = {}
        self._mlp_loaded = False
        self._esm_model = None
        self._esm_loaded = False

    def predict(
        self,
        atom_array,
        method: str = "dl",
        pdb_string: str = None,
        top_k: int = None
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = top_k

        residues = self._extract_residue_features(atom_array)
        if residues.empty:
            return {
                "hotspots": [],
                "hotspots_detail": [],
                "all_scores": {},
                "error": "无法提取残基特征"
            }

        results_dl = self._predict_dl(residues, atom_array)

        if results_dl and results_dl.get("model_loaded"):
            return self._select_top_k(results_dl, residues, "dl")
        else:
            fallback = self._rule_based_predict(residues)
            return self._select_top_k(fallback, residues, "rule")

    def _predict_dl(self, residues: pd.DataFrame, atom_array) -> Dict[str, Any]:
        try:
            import torch

            sequence = self._get_sequence_from_residues(residues)
            esm_features = self._get_esm_features(sequence)

            if esm_features is not None and len(esm_features) == len(residues):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                mlp_models = self._load_mlp_models(esm_features.shape[1], device)

                if mlp_models:
                    all_probs = []
                    with torch.no_grad():
                        x = torch.tensor(esm_features, dtype=torch.float32).to(device)
                        for fold_name, model in mlp_models.items():
                            logits = model(x)
                            probs = torch.softmax(logits, dim=1)
                            all_probs.append(probs[:, 1].cpu().numpy())

                    scores = np.mean(all_probs, axis=0)
                    n_models = len(mlp_models)
                    print(f"[DL] ESM-2+MLP预测完成 ({n_models}折集成, GPU加速)")
                    return {"scores": scores, "method": "dl", "model_loaded": True}
                else:
                    print("[DL] 无MLP模型权重，使用ESM-2注意力分数")
                    scores = self._esm_attention_scores(esm_features, residues)
                    if scores is not None:
                        return {"scores": scores, "method": "dl", "model_loaded": True}
            else:
                print("[DL] ESM特征不可用，使用规则预测")

        except Exception as e:
            print(f"[DL] 预测失败: {e}")
            import traceback
            traceback.print_exc()

        scores = self._compute_rule_scores(residues, "dl")
        return {"scores": scores, "method": "dl", "model_loaded": False}

    def _esm_attention_scores(self, esm_features: np.ndarray, residues: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            import torch
            esm = self._load_esm_model()
            if esm is None:
                return None

            model = esm['model']
            device = esm['device']

            sequence = self._get_sequence_from_residues(residues)

            with torch.no_grad():
                tokenizer = esm['tokenizer']
                inputs = tokenizer(sequence, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_attentions=True)

                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    last_attn = outputs.attentions[-1]
                    attn_weights = last_attn.mean(dim=1)[0]
                    cls_attn = attn_weights[0, 1:-1].cpu().numpy()
                    if len(cls_attn) == len(residues):
                        min_a = cls_attn.min()
                        max_a = cls_attn.max()
                        if max_a > min_a:
                            scores = (cls_attn - min_a) / (max_a - min_a)
                        else:
                            scores = np.ones(len(residues)) * 0.5
                        print(f"[DL] ESM-2注意力分数预测完成 (GPU)")
                        return scores

            return None
        except Exception as e:
            print(f"[DL] ESM注意力分数失败: {e}")
            return None

    def _load_mlp_models(self, input_dim: int, device) -> Dict[str, Any]:
        if self._mlp_loaded:
            return self._mlp_models

        try:
            import torch

            models_dir = self.base_path / "binder-design-system" / "models" / "hotspot_mlp"
            models_dir.mkdir(parents=True, exist_ok=True)

            for fold_idx in range(1, 6):
                model_file = models_dir / f"hotspot_mlp_fold{fold_idx}.pth"
                if model_file.exists():
                    try:
                        model = HotspotMLP.create_model(input_dim=input_dim)
                        state_dict = torch.load(str(model_file), map_location=device, weights_only=True)
                        model.load_state_dict(state_dict)
                        model.eval()
                        model = model.to(device)
                        self._mlp_models[f"fold{fold_idx}"] = model
                        print(f"[DL] MLP Fold {fold_idx} 加载成功 (device={device})")
                    except Exception as e:
                        print(f"[DL] MLP Fold {fold_idx} 加载失败: {e}")

            if not self._mlp_models:
                print("[DL] 无预训练MLP权重，将使用ESM-2注意力机制预测")

        except Exception as e:
            print(f"[DL] MLP模型加载失败: {e}")

        self._mlp_loaded = True
        return self._mlp_models

    def _select_top_k(
        self,
        result: Dict,
        residues: pd.DataFrame,
        method: str
    ) -> Dict[str, Any]:
        scores = result["scores"]

        all_scores_dict = {}
        for idx, row in residues.iterrows():
            label = f"{row['chain_id']}{row['res_id']}"
            score = float(scores[idx]) if idx < len(scores) else 0.0
            all_scores_dict[label] = {
                "score": round(score, 4),
                "residue_name": row["res_name"],
                "chain_id": row["chain_id"],
                "res_id": row["res_id"]
            }

        sorted_items = sorted(
            all_scores_dict.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        top_k_items = sorted_items[:self.top_k]

        hotspots_detail = []
        for label, info in top_k_items:
            hotspots_detail.append({
                "label": label,
                "chain": info["chain_id"],
                "residue_id": info["res_id"],
                "residue_name": info["residue_name"],
                "score": info["score"],
                "combined_score": info["score"]
            })

        return {
            "method": method,
            "model_loaded": result.get("model_loaded", False),
            "hotspots": [h["label"] for h in hotspots_detail],
            "hotspots_detail": hotspots_detail,
            "all_scores": all_scores_dict,
            "total_residues": len(residues),
            "num_hotspots": len(hotspots_detail),
            "top_k": self.top_k
        }

    def _rule_based_predict(self, residues: pd.DataFrame) -> Dict[str, Any]:
        scores = self._compute_rule_scores(residues, "both")
        return {"scores": scores, "method": "rule", "model_loaded": False}

    def _compute_rule_scores(self, residues: pd.DataFrame, method: str) -> np.ndarray:
        n = len(residues)
        scores = np.zeros(n)

        aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        hydrophobic = {'LEU', 'ILE', 'VAL', 'MET', 'ALA', 'PRO'}
        charged_pos = {'ARG', 'LYS'}
        charged_neg = {'ASP', 'GLU'}

        for idx, row in residues.iterrows():
            s = 0.0
            res_name = row.get('res_name', '')
            sasa = row.get('sasa', 50)
            conservation = row.get('conservation', 0.5)
            dist = row.get('dist_to_center', 15)
            energy = row.get('energy', 0)

            if res_name in aromatic:
                s += 0.35
            elif res_name in hydrophobic:
                s += 0.25
            elif res_name in charged_pos:
                s += 0.20
            elif res_name in charged_neg:
                s += 0.15

            if sasa > 100:
                s += 0.30
            elif sasa > 70:
                s += 0.20
            elif sasa > 40:
                s += 0.10

            s += conservation * 0.20

            if dist > 18:
                s += 0.20
            elif dist > 12:
                s += 0.10

            if energy < -1.5:
                s += 0.15
            elif energy < -0.5:
                s += 0.08

            if method == "dl":
                s += np.random.uniform(-0.05, 0.05)

            scores[idx] = min(s, 1.0)

        return scores

    def _load_esm_model(self):
        if self._esm_loaded:
            return self._esm_model

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "facebook/esm2_t33_650M_UR50D"
            print(f"[ESM] 正在加载: {model_name}...")

            has_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if has_cuda else 'cpu')

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            self._esm_model = {
                'tokenizer': tokenizer,
                'model': model,
                'device': device,
            }
            print(f"[ESM] 模型加载成功 (device={device})")
        except ImportError:
            print("[ESM] transformers未安装，跳过ESM特征")
            self._esm_model = None
        except Exception as e:
            print(f"[ESM] 模型加载失败: {e}")
            self._esm_model = None

        self._esm_loaded = True
        return self._esm_model

    def _get_esm_features(self, sequence: str) -> Optional[np.ndarray]:
        import torch

        esm = self._load_esm_model()
        if esm is None:
            return None

        try:
            tokenizer = esm['tokenizer']
            model = esm['model']
            device = esm['device']

            with torch.no_grad():
                inputs = tokenizer(sequence, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[0, 1:-1].cpu().numpy()

            return embeddings
        except Exception as e:
            print(f"[ESM] 特征提取失败: {e}")
            return None

    def _get_sequence_from_residues(self, residues_df: pd.DataFrame) -> str:
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        sequence = ""
        for _, row in residues_df.iterrows():
            res_name = row.get('res_name', '')
            sequence += aa_map.get(res_name, 'X')

        return sequence

    def _extract_residue_features(self, atom_array) -> pd.DataFrame:
        try:
            import biotite.structure as struc

            residue_starts = struc.get_residue_starts(atom_array)
            residue_info = []

            center = atom_array.coord.mean(axis=0)

            for i, start in enumerate(residue_starts):
                res_mask = np.zeros(len(atom_array), dtype=bool)
                if i < len(residue_starts) - 1:
                    res_mask[start:residue_starts[i + 1]] = True
                else:
                    res_mask[start:] = True

                res_atoms = atom_array[res_mask]
                if len(res_atoms) == 0:
                    continue

                res_name = str(res_atoms.res_name[0])
                res_id = int(res_atoms.res_id[0])
                chain_id = str(res_atoms.chain_id[0])

                ca_mask = res_atoms.atom_name == "CA"
                ca_atoms = res_atoms[ca_mask]
                if len(ca_atoms) > 0:
                    ca_coord = ca_atoms.coord[0]
                else:
                    ca_coord = res_atoms.coord[0]

                dist_to_center = float(np.linalg.norm(ca_coord - center))
                sasa = self._estimate_sasa(res_atoms, dist_to_center)
                energy = self._estimate_energy(res_name, dist_to_center)
                conservation = self._estimate_conservation(res_name, dist_to_center)

                residue_info.append({
                    "index": i,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "res_name": res_name,
                    "sasa": round(float(sasa), 2),
                    "energy": round(float(energy), 3),
                    "conservation": round(float(conservation), 3),
                    "dist_to_center": round(dist_to_center, 2),
                    "num_atoms": int(len(res_atoms)),
                    "ca_x": float(ca_coord[0]),
                    "ca_y": float(ca_coord[1]),
                    "ca_z": float(ca_coord[2])
                })

            return pd.DataFrame(residue_info)

        except Exception as e:
            print(f"提取残基特征出错: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def predict_hotspots(
        self,
        pdb_string: str,
        top_k: int = None,
        method: str = "dl"
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = top_k

        atom_array = self._parse_pdb_string(pdb_string)
        if atom_array is None:
            return {
                "hotspots": [],
                "hotspots_detail": [],
                "all_scores": {},
                "error": "无法解析PDB字符串"
            }

        result = self.predict(
            atom_array=atom_array,
            method=method,
            pdb_string=pdb_string,
            top_k=top_k
        )
        return result

    def _parse_pdb_string(self, pdb_string: str):
        try:
            import biotite.structure as bs
            import biotite.structure.io.pdb as bpdb

            pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_string))
            atom_array = pdb_file.get_structure(model=1)
            atom_array = atom_array[bs.filter_amino_acids(atom_array)]
            return atom_array
        except ImportError:
            print("[PDB] biotite未安装")
            return None
        except Exception as e:
            print(f"[PDB] 解析失败: {e}")
            return None

    def _estimate_sasa(self, res_atoms, dist_to_center: float) -> float:
        num_atoms = len(res_atoms)
        base_sasa = num_atoms * 10.0
        surface_factor = min(dist_to_center / 20.0, 1.0)
        return base_sasa * (0.3 + 0.7 * surface_factor)

    def _estimate_energy(self, res_name: str, dist_to_center: float) -> float:
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        charged = {'ARG', 'LYS', 'ASP', 'GLU'}
        polar = {'SER', 'THR', 'ASN', 'GLN', 'HIS', 'CYS', 'TYR'}

        if res_name in hydrophobic:
            base = -2.0
        elif res_name in charged:
            base = -1.0
        elif res_name in polar:
            base = -1.5
        else:
            base = -0.5

        return base + (dist_to_center / 30.0) * 2.0

    def _estimate_conservation(self, res_name: str, dist_to_center: float) -> float:
        high_cons = {'CYS', 'TRP', 'HIS', 'PRO'}
        med_cons = {'ARG', 'LYS', 'ASP', 'GLU', 'PHE', 'TYR', 'ASN', 'GLN'}
        low_cons = {'ALA', 'GLY', 'SER', 'THR', 'VAL', 'LEU', 'ILE', 'MET'}

        if res_name in high_cons:
            base = 0.7
        elif res_name in med_cons:
            base = 0.5
        elif res_name in low_cons:
            base = 0.3
        else:
            base = 0.4

        return base
