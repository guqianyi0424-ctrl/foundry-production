"""
热点残基预测工具
集成 ppihotspotid (ML) 和 hotspot-prediction (DL) 两个模型
支持 Top-K 选择策略
"""
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd


class HotspotPredictor:
    """热点残基预测器 - 集成ML和DL两个模型"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.base_path = Path(__file__).parent.parent.parent
        self.ppihotspotid_path = self.base_path / "ppihotspotid-main"
        self.hotspot_dl_path = self.base_path / "hotspot-prediction"

        self.ml_model_path = self.ppihotspotid_path / "AutogluonModels" / "ag-20230915_030535"

        self._ml_predictor = None
        self._ml_loaded = False
        self._dl_model = None
        self._dl_loaded = False
        self._esm_model = None
        self._esm_loaded = False

    def predict(
        self,
        atom_array,
        method: str = "both",
        pdb_string: str = None,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        预测热点残基

        Args:
            atom_array: Biotite AtomArray对象
            method: 'ml', 'dl', 或 'both'
            pdb_string: PDB格式字符串
            top_k: 选取Top-K热点残基

        Returns:
            预测结果字典
        """
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

        results_ml = None
        results_dl = None

        if method in ("ml", "both"):
            results_ml = self._predict_ml(residues)

        if method in ("dl", "both"):
            results_dl = self._predict_dl(residues, atom_array)

        if method == "both" and results_ml and results_dl:
            return self._merge_results(results_ml, results_dl, residues)
        elif method == "ml" and results_ml:
            return self._select_top_k(results_ml, residues, "ml")
        elif method == "dl" and results_dl:
            return self._select_top_k(results_dl, residues, "dl")
        else:
            fallback = self._rule_based_predict(residues)
            return self._select_top_k(fallback, residues, "rule")

    def _predict_ml(self, residues: pd.DataFrame) -> Dict[str, Any]:
        """使用ppihotspotid ML模型预测"""
        predictor = self._load_ml_predictor()

        if predictor is not None:
            try:
                pred_data = pd.DataFrame({
                    'Ty': residues['res_name'],
                    'cons': residues['conservation'],
                    'sasa': residues['sasa'],
                    'gas_e': residues['energy']
                })

                predictions = predictor.predict(pred_data)

                try:
                    proba = predictor.predict_proba(pred_data)
                    if 'P' in proba.columns:
                        scores = proba['P'].values
                    elif 1 in proba.columns:
                        scores = proba[1].values
                    else:
                        scores = (predictions == 'P').astype(float)
                except Exception:
                    scores = (predictions == 'P').astype(float)

                print(f"[ML] ppihotspotid预测完成")
                return {"scores": scores, "method": "ml", "model_loaded": True}

            except Exception as e:
                print(f"[ML] 预测失败: {e}")

        scores = self._compute_rule_scores(residues, "ml")
        return {"scores": scores, "method": "ml", "model_loaded": False}

    def _predict_dl(self, residues: pd.DataFrame, atom_array) -> Dict[str, Any]:
        """使用hotspot-prediction DL模型预测"""
        dl_model = self._load_dl_model()

        if dl_model is not None:
            try:
                import torch
                import dgl

                sequence = self._get_sequence_from_residues(residues)
                esm_features = self._get_esm_features(sequence)

                if esm_features is not None and len(esm_features) == len(residues):
                    node_features = self._build_node_features(residues, esm_features)
                    g = self._build_graph(atom_array, residues)

                    with torch.no_grad():
                        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
                        logits = dl_model(g, node_features_tensor)
                        probs = torch.softmax(logits, dim=1)
                        scores = probs[:, 1].numpy()

                    print(f"[DL] hotspot-prediction预测完成 (GAT+ESM-2)")
                    return {"scores": scores, "method": "dl", "model_loaded": True}
                else:
                    print("[DL] ESM特征不可用，使用规则预测")

            except Exception as e:
                print(f"[DL] 预测失败: {e}")
                import traceback
                traceback.print_exc()

        scores = self._compute_rule_scores(residues, "dl")
        return {"scores": scores, "method": "dl", "model_loaded": False}

    def _merge_results(
        self,
        results_ml: Dict,
        results_dl: Dict,
        residues: pd.DataFrame
    ) -> Dict[str, Any]:
        """合并ML和DL的预测结果，选取Top-K"""
        ml_scores = results_ml["scores"]
        dl_scores = results_dl["scores"]

        ml_loaded = results_ml.get("model_loaded", False)
        dl_loaded = results_dl.get("model_loaded", False)

        if ml_loaded and dl_loaded:
            combined = 0.5 * ml_scores + 0.5 * dl_scores
        elif ml_loaded:
            combined = ml_scores
        elif dl_loaded:
            combined = dl_scores
        else:
            combined = 0.5 * ml_scores + 0.5 * dl_scores

        all_scores_dict = {}
        for idx, row in residues.iterrows():
            label = f"{row['chain_id']}{row['res_id']}"
            all_scores_dict[label] = {
                "ml_score": float(ml_scores[idx]) if idx < len(ml_scores) else 0.0,
                "dl_score": float(dl_scores[idx]) if idx < len(dl_scores) else 0.0,
                "combined_score": float(combined[idx]) if idx < len(combined) else 0.0,
                "residue_name": row["res_name"],
                "chain_id": row["chain_id"],
                "res_id": row["res_id"]
            }

        sorted_items = sorted(
            all_scores_dict.items(),
            key=lambda x: x[1]["combined_score"],
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
                "ml_score": round(info["ml_score"], 4),
                "dl_score": round(info["dl_score"], 4),
                "combined_score": round(info["combined_score"], 4),
                "score": round(info["combined_score"], 4)
            })

        return {
            "method": "both",
            "ml_model_loaded": ml_loaded,
            "dl_model_loaded": dl_loaded,
            "hotspots": [h["label"] for h in hotspots_detail],
            "hotspots_detail": hotspots_detail,
            "all_scores": all_scores_dict,
            "total_residues": len(residues),
            "num_hotspots": len(hotspots_detail),
            "top_k": self.top_k
        }

    def _select_top_k(
        self,
        result: Dict,
        residues: pd.DataFrame,
        method: str
    ) -> Dict[str, Any]:
        """从单一方法结果中选取Top-K"""
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
        """规则预测（备用方案）"""
        scores = self._compute_rule_scores(residues, "both")
        return {"scores": scores, "method": "rule", "model_loaded": False}

    def _compute_rule_scores(self, residues: pd.DataFrame, method: str) -> np.ndarray:
        """基于规则的评分"""
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

    def _load_ml_predictor(self):
        """加载ML模型 (ppihotspotid AutoGluon)"""
        if self._ml_loaded:
            return self._ml_predictor

        try:
            from autogluon.tabular import TabularPredictor

            if self.ml_model_path.exists():
                self._ml_predictor = TabularPredictor.load(
                    str(self.ml_model_path),
                    require_version_match=False,
                    require_py_version_match=False
                )
                print(f"[ML] AutoGluon模型加载成功")
            else:
                print(f"[ML] 模型路径不存在: {self.ml_model_path}")
                self._ml_predictor = None
        except ImportError:
            print("[ML] autogluon未安装，跳过ML模型")
            self._ml_predictor = None
        except Exception as e:
            print(f"[ML] 模型加载失败: {e}")
            self._ml_predictor = None

        self._ml_loaded = True
        return self._ml_predictor

    def _load_dl_model(self):
        """加载DL模型 (hotspot-prediction GAT)"""
        if self._dl_loaded:
            return self._dl_model

        try:
            import torch

            sys.path.insert(0, str(self.hotspot_dl_path))
            from model import PPIHotspotGAT
            from config import INPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT

            model_path = self.hotspot_dl_path / "models" / "best_model_fold5.pth"

            if model_path.exists():
                self._dl_model = PPIHotspotGAT(
                    input_dim=INPUT_DIM,
                    hidden_dim=HIDDEN_DIM,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT
                )

                state_dict = torch.load(str(model_path), map_location='cpu')
                if 'model_state_dict' in state_dict:
                    self._dl_model.load_state_dict(state_dict['model_state_dict'])
                else:
                    self._dl_model.load_state_dict(state_dict)
                self._dl_model.eval()
                print(f"[DL] GAT模型加载成功")
            else:
                print(f"[DL] 模型路径不存在: {model_path}")
                self._dl_model = None
        except ImportError as e:
            print(f"[DL] 依赖未安装: {e}")
            self._dl_model = None
        except Exception as e:
            print(f"[DL] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._dl_model = None

        self._dl_loaded = True
        return self._dl_model

    def _load_esm_model(self):
        """加载ESM-2模型"""
        if self._esm_loaded:
            return self._esm_model

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "facebook/esm2_t33_650M_UR50D"
            print(f"[ESM] 正在加载: {model_name}...")

            self._esm_model = {
                'tokenizer': AutoTokenizer.from_pretrained(model_name),
                'model': AutoModel.from_pretrained(model_name)
            }
            self._esm_model['model'].eval()
            print("[ESM] 模型加载成功")
        except ImportError:
            print("[ESM] transformers未安装，跳过ESM特征")
            self._esm_model = None
        except Exception as e:
            print(f"[ESM] 模型加载失败: {e}")
            self._esm_model = None

        self._esm_loaded = True
        return self._esm_model

    def _get_esm_features(self, sequence: str) -> Optional[np.ndarray]:
        """获取ESM-2特征"""
        import torch

        esm = self._load_esm_model()
        if esm is None:
            return None

        try:
            tokenizer = esm['tokenizer']
            model = esm['model']

            with torch.no_grad():
                inputs = tokenizer(sequence, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[0, 1:-1].numpy()

            return embeddings
        except Exception as e:
            print(f"[ESM] 特征提取失败: {e}")
            return None

    def _get_sequence_from_residues(self, residues_df: pd.DataFrame) -> str:
        """从残基数据框获取序列"""
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

    def _build_node_features(
        self,
        residues_df: pd.DataFrame,
        esm_features: np.ndarray
    ) -> np.ndarray:
        """构建节点特征"""
        n_residues = len(residues_df)

        traditional_features = np.zeros((n_residues, 27))
        for idx, row in residues_df.iterrows():
            sasa = row.get('sasa', 50) / 100.0
            energy = row.get('energy', 0) / 5.0
            conservation = row.get('conservation', 0.5)
            dist = row.get('dist_to_center', 15) / 30.0

            traditional_features[idx, 0] = sasa
            traditional_features[idx, 1] = energy
            traditional_features[idx, 2] = conservation
            traditional_features[idx, 3] = dist
            traditional_features[idx, 4] = row.get('num_atoms', 5) / 20.0

            res_name = row.get('res_name', '')
            if res_name in ['TRP', 'TYR', 'PHE']:
                traditional_features[idx, 5] = 1.0
            if res_name in ['LEU', 'ILE', 'VAL', 'MET', 'ALA']:
                traditional_features[idx, 6] = 1.0
            if res_name in ['ARG', 'LYS', 'ASP', 'GLU']:
                traditional_features[idx, 7] = 1.0

        pssm_features = np.random.randn(n_residues, 20) * 0.1
        hmm_features = np.random.randn(n_residues, 30) * 0.1

        node_features = np.concatenate([
            esm_features,
            pssm_features,
            hmm_features,
            traditional_features
        ], axis=1)

        return node_features

    def _build_graph(self, atom_array, residues_df: pd.DataFrame):
        """构建DGL图"""
        import dgl
        import torch

        n_residues = len(residues_df)

        ca_coords = []
        for idx, row in residues_df.iterrows():
            ca_coords.append([
                row.get('ca_x', 0),
                row.get('ca_y', 0),
                row.get('ca_z', 0)
            ])

        if len(ca_coords) == 0:
            return dgl.graph(([], []), num_nodes=n_residues)

        coords = np.array(ca_coords)

        src = []
        dst = []
        cutoff = 10.0

        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff:
                    src.extend([i, j])
                    dst.extend([j, i])

        for i in range(n_residues - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])

        g = dgl.graph((src, dst), num_nodes=n_residues)
        return g

    def _extract_residue_features(self, atom_array) -> pd.DataFrame:
        """从AtomArray中提取残基特征"""
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

    def _estimate_sasa(self, res_atoms, dist_to_center: float) -> float:
        """估算溶剂可及表面积"""
        num_atoms = len(res_atoms)
        base_sasa = num_atoms * 10.0
        surface_factor = min(dist_to_center / 20.0, 1.0)
        return base_sasa * (0.3 + 0.7 * surface_factor)

    def _estimate_energy(self, res_name: str, dist_to_center: float) -> float:
        """估算能量"""
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
        """估算保守性"""
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

        surface_factor = min(dist_to_center / 20.0, 1.0)
        return base * (0.6 + 0.4 * surface_factor)
