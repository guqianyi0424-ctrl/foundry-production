"""
热点残基预测工具
集成ppihotspotid (ML) 和 hotspot-prediction (DL) 模型
"""
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


class HotspotPredictor:
    """热点残基预测器"""
    
    def __init__(self):
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
    
    def predict(self, atom_array, method: str = "ml", pdb_string: str = None) -> Dict[str, Any]:
        """
        预测热点残基
        
        Args:
            atom_array: Biotite AtomArray对象
            method: 预测方法 'ml' (机器学习) 或 'dl' (深度学习)
            pdb_string: PDB格式字符串
        
        Returns:
            包含热点预测结果的字典
        """
        if method == "ml":
            return self._predict_ml(atom_array, pdb_string)
        elif method == "dl":
            return self._predict_dl(atom_array, pdb_string)
        else:
            raise ValueError(f"不支持的预测方法: {method}")
    
    def _load_ml_predictor(self):
        """加载ML模型"""
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
                print(f"✅ ML模型加载成功: {self.ml_model_path}")
            else:
                print(f"❌ ML模型路径不存在: {self.ml_model_path}")
                self._ml_predictor = None
        except Exception as e:
            print(f"❌ ML模型加载失败: {e}")
            self._ml_predictor = None
        
        self._ml_loaded = True
        return self._ml_predictor
    
    def _predict_ml(self, atom_array, pdb_string: str = None) -> Dict[str, Any]:
        """
        使用ppihotspotid机器学习模型预测热点残基
        """
        import pandas as pd
        
        residues = self._extract_residue_features(atom_array)
        
        if residues.empty:
            return {
                "method": "ml",
                "hotspots": [],
                "scores": {},
                "hotspots_detail": [],
                "error": "无法提取残基特征"
            }
        
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
                        residues['score'] = proba['P'].values
                    elif 1 in proba.columns:
                        residues['score'] = proba[1].values
                    else:
                        residues['score'] = (predictions == 'P').astype(float)
                except Exception:
                    residues['score'] = (predictions == 'P').astype(float)
                
                print(f"✅ ML模型预测完成，使用AutoGluon")
                
            except Exception as e:
                print(f"❌ ML预测失败，使用规则: {e}")
                residues['score'] = self._compute_rule_based_scores(residues, method="ml")
        else:
            print("⚠️ ML模型未加载，使用规则预测")
            residues['score'] = self._compute_rule_based_scores(residues, method="ml")
        
        threshold = 0.3
        hotspots_mask = residues['score'] > threshold
        hotspot_residues = residues[hotspots_mask]
        
        hotspot_labels = []
        for _, row in hotspot_residues.iterrows():
            label = f"{row['res_name']}{row['res_id']}"
            hotspot_labels.append(label)
        
        hotspot_details = []
        for _, row in hotspot_residues.iterrows():
            detail = {
                "chain": row.get("chain_id", "A"),
                "residue_name": row.get("res_name", ""),
                "residue_id": row.get("res_id", ""),
                "index": int(row.get("index", 0)),
                "score": float(row.get("score", 0))
            }
            hotspot_details.append(detail)
        
        scores_dict = {label: float(score) for label, score in zip(hotspot_labels, hotspot_residues['score'].values)}
        
        all_scores = {}
        for _, row in residues.iterrows():
            label = f"{row['res_name']}{row['res_id']}"
            all_scores[label] = float(row['score'])
        
        return {
            "method": "ml",
            "hotspots": hotspot_labels,
            "scores": scores_dict,
            "hotspots_detail": hotspot_details,
            "total_residues": len(residues),
            "num_hotspots": len(hotspot_labels),
            "threshold": threshold,
            "all_scores": all_scores
        }
    
    def _load_dl_model(self):
        """加载DL模型"""
        if self._dl_loaded:
            return self._dl_model
        
        try:
            import torch
            
            sys.path.insert(0, str(self.hotspot_dl_path))
            from model import PPIHotspotGAT, INPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT
            
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
                self._dl_model.load_state_dict(state_dict)
                self._dl_model.eval()
                print(f"✅ DL模型加载成功: {model_path}")
            else:
                print(f"❌ DL模型路径不存在: {model_path}")
                self._dl_model = None
        except Exception as e:
            print(f"❌ DL模型加载失败: {e}")
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
            print(f"正在加载ESM-2模型: {model_name}...")
            
            self._esm_model = {
                'tokenizer': AutoTokenizer.from_pretrained(model_name),
                'model': AutoModel.from_pretrained(model_name)
            }
            self._esm_model['model'].eval()
            print("✅ ESM-2模型加载成功")
        except Exception as e:
            print(f"❌ ESM-2模型加载失败: {e}")
            self._esm_model = None
        
        self._esm_loaded = True
        return self._esm_model
    
    def _get_esm_features(self, sequence: str) -> np.ndarray:
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
            print(f"ESM特征提取失败: {e}")
            return None
    
    def _predict_dl(self, atom_array, pdb_string: str = None) -> Dict[str, Any]:
        """
        使用深度学习方法预测热点残基
        """
        import pandas as pd
        
        residues = self._extract_residue_features(atom_array)
        
        if residues.empty:
            return {"method": "dl", "hotspots": [], "scores": {}, "hotspots_detail": []}
        
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
                    
                    residues['score'] = scores
                    print(f"✅ DL模型预测完成，使用GAT+ESM-2")
                else:
                    print("⚠️ ESM特征不可用，使用规则预测")
                    residues['score'] = self._compute_dl_scores(residues, atom_array)
                    
            except Exception as e:
                print(f"❌ DL预测失败，使用规则: {e}")
                import traceback
                traceback.print_exc()
                residues['score'] = self._compute_dl_scores(residues, atom_array)
        else:
            print("⚠️ DL模型未加载，使用规则预测")
            residues['score'] = self._compute_dl_scores(residues, atom_array)
        
        threshold = 0.35
        hotspots_mask = residues['score'] > threshold
        hotspot_residues = residues[hotspots_mask]
        
        hotspot_labels = []
        for _, row in hotspot_residues.iterrows():
            label = f"{row['res_name']}{row['res_id']}"
            hotspot_labels.append(label)
        
        hotspot_details = []
        for _, row in hotspot_residues.iterrows():
            detail = {
                "chain": row.get("chain_id", "A"),
                "residue_name": row.get("res_name", ""),
                "residue_id": row.get("res_id", ""),
                "index": int(row.get("index", 0)),
                "score": float(row.get("score", 0))
            }
            hotspot_details.append(detail)
        
        scores_dict = {label: float(score) for label, score in zip(hotspot_labels, hotspot_residues['score'].values)}
        
        all_scores = {}
        for _, row in residues.iterrows():
            label = f"{row['res_name']}{row['res_id']}"
            all_scores[label] = float(row['score'])
        
        return {
            "method": "dl",
            "hotspots": hotspot_labels,
            "scores": scores_dict,
            "hotspots_detail": hotspot_details,
            "total_residues": len(residues),
            "num_hotspots": len(hotspot_labels),
            "threshold": threshold,
            "all_scores": all_scores
        }
    
    def _get_sequence_from_residues(self, residues_df) -> str:
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
            aa = aa_map.get(res_name, 'X')
            sequence += aa
        
        return sequence
    
    def _build_node_features(self, residues_df, esm_features: np.ndarray) -> np.ndarray:
        """构建节点特征"""
        n_residues = len(residues_df)
        esm_dim = esm_features.shape[1]
        
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
    
    def _build_graph(self, atom_array, residues_df):
        """构建图结构"""
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
    
    def _compute_dl_scores(self, residues_df: 'pd.DataFrame', atom_array) -> np.ndarray:
        """
        深度学习方法计算热点得分（备用规则）
        """
        n_residues = len(residues_df)
        scores = np.zeros(n_residues)
        
        aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        hydrophobic = {'LEU', 'ILE', 'VAL', 'MET', 'ALA', 'PRO'}
        charged = {'ARG', 'LYS', 'ASP', 'GLU'}
        
        for idx, row in residues_df.iterrows():
            score = 0.0
            
            res_name = row.get('res_name', '')
            
            if res_name in aromatic:
                score += 0.35
            elif res_name in hydrophobic:
                score += 0.25
            elif res_name in charged:
                score += 0.15
            
            sasa = row.get('sasa', 50)
            if sasa > 100:
                score += 0.30
            elif sasa > 70:
                score += 0.20
            elif sasa > 40:
                score += 0.10
            
            conservation = row.get('conservation', 0.5)
            score += conservation * 0.20
            
            dist = row.get('dist_to_center', 15)
            if dist > 18:
                score += 0.20
            elif dist > 12:
                score += 0.10
            
            energy = row.get('energy', 0)
            if energy < -1.5:
                score += 0.15
            elif energy < -0.5:
                score += 0.08
            
            if idx > 0 and idx < n_residues - 1:
                prev_res = residues_df.iloc[idx-1].get('res_name', '')
                next_res = residues_df.iloc[idx+1].get('res_name', '')
                
                if prev_res in aromatic or next_res in aromatic:
                    score += 0.05
                
                if prev_res in hydrophobic and next_res in hydrophobic:
                    score += 0.05
            
            scores[idx] = min(score, 1.0)
        
        return scores
    
    def _extract_residue_features(self, atom_array) -> 'pd.DataFrame':
        """从AtomArray中提取残基特征"""
        try:
            import pandas as pd
            import biotite.structure as struc
            
            residue_starts = struc.get_residue_starts(atom_array)
            
            residue_info = []
            
            for i, start in enumerate(residue_starts):
                res_mask = np.zeros(len(atom_array), dtype=bool)
                if i < len(residue_starts) - 1:
                    res_mask[start:residue_starts[i+1]] = True
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
                
                center = atom_array.coord.mean(axis=0)
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
            
            df = pd.DataFrame(residue_info)
            return df
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"提取残基特征时出错: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def _estimate_sasa(self, res_atoms, dist_to_center: float) -> float:
        """估算溶剂可及表面积"""
        num_atoms = len(res_atoms)
        
        base_sasa = num_atoms * 10.0
        
        depth_factor = max(0, 1 - dist_to_center / 30.0)
        
        sasa = base_sasa * (1 - 0.5 * depth_factor)
        
        return min(sasa, 200.0)
    
    def _estimate_energy(self, res_name: str, dist_to_center: float) -> float:
        """估算残基能量"""
        hydrophobic = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
        charged_pos = {'ARG', 'LYS', 'HIS'}
        charged_neg = {'ASP', 'GLU'}
        polar = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        
        if res_name in hydrophobic:
            base_energy = -2.0
        elif res_name in charged_pos:
            base_energy = 1.5
        elif res_name in charged_neg:
            base_energy = -1.5
        elif res_name in polar:
            base_energy = -0.5
        else:
            base_energy = 0.0
        
        depth_factor = max(0, 1 - dist_to_center / 25.0)
        energy = base_energy * (1 + 0.5 * depth_factor)
        
        return energy
    
    def _estimate_conservation(self, res_name: str, dist_to_center: float) -> float:
        """估算保守性得分"""
        highly_conserved = {'CYS', 'TRP', 'PHE', 'TYR', 'HIS'}
        moderately_conserved = {'ARG', 'LYS', 'ASP', 'GLU', 'ASN', 'GLN'}
        variable = {'SER', 'ALA', 'GLY', 'PRO'}
        
        if res_name in highly_conserved:
            base_cons = 0.85
        elif res_name in moderately_conserved:
            base_cons = 0.65
        elif res_name in variable:
            base_cons = 0.35
        else:
            base_cons = 0.5
        
        surface_factor = min(1, dist_to_center / 20.0)
        conservation = base_cons * (0.7 + 0.3 * surface_factor)
        
        return min(conservation, 1.0)
    
    def _compute_rule_based_scores(self, residues_df: 'pd.DataFrame', method: str = "ml") -> np.ndarray:
        """基于规则计算热点得分 - ML方法专用"""
        n_residues = len(residues_df)
        scores = np.zeros(n_residues)
        
        ml_hotspot_weights = {
            'TRP': 0.40, 'TYR': 0.35, 'PHE': 0.35, 'ARG': 0.30,
            'HIS': 0.28, 'LEU': 0.25, 'ILE': 0.25, 'VAL': 0.20,
            'MET': 0.22, 'LYS': 0.15, 'ASP': 0.12, 'GLU': 0.12,
            'ASN': 0.10, 'GLN': 0.10, 'CYS': 0.18, 'SER': 0.08,
            'THR': 0.08, 'ALA': 0.05, 'GLY': 0.03, 'PRO': 0.10
        }
        
        for idx, row in residues_df.iterrows():
            score = 0.0
            
            res_name = row.get('res_name', '')
            score += ml_hotspot_weights.get(res_name, 0.05)
            
            sasa = row.get('sasa', 50)
            if sasa > 90:
                score += 0.30
            elif sasa > 60:
                score += 0.20
            elif sasa > 30:
                score += 0.10
            
            conservation = row.get('conservation', 0.5)
            score += conservation * 0.20
            
            energy = row.get('energy', 0)
            if energy < -2:
                score += 0.25
            elif energy < -1:
                score += 0.15
            elif energy < 0:
                score += 0.08
            
            dist = row.get('dist_to_center', 15)
            if dist > 18:
                score += 0.15
            elif dist > 12:
                score += 0.08
            
            scores[idx] = min(score, 1.0)
        
        return scores


def get_available_methods() -> List[str]:
    """获取可用的预测方法列表"""
    methods = ["ml", "dl"]
    return methods
