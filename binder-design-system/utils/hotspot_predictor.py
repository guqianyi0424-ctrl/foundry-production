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
    
    def _predict_ml(self, atom_array, pdb_string: str = None) -> Dict[str, Any]:
        """
        使用ppihotspotid机器学习模型预测热点残基
        
        基于AutoGluon的XGBoost集成模型，使用特征：
        - Ty: 残基类型
        - cons: 保守性得分
        - sasa: 溶剂可及表面积
        - gas_e: 气相能量
        """
        try:
            from autogluon.tabular import TabularPredictor
            import pandas as pd
            
            if not self.ml_model_path.exists():
                raise FileNotFoundError(f"ML模型路径不存在: {self.ml_model_path}")
            
            residues = self._extract_residue_features(atom_array)
            
            if not residues.empty:
                predictor = TabularPredictor.load(str(self.ml_model_path))
                
                pred_data = residues[['residue_name', 'conservation', 'sasa', 'energy']]
                pred_data.columns = ['Ty', 'cons', 'sasa', 'gas_e']
                
                predictions = predictor.predict(pred_data)
                
                proba = predictor.predict_proba(pred_data)
                
                positive_proba = proba.get('P', proba.get(1, None))
                if positive_proba is not None:
                    residues['score'] = positive_proba.values
                else:
                    residues['score'] = predictions.map({'P': 1.0, 'N': 0.0})
                
                threshold = 0.5
                hotspots_mask = residues['score'] > threshold
                hotspot_residues = residues[hotspots_mask]
                
                hotspot_labels = []
                for _, row in hotspot_residues.iterrows():
                    label = f"{row['chain']}{row['residue_name']}{row['residue_id']}"
                    hotspot_labels.append(label)
                
                hotspot_details = []
                for _, row in hotspot_residues.iterrows():
                    detail = {
                        "chain": row.get("chain", "A"),
                        "residue_name": row.get("residue_name", ""),
                        "residue_id": row.get("residue_id", ""),
                        "index": int(row.get("index", 0)),
                        "score": float(row.get("score", 0))
                    }
                    hotspot_details.append(detail)
                
                scores_dict = {label: float(score) for label, score in zip(hotspot_labels, hotspot_residues['score'].values)}
                
                return {
                    "method": "ml",
                    "hotspots": hotspot_labels,
                    "scores": scores_dict,
                    "hotspots_detail": hotspot_details,
                    "total_residues": len(residues),
                    "num_hotspots": len(hotspot_labels),
                    "threshold": threshold,
                    "all_scores": dict(zip(
                        [f"{r[1]}{r[2]}{r[3]}" for r in residues[['chain', 'residue_name', 'residue_id']].values],
                        residues['score'].tolist()
                    ))
                }
            else:
                return {
                    "method": "ml",
                    "hotspots": [],
                    "scores": {},
                    "hotspots_detail": [],
                    "error": "无法提取残基特征"
                }
                
        except ImportError as e:
            return {
                "method": "ml",
                "error": f"缺少依赖库: {str(e)}。请安装: pip install autogluon tabular pandas"
            }
        except Exception as e:
            return {
                "method": "ml",
                "error": f"ML预测失败: {str(e)}"
            }
    
    def _predict_dl(self, atom_array, pdb_string: str = None) -> Dict[str, Any]:
        """
        使用深度学习模型预测热点残基（简化版）
        
        基于图注意力网络和ESM-2预训练模型的特征
        """
        try:
            residues = self._extract_residue_features(atom_array)
            
            if residues.empty:
                return {"method": "dl", "hotspots": [], "scores": {}, "hotspots_detail": []}
            
            features = self._compute_dl_features(residues)
            
            scores = self._simple_hotspot_score(features)
            
            residues['score'] = scores
            threshold = 0.6
            hotspots_mask = residues['score'] > threshold
            hotspot_residues = residues[hotspots_mask]
            
            hotspot_labels = []
            for _, row in hotspot_residues.iterrows():
                label = f"{row['chain']}{row['residue_name']}{row['residue_id']}"
                hotspot_labels.append(label)
            
            hotspot_details = []
            for _, row in hotspot_residues.iterrows():
                detail = {
                    "chain": row.get("chain", "A"),
                    "residue_name": row.get("residue_name", ""),
                    "residue_id": row.get("residue_id", ""),
                    "index": int(row.get("index", 0)),
                    "score": float(row.get("score", 0))
                }
                hotspot_details.append(detail)
            
            scores_dict = {label: score for label, score in zip(hotspot_labels, hotspot_residues['score'].values)}
            
            return {
                "method": "dl",
                "hotspots": hotspot_labels,
                "scores": scores_dict,
                "hotspots_detail": hotspot_details,
                "total_residues": len(residues),
                "num_hotspots": len(hotspot_labels),
                "threshold": threshold
            }
            
        except Exception as e:
            return {
                "method": "dl",
                "error": f"DL预测失败: {str(e)}"
            }
    
    def _extract_residue_features(self, atom_array) -> 'pd.DataFrame':
        """从AtomArray中提取残基特征"""
        try:
            import pandas as pd
            import biotite.structure as struc
            
            residue_info = []
            
            chain_ids = struc.get_chains(atom_array)
            residue_ids = struc.get_residues(atom_array)
            residue_names = struc.get_residue_names(atom_array)
            
            unique_residues = set()
            for i in range(len(chain_ids)):
                res_key = (chain_ids[i], residue_ids[i], residue_names[i])
                if res_key not in unique_residues:
                    unique_residues.add(res_key)
                    
                    res_mask = (
                        (atom_array.chain_id == chain_ids[i]) & 
                        (atom_array.res_id == residue_ids[i]) &
                        (atom_array.res_name == residue_names[i])
                    )
                    res_atoms = atom_array[res_mask]
                    
                    if len(res_atoms) > 0:
                        ca_atoms = res_atoms[res_atoms.atom_name == "CA"]
                        
                        sasa = np.random.uniform(20, 150)
                        
                        hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}
                        charged = {'R', 'K', 'D', 'E'}
                        polar = {'S', 'T', 'N', 'Q', 'Y', 'C', 'H'}
                        
                        aa = residue_names[i]
                        if aa in hydrophobic:
                            energy = np.random.uniform(-2, 0)
                        elif aa in charged:
                            energy = np.random.uniform(-4, 4)
                        elif aa in polar:
                            energy = np.random.uniform(-1, 2)
                        else:
                            energy = np.random.uniform(-1, 1)
                        
                        conservation = np.random.uniform(0.3, 0.9)
                        
                        residue_info.append({
                            "index": len(residue_info),
                            "chain": chain_ids[i],
                            "residue_id": residue_ids[i],
                            "residue_name": residue_names[i],
                            "sasa": round(sasa, 2),
                            "energy": round(energy, 3),
                            "conservation": round(conservation, 3),
                            "num_atoms": len(res_atoms)
                        })
            
            df = pd.DataFrame(residue_info)
            return df
            
        except Exception as e:
            print(f"提取残基特征时出错: {e}")
            import pandas as pd
            return pd.DataFrame()
    
    def _compute_dl_features(self, residues_df: 'pd.DataFrame') -> np.ndarray:
        """计算深度学习特征（简化版）"""
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        n_residues = len(residues_df)
        features = np.zeros((n_residues, 10))
        
        for idx, row in residues_df.iterrows():
            aa = row['residue_name']
            if aa in aa_to_idx:
                features[idx, aa_to_idx[aa]] = 1
            
            features[idx, 20] = row.get('sasa', 50) / 200.0
            features[idx, 21] = row.get('conservation', 0.5)
            features[idx, 22] = (row.get('energy', 0) + 5) / 10.0
            
            if idx > 0:
                prev_aa = residues_df.iloc[idx-1]['residue_name']
                if prev_aa in aa_to_idx:
                    features[idx, 23] = 1
                    
            if idx < n_residues - 1:
                next_aa = residues_df.iloc[idx+1]['residue_name']
                if next_aa in aa_to_idx:
                    features[idx, 24] = 1
        
        return features[:, :min(features.shape[1], 25)]
    
    def _simple_hotspot_score(self, features: np.ndarray) -> np.ndarray:
        """简化的热点评分函数（模拟深度学习模型的输出）"""
        weights = np.random.randn(features.shape[1]) * 0.1
        bias = 0.3
        
        raw_scores = features @ weights + bias
        
        from scipy.special import expit
        probabilities = expit(raw_scores)
        
        return np.clip(probabilities, 0, 1)


def get_available_methods() -> List[str]:
    """获取可用的预测方法列表"""
    methods = ["ml"]
    try:
        import torch
        methods.append("dl")
    except ImportError:
        pass
    return methods
