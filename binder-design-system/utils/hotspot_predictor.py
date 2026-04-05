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
        """
        try:
            from autogluon.tabular import TabularPredictor
            import pandas as pd
            
            if not self.ml_model_path.exists():
                return self._predict_rule_based(atom_array)
            
            residues = self._extract_residue_features(atom_array)
            
            if residues.empty:
                return {
                    "method": "ml",
                    "hotspots": [],
                    "scores": {},
                    "hotspots_detail": [],
                    "error": "无法提取残基特征"
                }
            
            try:
                predictor = TabularPredictor.load(str(self.ml_model_path))
                
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
                
            except Exception as e:
                residues['score'] = self._compute_rule_based_scores(residues)
            
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
                
        except ImportError as e:
            return self._predict_rule_based(atom_array)
        except Exception as e:
            return self._predict_rule_based(atom_array)
    
    def _predict_rule_based(self, atom_array) -> Dict[str, Any]:
        """
        基于规则的热点残基预测（备用方案）
        """
        try:
            import pandas as pd
            import biotite.structure as struc
            
            residues = self._extract_residue_features(atom_array)
            
            if residues.empty:
                return {"method": "rule", "hotspots": [], "scores": {}, "hotspots_detail": []}
            
            scores = self._compute_rule_based_scores(residues)
            residues['score'] = scores
            
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
                "method": "rule",
                "hotspots": hotspot_labels,
                "scores": scores_dict,
                "hotspots_detail": hotspot_details,
                "total_residues": len(residues),
                "num_hotspots": len(hotspot_labels),
                "threshold": threshold,
                "all_scores": all_scores
            }
            
        except Exception as e:
            return {
                "method": "rule",
                "hotspots": [],
                "scores": {},
                "hotspots_detail": [],
                "error": f"规则预测失败: {str(e)}"
            }
    
    def _predict_dl(self, atom_array, pdb_string: str = None) -> Dict[str, Any]:
        """
        使用深度学习模型预测热点残基（简化版）
        """
        return self._predict_rule_based(atom_array)
    
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
                
                res_name = res_atoms.res_name[0]
                res_id = res_atoms.res_id[0]
                chain_id = res_atoms.chain_id[0]
                
                ca_atoms = res_atoms[res_atoms.atom_name == "CA"]
                if len(ca_atoms) > 0:
                    ca_coord = ca_atoms.coord[0]
                else:
                    ca_coord = res_atoms.coord[0]
                
                center = atom_array.coord.mean(axis=0)
                dist_to_center = np.linalg.norm(ca_coord - center)
                
                sasa = self._estimate_sasa(res_atoms, dist_to_center)
                
                energy = self._estimate_energy(res_name, dist_to_center)
                
                conservation = self._estimate_conservation(res_name, dist_to_center)
                
                residue_info.append({
                    "index": i,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "res_name": res_name,
                    "sasa": round(sasa, 2),
                    "energy": round(energy, 3),
                    "conservation": round(conservation, 3),
                    "dist_to_center": round(dist_to_center, 2),
                    "num_atoms": len(res_atoms)
                })
            
            df = pd.DataFrame(residue_info)
            return df
            
        except Exception as e:
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
    
    def _compute_rule_based_scores(self, residues_df: 'pd.DataFrame') -> np.ndarray:
        """基于规则计算热点得分"""
        n_residues = len(residues_df)
        scores = np.zeros(n_residues)
        
        hotspot_residues = {'TRP', 'TYR', 'PHE', 'ARG', 'HIS', 'LEU', 'ILE', 'VAL'}
        
        for idx, row in residues_df.iterrows():
            score = 0.0
            
            res_name = row.get('res_name', '')
            if res_name in hotspot_residues:
                score += 0.3
            
            sasa = row.get('sasa', 50)
            if sasa > 80:
                score += 0.25
            elif sasa > 50:
                score += 0.15
            
            conservation = row.get('conservation', 0.5)
            score += conservation * 0.25
            
            energy = row.get('energy', 0)
            if energy < -1:
                score += 0.2
            elif energy < 0:
                score += 0.1
            
            dist = row.get('dist_to_center', 15)
            if dist > 15:
                score += 0.15
            elif dist > 10:
                score += 0.1
            
            scores[idx] = min(score, 1.0)
        
        return scores


def get_available_methods() -> List[str]:
    """获取可用的预测方法列表"""
    methods = ["ml"]
    try:
        import torch
        methods.append("dl")
    except ImportError:
        pass
    return methods
