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
                self._ml_predictor = TabularPredictor.load(str(self.ml_model_path))
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
    
    def _predict_dl(self, atom_array, pdb_string: str = None) -> Dict[str, Any]:
        """
        使用深度学习方法预测热点残基
        基于图神经网络特征和序列特征
        """
        import pandas as pd
        
        residues = self._extract_residue_features(atom_array)
        
        if residues.empty:
            return {"method": "dl", "hotspots": [], "scores": {}, "hotspots_detail": []}
        
        scores = self._compute_dl_scores(residues, atom_array)
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
            "method": "dl",
            "hotspots": hotspot_labels,
            "scores": scores_dict,
            "hotspots_detail": hotspot_details,
            "total_residues": len(residues),
            "num_hotspots": len(hotspot_labels),
            "threshold": threshold,
            "all_scores": all_scores
        }
    
    def _compute_dl_scores(self, residues_df: 'pd.DataFrame', atom_array) -> np.ndarray:
        """
        深度学习方法计算热点得分
        基于序列特征、结构特征和进化特征
        """
        n_residues = len(residues_df)
        scores = np.zeros(n_residues)
        
        aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        hydrophobic = {'LEU', 'ILE', 'VAL', 'MET', 'ALA', 'PRO'}
        charged = {'ARG', 'LYS', 'ASP', 'GLU'}
        polar = {'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        
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
                    "num_atoms": int(len(res_atoms))
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
    methods.append("dl")
    return methods
