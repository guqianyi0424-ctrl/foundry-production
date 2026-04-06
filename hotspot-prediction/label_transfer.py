"""
标签转移策略模块
功能：
1. 使用BLAST计算序列相似度
2. 使用TM-align计算结构相似度
3. 聚类相似的蛋白链
4. 转移热点残基标注

参考：DeepHotResi论文 2.7节
"""
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LabelTransfer:
    """标签转移策略实现"""
    
    def __init__(self, 
                 seq_similarity_threshold: float = 0.4,
                 tm_score_threshold: float = 0.5,
                 pdb_dir: str = None,
                 blast_db: str = None):
        """
        初始化标签转移器
        
        Args:
            seq_similarity_threshold: 序列相似度阈值（默认0.4）
            tm_score_threshold: TM-score阈值（默认0.5）
            pdb_dir: PDB文件目录
            blast_db: BLAST数据库路径
        """
        self.seq_similarity_threshold = seq_similarity_threshold
        self.tm_score_threshold = tm_score_threshold
        self.pdb_dir = pdb_dir or "./pdb_files"
        self.blast_db = blast_db
        
        self.similarity_matrix = {}
        self.clusters = []
        
    def run_blast(self, query_fasta: str, subject_fasta: str) -> Optional[float]:
        """
        运行BLAST计算序列相似度
        
        Args:
            query_fasta: 查询序列FASTA文件
            subject_fasta: 目标序列FASTA文件
            
        Returns:
            序列相似度（0-1之间）
        """
        try:
            cmd = [
                'blastp',
                '-query', query_fasta,
                '-subject', subject_fasta,
                '-outfmt', '6 pident length qlen slen',
                '-max_target_seqs', '1'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0 or not result.stdout.strip():
                return None
                
            parts = result.stdout.strip().split('\t')
            if len(parts) >= 4:
                pident = float(parts[0])
                length = float(parts[1])
                qlen = float(parts[2])
                slen = float(parts[3])
                
                coverage = length / max(qlen, slen)
                similarity = (pident / 100) * coverage
                
                return similarity
                
        except Exception as e:
            print(f"BLAST运行失败: {e}")
            
        return None
    
    def run_tmalign(self, pdb1: str, pdb2: str) -> Optional[float]:
        """
        运行TM-align计算结构相似度
        
        Args:
            pdb1: 第一个PDB文件路径
            pdb2: 第二个PDB文件路径
            
        Returns:
            TM-score（0-1之间）
        """
        try:
            cmd = ['TMalign', pdb1, pdb2]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return None
                
            for line in result.stdout.split('\n'):
                if 'TM-score=' in line:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        tm_score = float(parts[1].split()[0])
                        return tm_score
                        
        except Exception as e:
            print(f"TM-align运行失败: {e}")
            
        return None
    
    def calculate_sequence_similarity_simple(self, seq1: str, seq2: str) -> float:
        """
        简单序列相似度计算（无需BLAST）
        使用序列比对和相同残基比例
        
        Args:
            seq1: 序列1
            seq2: 序列2
            
        Returns:
            序列相似度
        """
        if not seq1 or not seq2:
            return 0.0
            
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        if max_len == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        
        similarity = matches / max_len
        
        return similarity
    
    def calculate_structure_similarity_simple(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        简单结构相似度计算（无需TM-align）
        使用RMSD估算
        
        Args:
            coords1: 坐标数组1 (N, 3)
            coords2: 坐标数组2 (M, 3)
            
        Returns:
            结构相似度分数
        """
        if coords1 is None or coords2 is None:
            return 0.0
        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0
            
        min_len = min(len(coords1), len(coords2))
        
        coords1_aligned = coords1[:min_len]
        coords2_aligned = coords2[:min_len]
        
        centroid1 = coords1_aligned.mean(axis=0)
        centroid2 = coords2_aligned.mean(axis=0)
        
        coords1_centered = coords1_aligned - centroid1
        coords2_centered = coords2_aligned - centroid2
        
        diff = coords1_centered - coords2_centered
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        
        tm_score_estimate = 1.0 / (1.0 + (rmsd / 5.0) ** 2)
        
        return tm_score_estimate
    
    def build_similarity_matrix(self, 
                                protein_data: Dict[str, Dict],
                                use_external_tools: bool = False) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        构建蛋白链之间的相似度矩阵
        
        Args:
            protein_data: 蛋白质数据字典
                {
                    'chain_id': {
                        'sequence': str,
                        'coords': np.ndarray,
                        'hotspots': List[int]
                    }
                }
            use_external_tools: 是否使用外部工具（BLAST, TM-align）
            
        Returns:
            相似度矩阵 {(chain1, chain2): (seq_sim, struct_sim)}
        """
        chain_ids = list(protein_data.keys())
        n = len(chain_ids)
        
        print(f"构建相似度矩阵: {n} 个蛋白链")
        
        similarity_matrix = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                chain1 = chain_ids[i]
                chain2 = chain_ids[j]
                
                seq1 = protein_data[chain1].get('sequence', '')
                seq2 = protein_data[chain2].get('sequence', '')
                
                coords1 = protein_data[chain1].get('coords')
                coords2 = protein_data[chain2].get('coords')
                
                if use_external_tools:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f1:
                        f1.write(f">chain1\n{seq1}")
                        fasta1 = f1.name
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f2:
                        f2.write(f">chain2\n{seq2}")
                        fasta2 = f2.name
                    
                    seq_sim = self.run_blast(fasta1, fasta2)
                    
                    os.unlink(fasta1)
                    os.unlink(fasta2)
                    
                    if seq_sim is None:
                        seq_sim = self.calculate_sequence_similarity_simple(seq1, seq2)
                else:
                    seq_sim = self.calculate_sequence_similarity_simple(seq1, seq2)
                
                if use_external_tools:
                    pdb1 = protein_data[chain1].get('pdb_file')
                    pdb2 = protein_data[chain2].get('pdb_file')
                    
                    if pdb1 and pdb2 and os.path.exists(pdb1) and os.path.exists(pdb2):
                        struct_sim = self.run_tmalign(pdb1, pdb2)
                    else:
                        struct_sim = self.calculate_structure_similarity_simple(coords1, coords2)
                else:
                    struct_sim = self.calculate_structure_similarity_simple(coords1, coords2)
                
                similarity_matrix[(chain1, chain2)] = (seq_sim, struct_sim)
                similarity_matrix[(chain2, chain1)] = (seq_sim, struct_sim)
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def cluster_proteins(self, 
                         protein_data: Dict[str, Dict],
                         similarity_matrix: Dict[Tuple[str, str], Tuple[float, float]] = None) -> List[List[str]]:
        """
        聚类相似的蛋白链
        
        Args:
            protein_data: 蛋白质数据字典
            similarity_matrix: 相似度矩阵
            
        Returns:
            聚类结果 [[chain1, chain2, ...], ...]
        """
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
            
        chain_ids = list(protein_data.keys())
        n = len(chain_ids)
        
        visited = set()
        clusters = []
        
        max_seq_sim = 0
        max_struct_sim = 0
        high_sim_pairs = []
        
        for chain in chain_ids:
            if chain in visited:
                continue
                
            cluster = [chain]
            visited.add(chain)
            
            for other in chain_ids:
                if other in visited:
                    continue
                    
                key = (chain, other)
                if key in similarity_matrix:
                    seq_sim, struct_sim = similarity_matrix[key]
                    
                    max_seq_sim = max(max_seq_sim, seq_sim)
                    max_struct_sim = max(max_struct_sim, struct_sim)
                    
                    if seq_sim >= self.seq_similarity_threshold and struct_sim >= self.tm_score_threshold:
                        cluster.append(other)
                        visited.add(other)
                    
                    if seq_sim >= 0.2 or struct_sim >= 0.2:
                        high_sim_pairs.append((chain, other, seq_sim, struct_sim))
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        print(f"相似度统计: 最大序列相似度={max_seq_sim:.4f}, 最大结构相似度={max_struct_sim:.4f}")
        print(f"高相似度对(>0.2): {len(high_sim_pairs)} 对")
        if high_sim_pairs and len(high_sim_pairs) <= 10:
            for pair in high_sim_pairs[:5]:
                print(f"  {pair[0]} vs {pair[1]}: seq={pair[2]:.4f}, struct={pair[3]:.4f}")
        
        self.clusters = clusters
        print(f"找到 {len(clusters)} 个聚类")
        
        return clusters
    
    def transfer_labels(self, 
                        protein_data: Dict[str, Dict],
                        clusters: List[List[str]] = None) -> Dict[str, Dict]:
        """
        转移热点残基标注
        
        策略：将同一聚类中的热点残基标注转移到残基最多的链上
        
        Args:
            protein_data: 蛋白质数据字典
            clusters: 聚类结果
            
        Returns:
            增强后的蛋白质数据
        """
        if clusters is None:
            clusters = self.clusters
            
        enhanced_data = {k: v.copy() for k, v in protein_data.items()}
        
        transferred_count = 0
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
                
            max_len = 0
            target_chain = None
            
            for chain in cluster:
                seq_len = len(protein_data[chain].get('sequence', ''))
                if seq_len > max_len:
                    max_len = seq_len
                    target_chain = chain
            
            if target_chain is None:
                continue
                
            all_hotspots = set(protein_data[target_chain].get('hotspots', []))
            
            for chain in cluster:
                if chain == target_chain:
                    continue
                    
                hotspots = protein_data[chain].get('hotspots', [])
                
                for hotspot in hotspots:
                    if hotspot <= max_len:
                        all_hotspots.add(hotspot)
                        transferred_count += 1
            
            enhanced_data[target_chain]['hotspots'] = sorted(list(all_hotspots))
            enhanced_data[target_chain]['transferred'] = True
        
        print(f"转移了 {transferred_count} 个热点残基标注")
        
        return enhanced_data
    
    def augment_dataset(self, 
                        protein_data: Dict[str, Dict],
                        use_external_tools: bool = False) -> Tuple[Dict[str, Dict], Dict]:
        """
        完整的数据增强流程
        
        Args:
            protein_data: 蛋白质数据字典
            use_external_tools: 是否使用外部工具
            
        Returns:
            (增强后的数据, 统计信息)
        """
        print("=" * 60)
        print("开始标签转移数据增强")
        print("=" * 60)
        
        original_hotspot_count = sum(
            len(v.get('hotspots', [])) for v in protein_data.values()
        )
        
        similarity_matrix = self.build_similarity_matrix(protein_data, use_external_tools)
        
        clusters = self.cluster_proteins(protein_data, similarity_matrix)
        
        enhanced_data = self.transfer_labels(protein_data, clusters)
        
        new_hotspot_count = sum(
            len(v.get('hotspots', [])) for v in enhanced_data.values()
        )
        
        stats = {
            'original_chains': len(protein_data),
            'num_clusters': len(clusters),
            'original_hotspots': original_hotspot_count,
            'new_hotspots': new_hotspot_count,
            'increase_ratio': (new_hotspot_count - original_hotspot_count) / original_hotspot_count * 100 if original_hotspot_count > 0 else 0
        }
        
        print("\n" + "=" * 60)
        print("数据增强统计")
        print("=" * 60)
        print(f"原始蛋白链数: {stats['original_chains']}")
        print(f"聚类数: {stats['num_clusters']}")
        print(f"原始热点残基数: {stats['original_hotspots']}")
        print(f"增强后热点残基数: {stats['new_hotspots']}")
        print(f"增加比例: {stats['increase_ratio']:.2f}%")
        print("=" * 60)
        
        return enhanced_data, stats


def prepare_protein_data_for_transfer(data_list: List[Dict]) -> Dict[str, Dict]:
    """
    准备蛋白质数据用于标签转移
    
    Args:
        data_list: 数据列表 [{'pdb_id': str, 'chain': str, 'sequence': str, 'coords': np.ndarray, 'hotspots': List[int]}]
        
    Returns:
        蛋白质数据字典
    """
    protein_data = {}
    
    for item in data_list:
        chain_id = f"{item['pdb_id']}_{item.get('chain', 'A')}"
        
        protein_data[chain_id] = {
            'sequence': item.get('sequence', ''),
            'coords': item.get('coords'),
            'hotspots': item.get('hotspots', []),
            'pdb_file': item.get('pdb_file')
        }
    
    return protein_data


def integrate_with_existing_pipeline(data_list: List[Dict], 
                                     use_external_tools: bool = False) -> Tuple[List[Dict], Dict]:
    """
    与现有数据处理流程集成
    
    Args:
        data_list: 原始数据列表
        use_external_tools: 是否使用外部工具
        
    Returns:
        (增强后的数据列表, 统计信息)
    """
    protein_data = prepare_protein_data_for_transfer(data_list)
    
    transfer = LabelTransfer()
    enhanced_data, stats = transfer.augment_dataset(protein_data, use_external_tools)
    
    enhanced_list = []
    for chain_id, data in enhanced_data.items():
        pdb_id, chain = chain_id.rsplit('_', 1)
        
        enhanced_list.append({
            'pdb_id': pdb_id,
            'chain': chain,
            'sequence': data['sequence'],
            'coords': data['coords'],
            'hotspots': data['hotspots'],
            'transferred': data.get('transferred', False)
        })
    
    return enhanced_list, stats


if __name__ == '__main__':
    print("标签转移模块测试")
    
    test_data = {
        '1A1T_A': {
            'sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH',
            'coords': np.random.randn(50, 3),
            'hotspots': [10, 20, 30]
        },
        '1A1T_B': {
            'sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH',
            'coords': np.random.randn(50, 3),
            'hotspots': [15, 25]
        },
        '2B2B_A': {
            'sequence': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'coords': np.random.randn(26, 3),
            'hotspots': [5, 10]
        }
    }
    
    transfer = LabelTransfer()
    enhanced_data, stats = transfer.augment_dataset(test_data)
    
    print("\n增强后的数据:")
    for chain_id, data in enhanced_data.items():
        print(f"{chain_id}: {len(data['hotspots'])} hotspots")
