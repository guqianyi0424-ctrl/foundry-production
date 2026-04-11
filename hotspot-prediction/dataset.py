"""
数据集处理模块
功能：
1. 加载原始数据
2. 构建蛋白质图
3. 提取ESM-2特征和传统特征
4. 创建PyTorch Dataset
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.Polypeptide import protein_letters_3to1
import dgl
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, TEST_DATA_FILE, PDB_DIR, FEATURES_DIR, AMINO_ACIDS, AA_PROPERTIES,
    MAP_CUTOFF, DIST_NORM, ESM2_MODEL, ESM2_DIM, DEVICE,
    PSSM_DIM, DSSP_DIM, HMM_DIM, TRADITIONAL_DIM
)


def load_raw_data():
    """加载原始Excel数据"""
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()
    
    if 'hotspot_list' in df.columns:
        df = df.rename(columns={
            'uniprot_id': 'uniprot_id',
            'partner_id': 'partner_id',
            'pdb_id': 'pdb_id',
            'hotspot_list': 'hotspot_list'
        })
    else:
        df = df.rename(columns={
            'Uniprot code': 'uniprot_id',
            'Uniprot code of the binding partner': 'partner_id',
            'PDB ID': 'pdb_id',
            'PDB # of PPI-hot spots': 'residue_num',
            'PPI-hot spots  assignment': 'label'
        })
    
    df['pdb_id_clean'] = df['pdb_id'].str.lower().str.split('-').str[0].str.strip()
    df['chain_id'] = df['pdb_id'].str.split('-').str[1].fillna('A').str.strip()
    
    return df


def download_pdb_files(pdb_ids):
    """批量下载PDB文件"""
    pdbl = PDBList()
    downloaded = []
    
    for pdb_id in tqdm(set(pdb_ids), desc="下载PDB文件"):
        pdb_id = pdb_id.lower().strip()
        if len(pdb_id) != 4:
            continue
        
        pdb_file = os.path.join(PDB_DIR, f'pdb{pdb_id}.ent')
        if os.path.exists(pdb_file):
            downloaded.append(pdb_id)
            continue
        
        try:
            pdbl.retrieve_pdb_file(pdb_id, pdir=PDB_DIR, file_format='pdb')
            downloaded.append(pdb_id)
        except Exception as e:
            print(f"下载失败 {pdb_id}: {e}")
    
    return downloaded


def extract_sequence_and_coords(pdb_file, chain_id='A'):
    """从PDB文件提取序列和坐标"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    model = structure[0]
    if chain_id not in model:
        available_chains = [c.id for c in model.get_chains()]
        chain_id = available_chains[0] if available_chains else 'A'
    
    chain = model[chain_id]
    
    sequence = []
    coords = []
    residue_indices = []
    b_factors = []
    
    for residue in chain.get_residues():
        resname = residue.get_resname()
        aa = protein_letters_3to1.get(resname, 'X')
        
        if aa == 'X':
            continue
        
        if 'CA' not in residue:
            continue
        
        ca_coord = residue['CA'].get_coord()
        b_factor = residue['CA'].get_bfactor()
        
        sequence.append(aa)
        coords.append(ca_coord)
        residue_indices.append(residue.get_id()[1])
        b_factors.append(b_factor)
    
    return {
        'sequence': ''.join(sequence),
        'coords': np.array(coords),
        'residue_indices': residue_indices,
        'b_factors': np.array(b_factors)
    }


def calculate_distance_matrix(coords):
    """计算距离矩阵"""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def build_graph(coords, cutoff=MAP_CUTOFF):
    """构建蛋白质图 - 对齐DeepHotResi: 距离+余弦相似度边特征"""
    n_nodes = len(coords)
    dist_matrix = calculate_distance_matrix(coords)
    
    mask = (dist_matrix > 0) & (dist_matrix <= cutoff)
    src, dst = np.where(mask)
    
    edge_index = np.array([src, dst])
    
    distances = dist_matrix[src, dst] / cutoff
    
    pos = np.array(coords)
    pos_ref = pos - pos[0]
    
    from numpy.linalg import norm
    edge_cos = []
    for s, d in zip(src, dst):
        v1 = pos_ref[s]
        v2 = pos_ref[d]
        n1 = norm(v1)
        n2 = norm(v2)
        if n1 > 0 and n2 > 0:
            cos_sim = np.dot(v1, v2) / (n1 * n2)
            edge_cos.append((cos_sim + 1) / 2)
        else:
            edge_cos.append(0.5)
    
    edge_feat = np.array([distances, np.array(edge_cos)])
    
    return edge_index, edge_feat, dist_matrix


def normalize_adjacency(dist_matrix, cutoff=MAP_CUTOFF):
    """归一化邻接矩阵"""
    mask = (dist_matrix > 0) & (dist_matrix <= cutoff)
    adj = mask.astype(np.float32)
    
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5, where=rowsum != 0)
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    
    norm_adj = r_mat_inv @ adj @ r_mat_inv
    return norm_adj


def one_hot_encode_aa(aa):
    """氨基酸one-hot编码"""
    encoding = np.zeros(len(AMINO_ACIDS))
    if aa in AMINO_ACIDS:
        encoding[AMINO_ACIDS.index(aa)] = 1
    return encoding


def extract_traditional_features(sequence, b_factors):
    """提取传统物理化学特征"""
    features = []
    
    for i, aa in enumerate(sequence):
        props = AA_PROPERTIES.get(aa, {'hydrophobicity': 0, 'charge': 0, 'mw': 0, 'volume': 0, 'polarity': 0})
        
        aa_onehot = one_hot_encode_aa(aa)
        
        rel_pos = i / max(len(sequence) - 1, 1)
        
        feat = np.concatenate([
            [props['hydrophobicity']],
            [props['charge']],
            [props['mw'] / 200],
            [props['volume'] / 250],
            [props['polarity']],
            [b_factors[i] / 100],
            [rel_pos],
            aa_onehot
        ])
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def extract_dssp_features(pdb_file, chain_id='A'):
    """
    从PDB文件提取DSSP特征
    返回: (L, 14) 维特征矩阵
    特征包括:
    - 二级结构 one-hot (8维)
    - 相对可及表面积 RSA (1维)
    - Phi角 sin/cos (2维)
    - Psi角 sin/cos (2维)
    - NH_O_1 氢键 (1维)
    """
    try:
        from Bio.PDB.DSSP import DSSP
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        
        if chain_id not in model:
            available_chains = [c.id for c in model.get_chains()]
            chain_id = available_chains[0] if available_chains else 'A'
        
        dssp = DSSP(model, pdb_file, dssp='mkdssp')
        
        features = []
        dssp_keys = [k for k in dssp.keys() if k[0] == chain_id]
        
        for key in dssp_keys:
            dssp_data = dssp[key]
            
            ss = dssp_data[2]
            ss_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, ' ': 7}
            ss_onehot = np.zeros(8)
            if ss in ss_dict:
                ss_onehot[ss_dict[ss]] = 1
            
            rsa = dssp_data[3] / 100.0 if dssp_data[3] else 0
            
            phi = dssp_data[4] if dssp_data[4] else 0
            phi_sin = np.sin(np.radians(phi))
            phi_cos = np.cos(np.radians(phi))
            
            psi = dssp_data[5] if dssp_data[5] else 0
            psi_sin = np.sin(np.radians(psi))
            psi_cos = np.cos(np.radians(psi))
            
            nh_o_1 = dssp_data[6] / 10.0 if dssp_data[6] else 0
            
            feat = np.concatenate([
                ss_onehot,
                [rsa],
                [phi_sin, phi_cos],
                [psi_sin, psi_cos],
                [nh_o_1]
            ])
            features.append(feat)
        
        if len(features) == 0:
            return None
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"DSSP提取失败: {e}")
        return None


def generate_default_dssp_features(seq_length):
    """生成默认DSSP特征 (当DSSP不可用时)"""
    features = np.zeros((seq_length, DSSP_DIM), dtype=np.float32)
    features[:, 7] = 1
    return features


def extract_pssm_features(sequence, pdb_id):
    """
    提取PSSM特征 (位置特异性打分矩阵)
    需要预先运行PSI-BLAST生成PSSM文件
    如果没有PSSM文件，使用默认特征
    返回: (L, 20) 维特征矩阵
    """
    pssm_file = os.path.join(FEATURES_DIR, 'pssm', f'{pdb_id}.npy')
    
    if os.path.exists(pssm_file):
        pssm = np.load(pssm_file)
        if pssm.shape[0] == len(sequence):
            return pssm.astype(np.float32)
    
    return generate_default_pssm_features(len(sequence))


def generate_default_pssm_features(seq_length):
    """生成默认PSSM特征 (当PSSM不可用时)"""
    features = np.zeros((seq_length, PSSM_DIM), dtype=np.float32)
    return features


def run_psi_blast(fasta_file, output_dir, database='swissprot'):
    """
    运行PSI-BLAST生成PSSM
    需要安装BLAST+并下载数据库
    
    参数:
        fasta_file: 输入FASTA文件
        output_dir: 输出目录
        database: BLAST数据库名称
    """
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'pssm.txt')
    
    cmd = [
        'psiblast',
        '-query', fasta_file,
        '-db', database,
        '-num_iterations', '3',
        '-out_ascii_pssm', output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_file
    except Exception as e:
        print(f"PSI-BLAST运行失败: {e}")
        return None


def parse_pssm_file(pssm_file, sequence):
    """
    解析PSSM文件
    
    参数:
        pssm_file: PSSM文件路径
        sequence: 蛋白质序列
    
    返回:
        (L, 20) 的PSSM矩阵
    """
    try:
        with open(pssm_file, 'r') as f:
            lines = f.readlines()
        
        pssm_values = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 22:
                    scores = [float(x) / 100.0 for x in parts[2:22]]
                    pssm_values.append(scores)
        
        pssm = np.array(pssm_values, dtype=np.float32)
        
        if pssm.shape[0] != len(sequence):
            pssm = generate_default_pssm_features(len(sequence))
        
        return pssm
        
    except Exception as e:
        print(f"PSSM解析失败: {e}")
        return generate_default_pssm_features(len(sequence))


def extract_hmm_features(sequence, pdb_id):
    """
    提取HMM特征 (隐马尔可夫模型)
    需要预先运行HHblits生成.hhm文件
    如果没有HMM文件，使用默认特征
    返回: (L, 30) 维特征矩阵
    
    HMM特征包括:
    - 20个氨基酸的发射概率
    - 10个转移概率相关特征
    """
    hmm_file = os.path.join(FEATURES_DIR, 'hmm', f'{pdb_id}.npy')
    
    if os.path.exists(hmm_file):
        hmm = np.load(hmm_file)
        if hmm.shape[0] == len(sequence):
            return hmm.astype(np.float32)
    
    return generate_default_hmm_features(len(sequence))


def generate_default_hmm_features(seq_length):
    """生成默认HMM特征 (当HMM不可用时)"""
    features = np.zeros((seq_length, HMM_DIM), dtype=np.float32)
    return features


def parse_hhm_file(hhm_file, sequence):
    """
    解析HHM文件 (HHblits输出)
    
    参数:
        hhm_file: .hhm文件路径
        sequence: 蛋白质序列
    
    返回:
        (L, 30) 的HMM特征矩阵
    """
    try:
        with open(hhm_file, 'r') as f:
            lines = f.readlines()
        
        hmm_values = []
        reading_probs = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#') or line == '':
                continue
            
            if line.startswith('NAME') or line.startswith('FAM') or line.startswith('FILE'):
                continue
            
            if line.startswith('SEQ') or line.startswith('SEQRES'):
                continue
            
            if line.startswith('HMM'):
                reading_probs = True
                continue
            
            if reading_probs:
                parts = line.split()
                if len(parts) >= 23:
                    try:
                        probs = []
                        for i in range(2, 22):
                            val = parts[i]
                            if val == '*':
                                probs.append(0.0)
                            else:
                                probs.append(float(val) / 100.0)
                        
                        for i in range(22, 32):
                            if i < len(parts):
                                val = parts[i]
                                if val == '*':
                                    probs.append(0.0)
                                else:
                                    probs.append(float(val) / 100.0)
                            else:
                                probs.append(0.0)
                        
                        hmm_values.append(probs[:30])
                    except (ValueError, IndexError):
                        continue
        
        if len(hmm_values) == 0:
            return generate_default_hmm_features(len(sequence))
        
        hmm = np.array(hmm_values, dtype=np.float32)
        
        if hmm.shape[0] != len(sequence):
            hmm = generate_default_hmm_features(len(sequence))
        
        return hmm
        
    except Exception as e:
        print(f"HMM解析失败: {e}")
        return generate_default_hmm_features(len(sequence))


def run_hhblits(fasta_file, output_dir, database='uniclust30_2018_08'):
    """
    运行HHblits生成HMM文件
    需要安装HH-suite并下载数据库
    
    参数:
        fasta_file: 输入FASTA文件
        output_dir: 输出目录
        database: HHblits数据库名称
    """
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'output.hhm')
    
    cmd = [
        'hhblits',
        '-i', fasta_file,
        '-d', database,
        '-o', os.path.join(output_dir, 'output.hhr'),
        '-ohhm', output_file,
        '-n', '3'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_file
    except Exception as e:
        print(f"HHblits运行失败: {e}")
        return None


def concatenate_features(esm2_feats, pssm_feats, hmm_feats, trad_feats):
    """根据配置拼接特征"""
    features_list = [esm2_feats]
    
    if PSSM_DIM > 0:
        features_list.append(pssm_feats)
    if HMM_DIM > 0:
        features_list.append(hmm_feats)
    if TRADITIONAL_DIM > 0:
        features_list.append(trad_feats)
    
    return np.concatenate(features_list, axis=1)


def extract_esm2_features(sequence, model=None, tokenizer=None):
    """提取ESM-2特征"""
    if model is None or tokenizer is None:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
        model = AutoModel.from_pretrained(ESM2_MODEL).to(DEVICE)
        model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
    
    return embeddings


class PPIHotspotDataset(Dataset):
    """PPI热点残基预测数据集"""
    
    def __init__(self, data_list, esm2_model=None, tokenizer=None):
        self.data_list = data_list
        self.esm2_model = esm2_model
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        node_features = torch.tensor(item['node_features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        adj_matrix = torch.tensor(item['adj_matrix'], dtype=torch.float32)
        
        src = item['edge_index'][0]
        dst = item['edge_index'][1]
        num_nodes = node_features.shape[0]
        
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        
        if 'edge_feat' in item and len(item['edge_feat']) > 0:
            g.edata['feat'] = torch.tensor(item['edge_feat'].T, dtype=torch.float32)
        
        return {
            'pdb_id': item['pdb_id'],
            'node_features': node_features,
            'labels': labels,
            'graph': g,
            'adj_matrix': adj_matrix,
            'sequence': item['sequence']
        }


def parse_hotspot_list(hotspot_str):
    """解析热点残基列表字符串"""
    if pd.isna(hotspot_str) or hotspot_str == '':
        return []
    
    hotspots = []
    for item in str(hotspot_str).split(','):
        item = item.strip()
        if item:
            try:
                hotspots.append(int(item))
            except ValueError:
                continue
    return hotspots


def prepare_dataset():
    """准备完整数据集"""
    print("=" * 60)
    print("步骤1: 加载原始数据")
    print("=" * 60)
    df = load_raw_data()
    print(f"加载了 {len(df)} 条记录")
    
    if 'hotspot_list' in df.columns:
        print(f"数据格式: 合并数据集")
        total_hotspots = df['hotspot_list'].apply(lambda x: len(parse_hotspot_list(x)) if pd.notna(x) else 0).sum()
        print(f"总热点数: {total_hotspots}")
    else:
        print(f"标签分布:\n{df['label'].value_counts()}")
    
    print("\n" + "=" * 60)
    print("步骤2: 下载PDB文件")
    print("=" * 60)
    unique_pdbs = df['pdb_id_clean'].unique()
    downloaded = download_pdb_files(unique_pdbs)
    print(f"成功下载 {len(downloaded)} 个PDB文件")
    
    print("\n" + "=" * 60)
    print("步骤3: 提取特征并构建图")
    print("=" * 60)
    
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    esm2_model = AutoModel.from_pretrained(ESM2_MODEL).to(DEVICE)
    esm2_model.eval()
    
    data_list = []
    
    if 'hotspot_list' in df.columns:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理蛋白质"):
            pdb_id = row['pdb_id_clean']
            chain_id = row['chain_id']
            
            pdb_file = os.path.join(PDB_DIR, f'pdb{pdb_id}.ent')
            
            if not os.path.exists(pdb_file):
                continue
            
            try:
                protein_data = extract_sequence_and_coords(pdb_file, chain_id)
                
                if len(protein_data['sequence']) == 0:
                    continue
                
                esm2_feats = extract_esm2_features(protein_data['sequence'], esm2_model, tokenizer)
                
                pssm_feats = extract_pssm_features(protein_data['sequence'], f"{pdb_id}_{chain_id}")
                
                hmm_feats = extract_hmm_features(protein_data['sequence'], f"{pdb_id}_{chain_id}")
                
                trad_feats = extract_traditional_features(
                    protein_data['sequence'], 
                    protein_data['b_factors']
                )
                
                node_features = concatenate_features(esm2_feats, pssm_feats, hmm_feats, trad_feats)
                
                edge_index, edge_feat, dist_matrix = build_graph(protein_data['coords'])
                
                adj_matrix = normalize_adjacency(dist_matrix)
                
                labels = np.zeros(len(protein_data['sequence']), dtype=np.int64)
                hotspots = parse_hotspot_list(row['hotspot_list'])
                for hotspot_pos in hotspots:
                    if hotspot_pos in protein_data['residue_indices']:
                        idx = protein_data['residue_indices'].index(hotspot_pos)
                        labels[idx] = 1
                
                data_list.append({
                    'pdb_id': f"{pdb_id}_{chain_id}",
                    'sequence': protein_data['sequence'],
                    'node_features': node_features,
                    'labels': labels,
                    'edge_index': edge_index,
                    'edge_feat': edge_feat,
                    'adj_matrix': adj_matrix,
                    'coords': protein_data['coords']
                })
                
            except Exception as e:
                print(f"处理 {pdb_id}_{chain_id} 时出错: {e}")
                continue
    else:
        grouped = df.groupby(['pdb_id_clean', 'chain_id'])
        
        for (pdb_id, chain_id), group in tqdm(grouped, desc="处理蛋白质"):
            pdb_file = os.path.join(PDB_DIR, f'pdb{pdb_id}.ent')
            
            if not os.path.exists(pdb_file):
                continue
            
            try:
                protein_data = extract_sequence_and_coords(pdb_file, chain_id)
                
                if len(protein_data['sequence']) == 0:
                    continue
                
                esm2_feats = extract_esm2_features(protein_data['sequence'], esm2_model, tokenizer)
                
                pssm_feats = extract_pssm_features(protein_data['sequence'], f"{pdb_id}_{chain_id}")
                
                hmm_feats = extract_hmm_features(protein_data['sequence'], f"{pdb_id}_{chain_id}")
                
                trad_feats = extract_traditional_features(
                    protein_data['sequence'], 
                    protein_data['b_factors']
                )
                
                node_features = concatenate_features(esm2_feats, pssm_feats, hmm_feats, trad_feats)
                
                edge_index, edge_feat, dist_matrix = build_graph(protein_data['coords'])
                
                adj_matrix = normalize_adjacency(dist_matrix)
                
                labels = np.zeros(len(protein_data['sequence']), dtype=np.int64)
                for _, row in group.iterrows():
                    res_num = row['residue_num']
                    if res_num in protein_data['residue_indices']:
                        idx = protein_data['residue_indices'].index(res_num)
                        labels[idx] = 1 if row['label'] == 'P' else 0
                
                data_list.append({
                    'pdb_id': f"{pdb_id}_{chain_id}",
                    'sequence': protein_data['sequence'],
                    'node_features': node_features,
                    'labels': labels,
                    'edge_index': edge_index,
                    'edge_feat': edge_feat,
                    'adj_matrix': adj_matrix,
                    'coords': protein_data['coords']
                })
                
            except Exception as e:
                print(f"处理 {pdb_id}_{chain_id} 时出错: {e}")
                continue
    
    print(f"\n成功处理 {len(data_list)} 个蛋白质")
    
    output_file = os.path.join(FEATURES_DIR, 'dataset.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"数据集已保存到: {output_file}")
    
    return data_list


def collate_fn(batch):
    """自定义批处理函数"""
    pdb_ids = [item['pdb_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch], batch_first=True, padding_value=-1
    )
    adj_matrices = [item['adj_matrix'] for item in batch]
    
    graphs = dgl.batch([item['graph'] for item in batch])
    
    node_features = torch.cat([item['node_features'] for item in batch], dim=0)
    
    return {
        'pdb_ids': pdb_ids,
        'sequences': sequences,
        'node_features': node_features,
        'labels': labels,
        'graphs': graphs,
        'adj_matrices': adj_matrices
    }


def calculate_sample_weights(data_list):
    """计算每个蛋白质样本的权重（基于热点残基比例）"""
    weights = []
    for item in data_list:
        labels = item['labels']
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        total = n_pos + n_neg
        
        if n_pos > 0 and n_neg > 0:
            weight = total / (2 * n_pos) if n_pos < n_neg else total / (2 * n_neg)
        else:
            weight = 1.0
        
        weights.append(weight)
    
    return weights


def get_weighted_sampler(data_list):
    """获取加权采样器"""
    from torch.utils.data import WeightedRandomSampler
    
    weights = calculate_sample_weights(data_list)
    weights = torch.DoubleTensor(weights)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def calculate_class_weights(data_list):
    """计算类别权重"""
    total_pos = 0
    total_neg = 0
    
    for item in data_list:
        labels = item['labels']
        total_pos += np.sum(labels == 1)
        total_neg += np.sum(labels == 0)
    
    total = total_pos + total_neg
    
    if total_pos > 0 and total_neg > 0:
        pos_weight = total / (2 * total_pos)
        neg_weight = total / (2 * total_neg)
    else:
        pos_weight = 1.0
        neg_weight = 1.0
    
    return {
        'pos_weight': pos_weight,
        'neg_weight': neg_weight,
        'class_weights': torch.FloatTensor([neg_weight, pos_weight]),
        'imbalance_ratio': total_neg / total_pos if total_pos > 0 else 1.0
    }


def apply_smote_to_features(node_features, labels, k_neighbors=5):
    """对节点特征应用 SMOTE 过采样"""
    from imblearn.over_sampling import SMOTE
    
    valid_mask = labels >= 0
    X = node_features[valid_mask]
    y = labels[valid_mask]
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if n_pos < k_neighbors or n_neg < k_neighbors:
        return node_features, labels
    
    try:
        smote = SMOTE(k_neighbors=min(k_neighbors, n_pos, n_neg), random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled
    except Exception as e:
        print(f"SMOTE 失败: {e}")
        return node_features, labels


def load_test_data():
    """加载独立测试数据"""
    df = pd.read_excel(TEST_DATA_FILE)
    df.columns = df.columns.str.strip()
    
    if 'hotspot_list' in df.columns:
        df['pdb_id_clean'] = df['pdb_id'].str.lower().str.split('-').str[0].str.strip()
        df['chain_id'] = df['pdb_id'].str.split('-').str[1].fillna('A').str.strip()
        return df
    
    print(f"测试数据列名: {list(df.columns)}")
    
    column_mapping = {
        'Uniprot code of protein A': 'uniprot_id',
        'Uniprot code of the binding partner, protein B': 'partner_id',
        'PDB ID of free protein A structure(length in Uniprot)': 'pdb_free',
        'PDB ID of bound protein A structure (length in Uniprot)': 'pdb_bound_a',
        'PDB ID of bound protein B structure': 'pdb_bound_b',
        'PDB ID of bound protein B structure ': 'pdb_bound_b',
        'Sequence identity': 'seq_identity',
        'PPI-hot spots (Uniprot numbering)': 'hotspot_list'
    }
    
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    
    df = df.rename(columns=rename_dict)
    
    if 'pdb_bound_b' not in df.columns:
        df['pdb_bound_b'] = None
    
    return df


def prepare_test_dataset(force_reload=False):
    """准备独立测试数据集"""
    output_file = os.path.join(FEATURES_DIR, 'test_dataset.pkl')
    
    if os.path.exists(output_file) and not force_reload:
        print(f"加载已保存的测试数据集: {output_file}")
        with open(output_file, 'rb') as f:
            return pickle.load(f)
    
    print("\n" + "=" * 60)
    print("准备独立测试数据集")
    print("=" * 60)
    
    df = load_test_data()
    print(f"加载 {len(df)} 条测试数据")
    
    print("\n步骤1: 下载PDB文件")
    
    if 'pdb_id_clean' in df.columns:
        pdb_ids = df['pdb_id_clean'].dropna().unique()
        pdb_ids = [p.lower().strip()[:4] for p in pdb_ids if len(str(p).strip()) >= 4]
    else:
        pdb_ids = set()
        for _, row in df.iterrows():
            for col in ['pdb_free', 'pdb_bound_a', 'pdb_bound_b']:
                if col in df.columns and pd.notna(row[col]):
                    pdb_id = str(row[col]).split('(')[0].strip().lower()
                    if len(pdb_id) >= 4:
                        pdb_ids.add(pdb_id[:4])
        pdb_ids = list(pdb_ids)
    
    download_pdb_files(list(set(pdb_ids)))
    
    print("\n步骤2: 加载ESM-2模型")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    esm2_model = AutoModel.from_pretrained(ESM2_MODEL).to(DEVICE)
    esm2_model.eval()
    
    print("\n步骤3: 处理测试数据")
    data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理测试数据"):
        if 'pdb_id_clean' in df.columns:
            pdb_id = row['pdb_id_clean']
            chain_id = row['chain_id']
        else:
            pdb_info = row.get('pdb_free')
            if pd.isna(pdb_info):
                continue
            
            pdb_id = str(pdb_info).split('(')[0].strip().lower()
            if len(pdb_id) < 4:
                continue
            
            pdb_id = pdb_id[:4]
            chain_id = pdb_id[4:].upper() if len(pdb_id) > 4 else 'A'
        
        if pd.isna(pdb_id) or len(str(pdb_id).strip()) < 4:
            continue
        
        pdb_id = str(pdb_id).strip().lower()[:4]
        
        pdb_file = os.path.join(PDB_DIR, f'pdb{pdb_id}.ent')
        if not os.path.exists(pdb_file):
            continue
        
        try:
            protein_data = extract_sequence_and_coords(pdb_file, chain_id)
            
            if len(protein_data['sequence']) == 0:
                continue
            
            esm2_feats = extract_esm2_features(protein_data['sequence'], esm2_model, tokenizer)
            
            pssm_feats = extract_pssm_features(protein_data['sequence'], f"{pdb_id}_{chain_id}")
            hmm_feats = extract_hmm_features(protein_data['sequence'], f"{pdb_id}_{chain_id}")
            trad_feats = extract_traditional_features(
                protein_data['sequence'], 
                protein_data['b_factors']
            )
            
            node_features = concatenate_features(esm2_feats, pssm_feats, hmm_feats, trad_feats)
            
            edge_index, edge_feat, dist_matrix = build_graph(protein_data['coords'])
            adj_matrix = normalize_adjacency(dist_matrix)
            
            hotspots = parse_hotspot_list(row.get('hotspot_list', ''))
            
            labels = np.zeros(len(protein_data['sequence']), dtype=np.int64)
            for hotspot_pos in hotspots:
                if hotspot_pos in protein_data['residue_indices']:
                    res_idx = protein_data['residue_indices'].index(hotspot_pos)
                    labels[res_idx] = 1
            
            data_list.append({
                'pdb_id': f"{pdb_id}_{chain_id}",
                'uniprot_id': row.get('uniprot_id', ''),
                'partner_id': row.get('partner_id', ''),
                'sequence': protein_data['sequence'],
                'node_features': node_features,
                'labels': labels,
                'edge_index': edge_index,
                'edge_feat': edge_feat,
                'adj_matrix': adj_matrix,
                'coords': protein_data['coords'],
                'hotspot_list': hotspots
            })
            
        except Exception as e:
            print(f"处理测试数据 {idx} 时出错: {e}")
            continue
    
    print(f"\n成功处理 {len(data_list)} 个测试蛋白质")
    
    with open(output_file, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"测试数据集已保存到: {output_file}")
    
    return data_list


if __name__ == '__main__':
    data_list = prepare_dataset()
