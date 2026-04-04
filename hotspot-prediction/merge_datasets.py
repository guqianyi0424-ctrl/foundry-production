"""
重新划分数据集
解决训练数据和测试数据分布不一致的问题
确保按 Uniprot ID 划分，避免数据泄露
修正：使用 PDB 编号而不是 Uniprot 编号
"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE1 = os.path.join(BASE_DIR, 'elife-96643-data1-v1.xlsx')
DATA_FILE2 = os.path.join(BASE_DIR, 'elife-96643-data2-v1.xlsx')

OUTPUT_TRAIN = os.path.join(BASE_DIR, 'merged_train.xlsx')
OUTPUT_TEST = os.path.join(BASE_DIR, 'merged_test.xlsx')


def parse_hotspots(hotspot_str):
    """解析热点残基字符串"""
    if pd.isna(hotspot_str):
        return []
    return [h.strip() for h in str(hotspot_str).split(',') if h.strip()]


def parse_pdb_offset(pdb_info):
    """
    从 PDB 信息中提取偏移量
    例如: '5f18-A(374-620)' -> (374, 620)
    """
    if pd.isna(pdb_info):
        return None, None
    
    match = re.search(r'\((\d+)-(\d+)\)', str(pdb_info))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def uniprot_to_pdb_numbering(uniprot_positions, start_offset):
    """
    将 Uniprot 编号转换为 PDB 编号
    例如: Uniprot 503, offset 374 -> PDB 130
    """
    if start_offset is None:
        return []
    
    pdb_positions = []
    for pos in uniprot_positions:
        try:
            pdb_pos = int(pos) - start_offset + 1
            if pdb_pos > 0:
                pdb_positions.append(str(pdb_pos))
        except:
            continue
    return pdb_positions


def merge_datasets():
    """合并两个数据集"""
    print("=" * 60)
    print("合并数据集 (使用 PDB 编号)")
    print("=" * 60)
    
    df1 = pd.read_excel(DATA_FILE1)
    df2 = pd.read_excel(DATA_FILE2)
    
    print(f"\n数据集1: {len(df1)} 条记录")
    print(f"数据集2: {len(df2)} 条记录")
    
    merged_data = []
    
    for idx, row in df1.iterrows():
        pdb_hotspots = parse_hotspots(row.get('PDB # of PPI-hot spots ', ''))
        uniprot_hotspots = parse_hotspots(row.get('Uniprot # of PPI-hot spots ', ''))
        
        hotspots_to_use = pdb_hotspots if pdb_hotspots else uniprot_hotspots
        
        merged_data.append({
            'uniprot_id': row.get('Uniprot code', ''),
            'partner_id': row.get('Uniprot code of the binding partner', ''),
            'pdb_id': row.get('PDB ID', ''),
            'hotspot_list': ','.join(hotspots_to_use) if hotspots_to_use else '',
            'n_hotspots': len(hotspots_to_use),
            'source': 'data1',
            'numbering_type': 'pdb' if pdb_hotspots else 'uniprot'
        })
    
    for idx, row in df2.iterrows():
        pdb_info = row.get('PDB ID of free protein A structure(length in Uniprot)', '')
        pdb_id = str(pdb_info).split('(')[0].strip() if pd.notna(pdb_info) else ''
        
        uniprot_hotspots = parse_hotspots(row.get('PPI-hot spots (Uniprot numbering)', ''))
        
        start_offset, end_offset = parse_pdb_offset(pdb_info)
        
        if start_offset is not None and uniprot_hotspots:
            pdb_hotspots = uniprot_to_pdb_numbering(uniprot_hotspots, start_offset)
            numbering_type = 'pdb_converted'
        else:
            pdb_hotspots = uniprot_hotspots
            numbering_type = 'uniprot'
        
        merged_data.append({
            'uniprot_id': row.get('Uniprot code of protein A', ''),
            'partner_id': row.get('Uniprot code of the binding partner, protein B', ''),
            'pdb_id': pdb_id,
            'hotspot_list': ','.join(pdb_hotspots) if pdb_hotspots else '',
            'n_hotspots': len(pdb_hotspots),
            'source': 'data2',
            'numbering_type': numbering_type
        })
    
    merged_df = pd.DataFrame(merged_data)
    
    print(f"\n合并后: {len(merged_df)} 条记录")
    print(f"总热点数: {merged_df['n_hotspots'].sum()}")
    print(f"平均每条记录热点数: {merged_df['n_hotspots'].mean():.2f}")
    
    print("\n热点分布:")
    print(merged_df['n_hotspots'].value_counts().sort_index())
    
    print("\n数据来源分布:")
    print(merged_df['source'].value_counts())
    
    print("\n编号类型分布:")
    print(merged_df['numbering_type'].value_counts())
    
    return merged_df


def split_data_by_uniprot(merged_df, test_ratio=0.3, random_state=42):
    """按 Uniprot ID 划分训练集和测试集，避免数据泄露"""
    print("\n" + "=" * 60)
    print("按 Uniprot ID 划分数据集 (避免数据泄露)")
    print("=" * 60)
    
    unique_uniprots = merged_df['uniprot_id'].unique()
    print(f"\n唯一 Uniprot ID 数: {len(unique_uniprots)}")
    
    uniprot_hotspot_counts = merged_df.groupby('uniprot_id')['n_hotspots'].mean()
    
    uniprot_df = pd.DataFrame({
        'uniprot_id': unique_uniprots,
        'avg_hotspots': [uniprot_hotspot_counts.get(uid, 0) for uid in unique_uniprots]
    })
    
    uniprot_df['hotspot_bin'] = pd.cut(
        uniprot_df['avg_hotspots'], 
        bins=[0, 1, 3, 5, 10, 100], 
        labels=[1, 2, 3, 4, 5]
    )
    
    train_uniprots, test_uniprots = train_test_split(
        uniprot_df['uniprot_id'],
        test_size=test_ratio,
        random_state=random_state,
        stratify=uniprot_df['hotspot_bin']
    )
    
    train_df = merged_df[merged_df['uniprot_id'].isin(train_uniprots)].copy()
    test_df = merged_df[merged_df['uniprot_id'].isin(test_uniprots)].copy()
    
    train_uniprot_set = set(train_df['uniprot_id'])
    test_uniprot_set = set(test_df['uniprot_id'])
    overlap = train_uniprot_set & test_uniprot_set
    
    print(f"\n训练集: {len(train_df)} 条记录")
    print(f"  唯一 Uniprot ID: {len(train_uniprot_set)}")
    print(f"  总热点数: {train_df['n_hotspots'].sum()}")
    print(f"  平均每条记录热点数: {train_df['n_hotspots'].mean():.2f}")
    print(f"  数据来源:")
    print(f"    - data1: {len(train_df[train_df['source'] == 'data1'])}")
    print(f"    - data2: {len(train_df[train_df['source'] == 'data2'])}")
    
    print(f"\n测试集: {len(test_df)} 条记录")
    print(f"  唯一 Uniprot ID: {len(test_uniprot_set)}")
    print(f"  总热点数: {test_df['n_hotspots'].sum()}")
    print(f"  平均每条记录热点数: {test_df['n_hotspots'].mean():.2f}")
    print(f"  数据来源:")
    print(f"    - data1: {len(test_df[test_df['source'] == 'data1'])}")
    print(f"    - data2: {len(test_df[test_df['source'] == 'data2'])}")
    
    print(f"\n数据泄露检查:")
    print(f"  训练集和测试集重叠的 Uniprot ID 数: {len(overlap)}")
    if len(overlap) == 0:
        print(f"  ✅ 没有数据泄露！")
    else:
        print(f"  ❌ 存在数据泄露！")
    
    return train_df, test_df


def save_data(train_df, test_df):
    """保存数据"""
    print("\n" + "=" * 60)
    print("保存数据")
    print("=" * 60)
    
    train_df.to_excel(OUTPUT_TRAIN, index=False)
    test_df.to_excel(OUTPUT_TEST, index=False)
    
    print(f"\n训练集保存到: {OUTPUT_TRAIN}")
    print(f"测试集保存到: {OUTPUT_TEST}")


def main():
    print("\n" + "=" * 60)
    print("重新划分数据集")
    print("使用 PDB 编号而不是 Uniprot 编号")
    print("确保按 Uniprot ID 划分，避免数据泄露")
    print("=" * 60)
    
    merged_df = merge_datasets()
    
    train_df, test_df = split_data_by_uniprot(merged_df)
    
    save_data(train_df, test_df)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    
    print("""
下一步：
1. 删除旧的缓存文件：
   rm -rf data/features/*.pkl

2. 重新训练模型：
   python main.py --mode train --reprocess

3. 运行测试集评估：
   python fair_comparison.py
""")


if __name__ == '__main__':
    main()
