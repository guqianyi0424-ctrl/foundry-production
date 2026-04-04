"""检查测试数据处理问题"""
import pandas as pd
import os

test_df = pd.read_excel('merged_test.xlsx')

print('=' * 60)
print('测试数据分析')
print('=' * 60)

test_df['pdb_id_clean'] = test_df['pdb_id'].str.lower().str.split('-').str[0].str.strip()
test_df['chain_id'] = test_df['pdb_id'].str.split('-').str[1].fillna('A').str.strip()

print(f'\n测试数据行数: {len(test_df)}')
print(f'唯一 PDB ID 数: {test_df["pdb_id_clean"].nunique()}')
print(f'唯一 Uniprot ID 数: {test_df["uniprot_id"].nunique()}')

print(f'\nPDB ID 示例:')
print(sorted(test_df['pdb_id_clean'].unique())[:20])

pdb_dir = 'data/pdb_files'
if os.path.exists(pdb_dir):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.ent')]
    print(f'\nPDB 文件数: {len(pdb_files)}')
    
    needed_pdbs = set(test_df['pdb_id_clean'].unique())
    existing_pdbs = set(f.replace('pdb', '').replace('.ent', '') for f in pdb_files)
    
    missing_pdbs = needed_pdbs - existing_pdbs
    available_pdbs = needed_pdbs & existing_pdbs
    
    print(f'需要的 PDB 数: {len(needed_pdbs)}')
    print(f'已有的 PDB 数: {len(available_pdbs)}')
    print(f'缺失的 PDB 数: {len(missing_pdbs)}')
    
    if len(missing_pdbs) > 0:
        print(f'\n缺失的 PDB 示例:')
        print(list(missing_pdbs)[:10])
else:
    print(f'\nPDB 目录不存在: {pdb_dir}')

print('\n' + '=' * 60)
print('数据来源分布')
print('=' * 60)
print(test_df['source'].value_counts())
