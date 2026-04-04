"""检查测试数据的热点分布"""
import pandas as pd
import os
import pickle

print('=' * 60)
print('检查测试数据的热点分布')
print('=' * 60)

test_df = pd.read_excel('merged_test.xlsx')

print(f'\n测试数据行数: {len(test_df)}')
print(f'热点列表列: {test_df["hotspot_list"].head(10).tolist()}')

test_df['n_hotspots_list'] = test_df['hotspot_list'].apply(
    lambda x: len([h for h in str(x).split(',') if h.strip()]) if pd.notna(x) else 0
)

print(f'\n热点数量分布:')
print(test_df['n_hotspots_list'].value_counts().sort_index())

print(f'\n有热点的数据行数: {(test_df["n_hotspots_list"] > 0).sum()}')
print(f'无热点的数据行数: {(test_df["n_hotspots_list"] == 0).sum()}')

test_dataset_path = 'data/features/test_dataset.pkl'
if os.path.exists(test_dataset_path):
    with open(test_dataset_path, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f'\n已处理的测试数据:')
    print(f'  蛋白质数: {len(test_data)}')
    
    proteins_with_hotspots = 0
    total_hotspots = 0
    
    for item in test_data:
        n_hotspots = item['labels'].sum()
        if n_hotspots > 0:
            proteins_with_hotspots += 1
            total_hotspots += n_hotspots
    
    print(f'  有热点的蛋白质数: {proteins_with_hotspots}')
    print(f'  无热点的蛋白质数: {len(test_data) - proteins_with_hotspots}')
    print(f'  总热点数: {total_hotspots}')
else:
    print(f'\n测试数据集文件不存在: {test_dataset_path}')
