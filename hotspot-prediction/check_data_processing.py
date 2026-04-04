"""
检查数据处理是否有问题
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, 'merged_train.xlsx')
TEST_FILE = os.path.join(BASE_DIR, 'merged_test.xlsx')

print("=" * 70)
print("检查数据处理")
print("=" * 70)

print("\n1. 检查训练数据...")
train_df = pd.read_excel(TRAIN_FILE)
print(f"   训练数据列名: {list(train_df.columns)}")
print(f"   训练数据行数: {len(train_df)}")
print(f"   前5行:")
print(train_df.head())

if 'hotspot_list' in train_df.columns:
    print(f"\n   热点列表示例:")
    for i, val in enumerate(train_df['hotspot_list'].head(10)):
        print(f"   {i}: {val} (类型: {type(val)})")
    
    total_hotspots = 0
    for val in train_df['hotspot_list']:
        if pd.notna(val) and val != '':
            hotspots = [int(x.strip()) for x in str(val).split(',') if x.strip().isdigit()]
            total_hotspots += len(hotspots)
    print(f"   总热点数: {total_hotspots}")

print("\n2. 检查测试数据...")
test_df = pd.read_excel(TEST_FILE)
print(f"   测试数据列名: {list(test_df.columns)}")
print(f"   测试数据行数: {len(test_df)}")
print(f"   前5行:")
print(test_df.head())

if 'hotspot_list' in test_df.columns:
    print(f"\n   热点列表示例:")
    for i, val in enumerate(test_df['hotspot_list'].head(10)):
        print(f"   {i}: {val} (类型: {type(val)})")
    
    total_hotspots = 0
    for val in test_df['hotspot_list']:
        if pd.notna(val) and val != '':
            hotspots = [int(x.strip()) for x in str(val).split(',') if x.strip().isdigit()]
            total_hotspots += len(hotspots)
    print(f"   总热点数: {total_hotspots}")

print("\n3. 检查数据格式一致性...")
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
print(f"   训练数据列: {train_cols}")
print(f"   测试数据列: {test_cols}")
print(f"   列名是否一致: {train_cols == test_cols}")

print("\n4. 检查PDB ID格式...")
if 'pdb_id' in train_df.columns:
    print(f"   训练数据PDB ID示例: {train_df['pdb_id'].head(10).tolist()}")
if 'pdb_id' in test_df.columns:
    print(f"   测试数据PDB ID示例: {test_df['pdb_id'].head(10).tolist()}")

print("\n5. 检查数据重叠...")
if 'pdb_id' in train_df.columns and 'pdb_id' in test_df.columns:
    train_pdbs = set(train_df['pdb_id'].str.lower().str.split('-').str[0])
    test_pdbs = set(test_df['pdb_id'].str.lower().str.split('-').str[0])
    overlap = train_pdbs & test_pdbs
    print(f"   训练数据PDB数: {len(train_pdbs)}")
    print(f"   测试数据PDB数: {len(test_pdbs)}")
    print(f"   重叠PDB数: {len(overlap)}")
    if overlap:
        print(f"   重叠PDB: {overlap}")

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)
