"""检查热点编号问题"""
import pandas as pd

print('=' * 60)
print('检查原始数据的热点编号')
print('=' * 60)

df1 = pd.read_excel('elife-96643-data1-v1.xlsx')
df2 = pd.read_excel('elife-96643-data2-v1.xlsx')

print('\ndata1 列名:')
print(list(df1.columns))

print('\ndata2 列名:')
print(list(df2.columns))

print('\ndata1 热点相关列:')
for col in df1.columns:
    if 'hot' in col.lower() or 'spot' in col.lower():
        print(f'  {col}: {df1[col].head(3).tolist()}')

print('\ndata2 热点相关列:')
for col in df2.columns:
    if 'hot' in col.lower() or 'spot' in col.lower():
        print(f'  {col}: {df2[col].head(3).tolist()}')

print('\ndata1 示例数据 (所有列):')
print(df1.head(3).to_string())

print('\ndata2 示例数据 (所有列):')
print(df2.head(3).to_string())
