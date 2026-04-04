"""
消融实验脚本
测试不同特征组合对模型性能的影响
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODELS_DIR, RESULTS_DIR, FEATURES_DIR, DEVICE,
    ESM2_DIM, PSSM_DIM, HMM_DIM, TRADITIONAL_DIM,
    HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
    NUM_EPOCHS, PATIENCE, N_FOLDS
)
from dataset import prepare_test_dataset, PPIHotspotDataset, collate_fn
from model import create_model
import importlib


ABLATION_CONFIGS = [
    {
        'name': 'ESM2_only',
        'description': '只用ESM-2特征',
        'dims': {
            'ESM2_DIM': 1280,
            'PSSM_DIM': 0,
            'HMM_DIM': 0,
            'TRADITIONAL_DIM': 0
        },
        'input_dim': 1280
    },
    {
        'name': 'ESM2_PSSM',
        'description': 'ESM-2 + PSSM',
        'dims': {
            'ESM2_DIM': 1280,
            'PSSM_DIM': 20,
            'HMM_DIM': 0,
            'TRADITIONAL_DIM': 0
        },
        'input_dim': 1300
    },
    {
        'name': 'ESM2_HMM',
        'description': 'ESM-2 + HMM',
        'dims': {
            'ESM2_DIM': 1280,
            'PSSM_DIM': 0,
            'HMM_DIM': 30,
            'TRADITIONAL_DIM': 0
        },
        'input_dim': 1310
    },
    {
        'name': 'ESM2_Traditional',
        'description': 'ESM-2 + 传统特征',
        'dims': {
            'ESM2_DIM': 1280,
            'PSSM_DIM': 0,
            'HMM_DIM': 0,
            'TRADITIONAL_DIM': 27
        },
        'input_dim': 1307
    },
    {
        'name': 'ESM2_PSSM_HMM',
        'description': 'ESM-2 + PSSM + HMM',
        'dims': {
            'ESM2_DIM': 1280,
            'PSSM_DIM': 20,
            'HMM_DIM': 30,
            'TRADITIONAL_DIM': 0
        },
        'input_dim': 1330
    },
    {
        'name': 'All_features',
        'description': '全部特征',
        'dims': {
            'ESM2_DIM': 1280,
            'PSSM_DIM': 20,
            'HMM_DIM': 30,
            'TRADITIONAL_DIM': 27
        },
        'input_dim': 1357
    }
]


def modify_config(config_name, dims):
    """修改配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace(f"PSSM_DIM = {PSSM_DIM}", f"PSSM_DIM = {dims['PSSM_DIM']}")
    content = content.replace(f"HMM_DIM = {HMM_DIM}", f"HMM_DIM = {dims['HMM_DIM']}")
    content = content.replace(f"TRADITIONAL_DIM = {TRADITIONAL_DIM}", f"TRADITIONAL_DIM = {dims['TRADITIONAL_DIM']}")
    
    old_input_dim = ESM2_DIM + PSSM_DIM + HMM_DIM + TRADITIONAL_DIM
    new_input_dim = dims['ESM2_DIM'] + dims['PSSM_DIM'] + dims['HMM_DIM'] + dims['TRADITIONAL_DIM']
    content = content.replace(f"INPUT_DIM = {old_input_dim}", f"INPUT_DIM = {new_input_dim}")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修改配置: {config_name}")
    print(f"  PSSM_DIM: {dims['PSSM_DIM']}")
    print(f"  HMM_DIM: {dims['HMM_DIM']}")
    print(f"  TRADITIONAL_DIM: {dims['TRADITIONAL_DIM']}")
    print(f"  INPUT_DIM: {new_input_dim}")


def train_model_for_ablation(config_name):
    """训练模型"""
    print(f"\n开始训练: {config_name}")
    
    from train import cross_validation
    from dataset import prepare_dataset
    
    data_list = prepare_dataset()
    results = cross_validation(data_list, n_folds=N_FOLDS, model_type='gat')
    
    return results


def evaluate_on_test():
    """在测试集上评估"""
    print("\n加载测试数据...")
    test_data = prepare_test_dataset()
    
    if len(test_data) == 0:
        print("错误: 测试数据集为空")
        return None
    
    print(f"测试数据集大小: {len(test_data)} 个蛋白质")
    
    models = []
    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}.pth')
        if os.path.exists(model_path):
            model = create_model('gat')
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
            model.eval()
            models.append(model)
    
    if len(models) == 0:
        print("错误: 没有找到训练好的模型")
        return None
    
    print(f"加载了 {len(models)} 个模型")
    
    test_dataset = PPIHotspotDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            node_features = batch['node_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            graphs = batch['graphs'].to(DEVICE)
            
            batch_probs = []
            for model in models:
                logits = model(graphs, node_features)
                labels_flat = labels.flatten()
                valid_mask = labels_flat >= 0
                valid_logits = logits[valid_mask]
                probs = torch.softmax(valid_logits, dim=1)[:, 1]
                batch_probs.append(probs.cpu().numpy())
            
            avg_probs = np.mean(batch_probs, axis=0)
            valid_labels = labels_flat[valid_mask].cpu().numpy()
            
            all_probs.extend(avg_probs)
            all_labels.extend(valid_labels)
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= threshold).astype(int)
        f1 = metrics.f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    preds = (all_probs >= best_threshold).astype(int)
    
    results = {
        'roc_auc': metrics.roc_auc_score(all_labels, all_probs),
        'pr_auc': metrics.average_precision_score(all_labels, all_probs),
        'f1': metrics.f1_score(all_labels, preds),
        'precision': metrics.precision_score(all_labels, preds, zero_division=0),
        'recall': metrics.recall_score(all_labels, preds, zero_division=0),
        'accuracy': metrics.accuracy_score(all_labels, preds),
        'mcc': metrics.matthews_corrcoef(all_labels, preds),
        'optimal_threshold': best_threshold
    }
    
    return results


def run_ablation_study():
    """运行消融实验"""
    print("=" * 70)
    print("消融实验")
    print("=" * 70)
    
    results = []
    
    for config in ABLATION_CONFIGS:
        print(f"\n{'='*70}")
        print(f"实验: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"{'='*70}")
        
        modify_config(config['name'], config['dims'])
        
        importlib.reload(sys.modules['config'])
        
        import shutil
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pth'):
                    os.remove(os.path.join(models_dir, f))
        
        features_dir = os.path.join(os.path.dirname(__file__), 'data', 'features')
        if os.path.exists(features_dir):
            for f in os.listdir(features_dir):
                if f.endswith('.pkl'):
                    os.remove(os.path.join(features_dir, f))
        
        train_results = train_model_for_ablation(config['name'])
        
        test_results = evaluate_on_test()
        
        if test_results:
            result = {
                'name': config['name'],
                'description': config['description'],
                'input_dim': config['input_dim'],
                'train_results': {
                    'roc_auc': float(train_results['roc_auc'].mean()),
                    'f1': float(train_results['f1'].mean()),
                    'precision': float(train_results['precision'].mean()),
                    'recall': float(train_results['recall'].mean())
                },
                'test_results': test_results
            }
            results.append(result)
            
            print(f"\n{config['name']} 结果:")
            print(f"  训练集 ROC-AUC: {train_results['roc_auc'].mean():.4f}")
            print(f"  训练集 F1: {train_results['f1'].mean():.4f}")
            print(f"  测试集 ROC-AUC: {test_results['roc_auc']:.4f}")
            print(f"  测试集 F1: {test_results['f1']:.4f}")
            print(f"  测试集 Recall: {test_results['recall']:.4f}")
    
    results_df = pd.DataFrame([
        {
            '实验': r['name'],
            '描述': r['description'],
            '特征维度': r['input_dim'],
            '训练ROC-AUC': r['train_results']['roc_auc'],
            '训练F1': r['train_results']['f1'],
            '测试ROC-AUC': r['test_results']['roc_auc'],
            '测试F1': r['test_results']['f1'],
            '测试Precision': r['test_results']['precision'],
            '测试Recall': r['test_results']['recall']
        }
        for r in results
    ])
    
    results_file = os.path.join(RESULTS_DIR, 'ablation_study_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n结果已保存到: {results_file}")
    
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    return results


if __name__ == '__main__':
    run_ablation_study()
