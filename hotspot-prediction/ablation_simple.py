"""
简化版消融实验脚本
手动运行不同特征组合的实验
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, RESULTS_DIR, DEVICE, N_FOLDS
from dataset import prepare_test_dataset, PPIHotspotDataset, collate_fn
from model import create_model


def load_models():
    """加载所有训练好的模型"""
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
            print(f"加载模型: {model_path}")
    return models


def evaluate_on_test(models):
    """在测试集上评估"""
    print("\n加载测试数据...")
    test_data = prepare_test_dataset()
    
    if len(test_data) == 0:
        print("错误: 测试数据集为空")
        return None
    
    print(f"测试数据集大小: {len(test_data)} 个蛋白质")
    
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


def record_result(experiment_name, description, input_dim, results):
    """记录实验结果"""
    result_file = os.path.join(RESULTS_DIR, 'ablation_results.csv')
    
    new_row = {
        'experiment': experiment_name,
        'description': description,
        'input_dim': input_dim,
        'roc_auc': results['roc_auc'],
        'pr_auc': results['pr_auc'],
        'f1': results['f1'],
        'precision': results['precision'],
        'recall': results['recall'],
        'accuracy': results['accuracy'],
        'mcc': results['mcc'],
        'optimal_threshold': results['optimal_threshold'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    return df


def main():
    print("=" * 70)
    print("消融实验 - 测试集评估")
    print("=" * 70)
    
    from config import ESM2_DIM, PSSM_DIM, HMM_DIM, TRADITIONAL_DIM, INPUT_DIM
    
    print(f"\n当前特征配置:")
    print(f"  ESM2_DIM: {ESM2_DIM}")
    print(f"  PSSM_DIM: {PSSM_DIM}")
    print(f"  HMM_DIM: {HMM_DIM}")
    print(f"  TRADITIONAL_DIM: {TRADITIONAL_DIM}")
    print(f"  INPUT_DIM: {INPUT_DIM}")
    
    print("\n加载模型...")
    models = load_models()
    
    if len(models) == 0:
        print("错误: 没有找到训练好的模型")
        return
    
    print(f"加载了 {len(models)} 个模型")
    
    results = evaluate_on_test(models)
    
    if results:
        print("\n" + "=" * 70)
        print("测试集评估结果")
        print("=" * 70)
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        print(f"PR-AUC: {results['pr_auc']:.4f}")
        print(f"F1: {results['f1']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"MCC: {results['mcc']:.4f}")
        print(f"最优阈值: {results['optimal_threshold']:.4f}")
        
        experiment_name = input("\n请输入实验名称 (如 ESM2_only): ")
        description = input("请输入实验描述 (如 只用ESM-2特征): ")
        
        df = record_result(experiment_name, description, INPUT_DIM, results)
        
        print("\n当前所有实验结果:")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
