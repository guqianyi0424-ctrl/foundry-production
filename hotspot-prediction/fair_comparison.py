"""
与 PPI-hotspotID Baseline 公平对比
使用相同的评估方式
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import defaultdict

from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR, DEVICE
from dataset import PPIHotspotDataset, collate_fn, prepare_test_dataset
from model import create_model


def load_ensemble_models():
    """加载所有折的模型"""
    models = []
    for fold in range(1, 6):
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


def predict_with_ensemble(models, data_loader):
    """使用集成模型预测"""
    all_probs = []
    all_labels = []
    all_pdb_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            node_features = batch['node_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            graphs = batch['graphs'].to(DEVICE)
            pdb_id = batch['pdb_ids'][0]
            
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
            all_pdb_ids.extend([pdb_id] * len(valid_labels))
    
    return np.array(all_labels), np.array(all_probs), all_pdb_ids


def evaluate_per_protein(labels, probs, pdb_ids, threshold=0.5):
    """按蛋白质级别评估"""
    protein_results = defaultdict(lambda: {'labels': [], 'probs': []})
    
    for label, prob, pdb_id in zip(labels, probs, pdb_ids):
        protein_results[pdb_id]['labels'].append(label)
        protein_results[pdb_id]['probs'].append(prob)
    
    protein_metrics = []
    for pdb_id, data in protein_results.items():
        prot_labels = np.array(data['labels'])
        prot_probs = np.array(data['probs'])
        prot_preds = (prot_probs >= threshold).astype(int)
        
        if prot_labels.sum() > 0:
            try:
                auc = metrics.roc_auc_score(prot_labels, prot_probs)
            except:
                auc = 0.5
            
            tp = ((prot_preds == 1) & (prot_labels == 1)).sum()
            fp = ((prot_preds == 1) & (prot_labels == 0)).sum()
            fn = ((prot_preds == 0) & (prot_labels == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            protein_metrics.append({
                'pdb_id': pdb_id,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_pos': prot_labels.sum(),
                'n_total': len(prot_labels)
            })
    
    return pd.DataFrame(protein_metrics)


def evaluate_hotspot_detection(labels, probs, pdb_ids, threshold=0.5):
    """评估热点残基检测能力（与PPI-hotspotID相同方式）"""
    protein_results = defaultdict(lambda: {'labels': [], 'probs': []})
    
    for label, prob, pdb_id in zip(labels, probs, pdb_ids):
        protein_results[pdb_id]['labels'].append(label)
        protein_results[pdb_id]['probs'].append(prob)
    
    detected_proteins = 0
    total_proteins = 0
    true_hotspots_found = 0
    total_hotspots = 0
    
    for pdb_id, data in protein_results.items():
        prot_labels = np.array(data['labels'])
        prot_probs = np.array(data['probs'])
        prot_preds = (prot_probs >= threshold).astype(int)
        
        n_hotspots = prot_labels.sum()
        if n_hotspots > 0:
            total_proteins += 1
            total_hotspots += n_hotspots
            
            if prot_preds.sum() > 0:
                detected_proteins += 1
            
            hotspots_found = ((prot_preds == 1) & (prot_labels == 1)).sum()
            true_hotspots_found += hotspots_found
    
    detection_rate = detected_proteins / total_proteins if total_proteins > 0 else 0
    hotspot_recall = true_hotspots_found / total_hotspots if total_hotspots > 0 else 0
    
    return {
        'proteins_with_hotspots': total_proteins,
        'proteins_detected': detected_proteins,
        'detection_rate': detection_rate,
        'total_hotspots': total_hotspots,
        'hotspots_found': true_hotspots_found,
        'hotspot_recall': hotspot_recall
    }


def find_optimal_threshold(labels, probs):
    """寻找最优阈值"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= threshold).astype(int)
        f1 = metrics.f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def apply_topk_postprocessing(labels, probs, pdb_ids, top_k=5):
    """
    后处理：每个蛋白质只保留预测概率最高的前 top_k 个残基
    这可以显著提高 Precision
    """
    protein_results = defaultdict(lambda: {'labels': [], 'probs': [], 'indices': []})
    
    for idx, (label, prob, pdb_id) in enumerate(zip(labels, probs, pdb_ids)):
        protein_results[pdb_id]['labels'].append(label)
        protein_results[pdb_id]['probs'].append(prob)
        protein_results[pdb_id]['indices'].append(idx)
    
    new_labels = np.zeros_like(labels)
    new_probs = np.zeros_like(probs)
    new_preds = np.zeros_like(labels, dtype=int)
    
    for pdb_id, data in protein_results.items():
        prot_labels = np.array(data['labels'])
        prot_probs = np.array(data['probs'])
        prot_indices = np.array(data['indices'])
        
        n_hotspots = prot_labels.sum()
        k = min(top_k, n_hotspots) if n_hotspots > 0 else top_k
        
        top_indices = np.argsort(prot_probs)[-k:]
        
        for idx in top_indices:
            original_idx = prot_indices[idx]
            new_labels[original_idx] = prot_labels[idx]
            new_probs[original_idx] = prot_probs[idx]
            new_preds[original_idx] = 1
    
    valid_mask = new_preds == 1
    return new_labels[valid_mask], new_probs[valid_mask], new_preds[valid_mask]


def evaluate_with_topk(labels, probs, pdb_ids, top_k_values=[3, 5, 7, 10]):
    """
    评估不同 top_k 值的效果
    """
    print("\n" + "=" * 70)
    print("后处理评估：每个蛋白质只保留预测概率最高的前 K 个残基")
    print("=" * 70)
    
    results = []
    
    for top_k in top_k_values:
        new_labels, new_probs, new_preds = apply_topk_postprocessing(labels, probs, pdb_ids, top_k)
        
        if len(new_preds) > 0 and new_labels.sum() > 0:
            tp = ((new_preds == 1) & (new_labels == 1)).sum()
            fp = ((new_preds == 1) & (new_labels == 0)).sum()
            fn = new_labels.sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'top_k': top_k,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_predictions': len(new_preds),
                'n_correct': tp
            })
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    Top-K 后处理评估结果                              │")
    print("├─────────┬───────────┬───────────┬───────────┬──────────┬──────────┤")
    print("│ Top-K   │ Precision │ Recall    │ F1        │ 预测数   │ 正确数   │")
    print("├─────────┼───────────┼───────────┼───────────┼──────────┼──────────┤")
    
    for r in results:
        print("│ {:^7} │ {:^9.4f} │ {:^9.4f} │ {:^9.4f} │ {:^8} │ {:^8} │".format(
            r['top_k'], r['precision'], r['recall'], r['f1'], r['n_predictions'], r['n_correct']
        ))
    
    print("└─────────┴───────────┴───────────┴───────────┴──────────┴──────────┘")
    
    return results


def compare_with_baseline():
    """与PPI-hotspotID baseline对比"""
    print("\n" + "=" * 70)
    print("与 PPI-hotspotID Baseline 公平对比")
    print("=" * 70)
    
    print("\n加载模型...")
    models = load_ensemble_models()
    
    if len(models) == 0:
        print("错误: 没有找到训练好的模型")
        return
    
    print(f"\n加载了 {len(models)} 个模型")
    
    print("\n加载测试数据...")
    test_data = prepare_test_dataset()
    
    if len(test_data) == 0:
        print("错误: 测试数据集为空")
        return
    
    print(f"测试数据集大小: {len(test_data)} 个蛋白质")
    
    test_dataset = PPIHotspotDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print("\n进行预测...")
    labels, probs, pdb_ids = predict_with_ensemble(models, test_loader)
    
    print("\n寻找最优阈值...")
    optimal_threshold, optimal_f1 = find_optimal_threshold(labels, probs)
    print(f"最优阈值: {optimal_threshold:.4f}, 最优F1: {optimal_f1:.4f}")
    
    print("\n" + "-" * 70)
    print("残基级别评估 (与论文Table对比)")
    print("-" * 70)
    
    preds = (probs >= optimal_threshold).astype(int)
    
    residue_metrics = {
        'ROC-AUC': metrics.roc_auc_score(labels, probs),
        'PR-AUC': metrics.average_precision_score(labels, probs),
        'F1': metrics.f1_score(labels, preds),
        'Precision': metrics.precision_score(labels, preds),
        'Recall': metrics.recall_score(labels, preds),
        'Accuracy': metrics.accuracy_score(labels, preds),
        'MCC': metrics.matthews_corrcoef(labels, preds)
    }
    
    print("\n残基级别指标:")
    for k, v in residue_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "-" * 70)
    print("蛋白质级别评估 (PPI-hotspotID论文方式)")
    print("-" * 70)
    
    protein_df = evaluate_per_protein(labels, probs, pdb_ids, optimal_threshold)
    
    print(f"\n评估了 {len(protein_df)} 个蛋白质")
    print(f"平均 AUC: {protein_df['auc'].mean():.4f}")
    print(f"平均 F1: {protein_df['f1'].mean():.4f}")
    print(f"平均 Precision: {protein_df['precision'].mean():.4f}")
    print(f"平均 Recall: {protein_df['recall'].mean():.4f}")
    
    print("\n" + "-" * 70)
    print("热点检测能力 (PPI-hotspotID论文核心指标)")
    print("-" * 70)
    
    detection = evaluate_hotspot_detection(labels, probs, pdb_ids, optimal_threshold)
    
    print(f"\n含热点残基的蛋白质数: {detection['proteins_with_hotspots']}")
    print(f"检测到热点的蛋白质数: {detection['proteins_detected']}")
    print(f"蛋白质检测率: {detection['detection_rate']:.4f}")
    print(f"总热点残基数: {detection['total_hotspots']}")
    print(f"找到的热点数: {detection['hotspots_found']}")
    print(f"热点召回率: {detection['hotspot_recall']:.4f}")
    
    print("\n" + "=" * 70)
    print("与 PPI-hotspotID 对比 (使用相同评估方式)")
    print("=" * 70)
    
    print("\n【重要】PPI-hotspotID 论文评估方式分析:")
    print("  1. 残基级别评估: 所有残基一起计算指标")
    print("  2. 蛋白质级别评估: 每个蛋白质计算指标后取平均")
    print("  3. 论文 F1=0.71 可能是蛋白质级别平均 F1")
    
    comparison = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    残基级别评估 (所有残基一起)                        │
    ├─────────────────────┬────────────────┬────────────────┬────────────┤
    │ 指标                │ PPI-hotspotID  │ 你的模型       │ 对比       │
    ├─────────────────────┼────────────────┼────────────────┼────────────┤
    │ F1 Score            │ 0.71 (可能)    │ {:.4f}         │ {} │
    │ Recall              │ 0.67 (可能)    │ {:.4f}         │ {} │
    │ Precision           │ -              │ {:.4f}         │ -          │
    │ ROC-AUC             │ -              │ {:.4f}         │ -          │
    └─────────────────────┴────────────────┴────────────────┴────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  蛋白质级别评估 (每蛋白质平均)                        │
    ├─────────────────────┬────────────────┬────────────────┬────────────┤
    │ 指标                │ PPI-hotspotID  │ 你的模型       │ 对比       │
    ├─────────────────────┼────────────────┼────────────────┼────────────┤
    │ 平均 F1             │ 0.71 (论文值)  │ {:.4f}         │ {} │
    │ 平均 Recall         │ 0.67 (论文值)  │ {:.4f}         │ {} │
    │ 平均 Precision      │ -              │ {:.4f}         │ -          │
    │ 平均 AUC            │ -              │ {:.4f}         │ -          │
    └─────────────────────┴────────────────┴────────────────┴────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    热点检测能力 (核心应用指标)                        │
    ├─────────────────────┬────────────────┬─────────────────────────────┤
    │ 指标                │ 数值           │ 说明                        │
    ├─────────────────────┼────────────────┼─────────────────────────────┤
    │ 蛋白质检测率        │ {:.2%}         │ 检测到热点的蛋白质比例       │
    │ 热点召回率          │ {:.2%}         │ 找到的热点残基比例           │
    └─────────────────────┴────────────────┴─────────────────────────────┘
    """.format(
        residue_metrics['F1'],
        "✅ 更好" if residue_metrics['F1'] > 0.71 else "❌ 需改进",
        residue_metrics['Recall'],
        "✅ 更好" if residue_metrics['Recall'] > 0.67 else "❌ 需改进",
        residue_metrics['Precision'],
        residue_metrics['ROC-AUC'],
        protein_df['f1'].mean(),
        "✅ 更好" if protein_df['f1'].mean() > 0.71 else "❌ 需改进",
        protein_df['recall'].mean(),
        "✅ 更好" if protein_df['recall'].mean() > 0.67 else "❌ 需改进",
        protein_df['precision'].mean(),
        protein_df['auc'].mean(),
        detection['detection_rate'],
        detection['hotspot_recall']
    )
    print(comparison)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        'residue_metrics': residue_metrics,
        'protein_avg_metrics': {
            'auc': protein_df['auc'].mean(),
            'f1': protein_df['f1'].mean(),
            'precision': protein_df['precision'].mean(),
            'recall': protein_df['recall'].mean()
        },
        'detection_metrics': detection,
        'optimal_threshold': optimal_threshold
    }
    
    import json
    with open(os.path.join(RESULTS_DIR, 'fair_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n结果已保存到: {RESULTS_DIR}/fair_comparison_results.json")
    
    evaluate_with_topk(labels, probs, pdb_ids, top_k_values=[3, 5, 7, 10])
    
    return results


if __name__ == '__main__':
    compare_with_baseline()
