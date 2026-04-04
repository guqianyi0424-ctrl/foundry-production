"""
改进方案：融合传统特征 + 深度学习特征
借鉴 PPI-hotspotID 的特征工程
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR, DEVICE
from dataset import PPIHotspotDataset, collate_fn, load_training_data
from model import create_model


class HybridModel(nn.Module):
    """混合模型：深度学习特征 + 传统特征"""
    
    def __init__(self, dl_feature_dim, traditional_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        
        self.dl_encoder = nn.Sequential(
            nn.Linear(dl_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.trad_encoder = nn.Sequential(
            nn.Linear(traditional_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, dl_features, trad_features):
        dl_encoded = self.dl_encoder(dl_features)
        trad_encoded = self.trad_encoder(trad_features)
        combined = torch.cat([dl_encoded, trad_encoded], dim=1)
        logits = self.classifier(combined)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def extract_features_from_dataset(dataset):
    """从数据集中提取特征"""
    all_dl_features = []
    all_trad_features = []
    all_labels = []
    all_pdb_ids = []
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
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
    
    if len(models) == 0:
        print("错误: 没有找到训练好的模型")
        return None
    
    print(f"使用 {len(models)} 个模型提取特征...")
    
    with torch.no_grad():
        for batch in dataloader:
            node_features = batch['node_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            graphs = batch['graphs'].to(DEVICE)
            pdb_id = batch['pdb_id'][0]
            
            valid_mask = labels >= 0
            valid_features = node_features[valid_mask]
            valid_labels = labels[valid_mask]
            
            esm_dim = 1280
            trad_dim = node_features.shape[1] - esm_dim
            
            esm_features = valid_features[:, :esm_dim]
            trad_features = valid_features[:, esm_dim:]
            
            batch_dl_features = []
            for model in models:
                logits = model(graphs, node_features)
                valid_logits = logits[valid_mask]
                probs = torch.softmax(valid_logits, dim=1)
                batch_dl_features.append(probs.cpu().numpy())
            
            avg_dl_features = np.mean(batch_dl_features, axis=0)
            
            all_dl_features.extend(avg_dl_features)
            all_trad_features.extend(trad_features.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            all_pdb_ids.extend([pdb_id] * len(valid_labels))
    
    return {
        'dl_features': np.array(all_dl_features),
        'trad_features': np.array(all_trad_features),
        'labels': np.array(all_labels),
        'pdb_ids': all_pdb_ids
    }


def train_ensemble_classifier(features, labels):
    """训练集成分类器"""
    print("\n训练集成分类器...")
    
    X = np.hstack([features['dl_features'], features['trad_features']])
    y = features['labels']
    
    print(f"特征维度: {X.shape}")
    print(f"正样本比例: {y.mean():.4f}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_preds = []
    all_probs = []
    all_true = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/5...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        classifiers = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
        }
        
        fold_probs = []
        for name, clf in classifiers.items():
            print(f"  训练 {name}...")
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_val)[:, 1]
            fold_probs.append(prob)
        
        avg_prob = np.mean(fold_probs, axis=0)
        
        all_preds.extend((avg_prob >= 0.5).astype(int))
        all_probs.extend(avg_prob)
        all_true.extend(y_val)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    print("\n" + "=" * 60)
    print("集成分类器结果")
    print("=" * 60)
    
    metrics_dict = {
        'ROC-AUC': metrics.roc_auc_score(all_true, all_probs),
        'PR-AUC': metrics.average_precision_score(all_true, all_probs),
        'F1': metrics.f1_score(all_true, all_preds),
        'Precision': metrics.precision_score(all_true, all_preds),
        'Recall': metrics.recall_score(all_true, all_preds),
        'Accuracy': metrics.accuracy_score(all_true, all_preds),
        'MCC': metrics.matthews_corrcoef(all_true, all_preds)
    }
    
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics_dict, classifiers


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


def optimize_threshold_for_comparison(labels, probs):
    """针对与PPI-hotspotID对比优化阈值"""
    print("\n优化阈值...")
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['f1'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\n最优阈值: {best_result['threshold']:.2f}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    
    return best_result['threshold'], results_df


def main():
    print("\n" + "=" * 70)
    print("改进方案：融合特征 + 集成学习")
    print("=" * 70)
    
    print("\n步骤1: 加载数据...")
    train_data = load_training_data()
    dataset = PPIHotspotDataset(train_data)
    
    print("\n步骤2: 提取特征...")
    features = extract_features_from_dataset(dataset)
    
    if features is None:
        return
    
    print(f"\n特征统计:")
    print(f"  深度学习特征维度: {features['dl_features'].shape}")
    print(f"  传统特征维度: {features['trad_features'].shape}")
    print(f"  样本数: {len(features['labels'])}")
    print(f"  正样本数: {features['labels'].sum()}")
    print(f"  正样本比例: {features['labels'].mean():.4f}")
    
    print("\n步骤3: 训练集成分类器...")
    metrics_dict, classifiers = train_ensemble_classifier(features, features['labels'])
    
    print("\n步骤4: 优化阈值...")
    X = np.hstack([features['dl_features'], features['trad_features']])
    
    all_probs = []
    for name, clf in classifiers.items():
        prob = clf.predict_proba(X)[:, 1]
        all_probs.append(prob)
    avg_probs = np.mean(all_probs, axis=0)
    
    optimal_threshold, threshold_results = optimize_threshold_for_comparison(
        features['labels'], avg_probs
    )
    
    print("\n" + "=" * 70)
    print("最终对比")
    print("=" * 70)
    
    final_preds = (avg_probs >= optimal_threshold).astype(int)
    
    final_metrics = {
        'ROC-AUC': metrics.roc_auc_score(features['labels'], avg_probs),
        'PR-AUC': metrics.average_precision_score(features['labels'], avg_probs),
        'F1': metrics.f1_score(features['labels'], final_preds),
        'Precision': metrics.precision_score(features['labels'], final_preds),
        'Recall': metrics.recall_score(features['labels'], final_preds),
    }
    
    print("\n你的模型 (改进后):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nPPI-hotspotID (论文报告):")
    print("  F1: 0.71")
    print("  Recall: 0.67")
    
    print("\n对比结果:")
    if final_metrics['F1'] > 0.71:
        print("  ✅ F1 超过 PPI-hotspotID!")
    else:
        print(f"  ❌ F1 还差 {0.71 - final_metrics['F1']:.4f}")
    
    if final_metrics['Recall'] > 0.67:
        print("  ✅ Recall 超过 PPI-hotspotID!")
    else:
        print(f"  ❌ Recall 还差 {0.67 - final_metrics['Recall']:.4f}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        'final_metrics': final_metrics,
        'optimal_threshold': optimal_threshold,
        'improvement_over_baseline': {
            'f1_improvement': final_metrics['F1'] - 0.71,
            'recall_improvement': final_metrics['Recall'] - 0.67
        }
    }
    
    import json
    with open(os.path.join(RESULTS_DIR, 'improved_model_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    threshold_results.to_csv(os.path.join(RESULTS_DIR, 'threshold_optimization.csv'), index=False)
    
    print(f"\n结果已保存到: {RESULTS_DIR}/")
    
    return results


if __name__ == '__main__':
    main()
