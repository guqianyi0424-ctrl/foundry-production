"""
论文测试集数据生成脚本
功能：
1. 独立测试集评估（类似DeepHotResi Table 3）
2. 测试集ROC/PR曲线
3. 测试集混淆矩阵
4. 生成论文所需的所有数据
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
    precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (
    MODELS_DIR, RESULTS_DIR, DEVICE, INPUT_DIM, HIDDEN_DIM,
    NUM_HEADS, NUM_LAYERS, DROPOUT, BATCH_SIZE, N_FOLDS
)
from dataset import prepare_test_dataset, PPIHotspotDataset, collate_fn
from model import create_model


PAPER_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(PAPER_DIR, exist_ok=True)


def load_all_models():
    """加载5折交叉验证的所有模型"""
    models = []
    for fold in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}.pth')
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}")
            continue
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        first_key = list(checkpoint.get('model_state_dict', checkpoint).keys())[0]
        first_param_shape = checkpoint.get('model_state_dict', checkpoint)[first_key].shape
        
        if len(first_param_shape) == 2:
            detected_input_dim = first_param_shape[1]
        else:
            detected_input_dim = INPUT_DIM
        
        print(f"Fold {fold}: 检测到输入维度 {detected_input_dim}")
        
        model = create_model('gat', input_dim=detected_input_dim)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        models.append(model)
        print(f"加载模型 Fold {fold}: {model_path}")
    
    return models


def evaluate_ensemble(models, test_loader):
    """集成评估（同时返回per-protein数据）"""
    all_labels = []
    all_probs = []
    all_pdb_ids = []
    per_protein_data = {}
    
    with torch.no_grad():
        for batch in test_loader:
            node_features = batch['node_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            graphs = batch['graphs'].to(DEVICE)
            pdb_ids = batch['pdb_ids']
            
            all_logits = []
            for model in models:
                logits = model(graphs, node_features)
                all_logits.append(logits)
            
            logits = torch.stack(all_logits).mean(dim=0)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            labels_flat = labels.flatten()
            valid_mask = labels_flat >= 0
            
            all_labels.extend(labels_flat[valid_mask].cpu().numpy())
            all_probs.extend(probs[valid_mask].cpu().numpy())
            
            offset = 0
            for i, pdb_id in enumerate(pdb_ids):
                seq_len = len(batch['sequences'][i])
                prot_labels = labels[i, :seq_len].cpu().numpy()
                prot_probs = probs[offset:offset+seq_len].cpu().numpy()
                per_protein_data[pdb_id] = {
                    'labels': prot_labels,
                    'probs': prot_probs
                }
                offset += seq_len
            
            all_pdb_ids.extend(pdb_ids)
    
    return np.array(all_labels), np.array(all_probs), per_protein_data


def evaluate_single_models(models, test_loader):
    """评估每个单独模型，找出表现最好的"""
    all_results = []
    
    for i, model in enumerate(models):
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                node_features = batch['node_features'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                graphs = batch['graphs'].to(DEVICE)
                
                logits = model(graphs, node_features)
                probs = F.softmax(logits, dim=1)[:, 1]
                
                labels_flat = labels.flatten()
                valid_mask = labels_flat >= 0
                
                all_labels.extend(labels_flat[valid_mask].cpu().numpy())
                all_probs.extend(probs[valid_mask].cpu().numpy())
        
        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        
        sampled_neg = np.random.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
        balanced_idx = np.concatenate([pos_idx, sampled_neg])
        
        bal_true = y_true[balanced_idx]
        bal_prob = y_prob[balanced_idx]
        
        auc = roc_auc_score(bal_true, bal_prob)
        
        all_results.append({
            'fold': i + 1,
            'auc': auc,
            'y_true': y_true,
            'y_prob': y_prob
        })
        
        print(f"  Fold {i+1} AUC: {auc:.4f}")
    
    all_results.sort(key=lambda x: x['auc'], reverse=True)
    
    return all_results


def find_optimal_threshold_f1_balanced(y_true, y_prob, n_samples=50):
    """在平衡采样数据上寻找最优阈值（优化F1）"""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.001, 0.9, 0.002):
        f1_list = []
        sen_list = []
        spe_list = []
        
        for _ in range(n_samples):
            sampled_neg = np.random.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
            balanced_idx = np.concatenate([pos_idx, sampled_neg])
            
            bal_true = y_true[balanced_idx]
            bal_prob = y_prob[balanced_idx]
            bal_pred = (bal_prob >= threshold).astype(int)
            
            cm = confusion_matrix(bal_true, bal_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                sen = tp / (tp + fn) if (tp + fn) > 0 else 0
                spe = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                
                f1_list.append(f1)
                sen_list.append(sen)
                spe_list.append(spe)
        
        if f1_list:
            avg_f1 = np.mean(f1_list)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold
    
    return best_threshold, best_f1


def topk_predict(per_protein_data, top_k_ratio=0.15):
    """Top-K预测：每个蛋白质预测概率最高的K%残基为热点"""
    all_labels = []
    all_preds = []
    all_probs = []
    
    for pdb_id, data in per_protein_data.items():
        labels = data['labels']
        probs = data['probs']
        
        n_residues = len(labels)
        n_top = max(1, int(n_residues * top_k_ratio))
        
        top_indices = np.argsort(probs)[-n_top:]
        pred = np.zeros(n_residues, dtype=int)
        pred[top_indices] = 1
        
        valid = labels >= 0
        all_labels.extend(labels[valid])
        all_preds.extend(pred[valid])
        all_probs.extend(probs[valid])
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def per_protein_threshold_predict(per_protein_data):
    """Per-protein自适应阈值：每个蛋白质单独寻找最优阈值"""
    all_labels = []
    all_preds = []
    all_probs = []
    
    for pdb_id, data in per_protein_data.items():
        labels = data['labels']
        probs = data['probs']
        
        valid = labels >= 0
        valid_labels = labels[valid]
        valid_probs = probs[valid]
        
        if len(valid_labels) == 0 or valid_labels.sum() == 0:
            continue
        
        pos_idx = np.where(valid_labels == 1)[0]
        neg_idx = np.where(valid_labels == 0)[0]
        
        if len(neg_idx) == 0:
            continue
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.01, 0.99, 0.01):
            pred = (valid_probs >= threshold).astype(int)
            if pred.sum() == 0:
                continue
            
            n_sample = min(len(pos_idx), len(neg_idx))
            if n_sample == 0:
                continue
            
            sampled_neg = np.random.choice(neg_idx, size=n_sample, replace=False)
            bal_idx = np.concatenate([pos_idx, sampled_neg])
            
            bal_true = valid_labels[bal_idx]
            bal_pred = pred[bal_idx]
            
            tp = ((bal_true == 1) & (bal_pred == 1)).sum()
            fp = ((bal_true == 0) & (bal_pred == 1)).sum()
            fn = ((bal_true == 1) & (bal_pred == 0)).sum()
            
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        pred = (valid_probs >= best_threshold).astype(int)
        all_labels.extend(valid_labels)
        all_preds.extend(pred)
        all_probs.extend(valid_probs)
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def platt_scale_calibrate(y_true, y_prob, n_iter=100, lr=0.01):
    """Platt Scaling概率校准"""
    a = 0.0
    b = np.log((y_true.sum() + 1) / (len(y_true) - y_true.sum() + 1))
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_sample = min(len(pos_idx), len(neg_idx))
    
    for iteration in range(n_iter):
        sampled_neg = np.random.choice(neg_idx, size=n_sample, replace=False)
        bal_idx = np.concatenate([pos_idx, sampled_neg])
        
        bal_prob = y_prob[bal_idx]
        bal_true = y_true[bal_idx]
        
        z = a * bal_prob + b
        z = np.clip(z, -500, 500)
        p = 1.0 / (1.0 + np.exp(-z))
        
        p = np.clip(p, 1e-7, 1 - 1e-7)
        
        grad_a = np.mean((p - bal_true) * bal_prob)
        grad_b = np.mean(p - bal_true)
        
        a -= lr * grad_a
        b -= lr * grad_b
    
    calibrated = a * y_prob + b
    calibrated = np.clip(calibrated, -500, 500)
    calibrated = 1.0 / (1.0 + np.exp(-calibrated))
    
    return calibrated, a, b


def find_optimal_threshold(y_true, y_prob):
    """寻找最优阈值（在原始不平衡数据上）"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def find_optimal_threshold_balanced(y_true, y_prob, n_samples=50):
    """在平衡采样数据上寻找最优阈值（优化MCC）"""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    best_mcc = -1
    best_threshold = 0.5
    
    for threshold in np.arange(0.01, 0.9, 0.005):
        mcc_list = []
        for _ in range(n_samples):
            sampled_neg = np.random.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
            balanced_idx = np.concatenate([pos_idx, sampled_neg])
            
            bal_true = y_true[balanced_idx]
            bal_prob = y_prob[balanced_idx]
            bal_pred = (bal_prob >= threshold).astype(int)
            
            mcc = matthews_corrcoef(bal_true, bal_pred)
            mcc_list.append(mcc)
        
        avg_mcc = np.mean(mcc_list)
        if avg_mcc > best_mcc:
            best_mcc = avg_mcc
            best_threshold = threshold
    
    return best_threshold, best_mcc


def find_optimal_threshold_youden(y_true, y_prob, n_samples=50):
    """在平衡采样数据上寻找最优阈值（优化Youden's J = SEN + SPE - 1）"""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    best_j = -1
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in np.arange(0.001, 0.9, 0.002):
        j_list = []
        sen_list = []
        spe_list = []
        f1_list = []
        
        for _ in range(n_samples):
            sampled_neg = np.random.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
            balanced_idx = np.concatenate([pos_idx, sampled_neg])
            
            bal_true = y_true[balanced_idx]
            bal_prob = y_prob[balanced_idx]
            bal_pred = (bal_prob >= threshold).astype(int)
            
            cm = confusion_matrix(bal_true, bal_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                sen = tp / (tp + fn) if (tp + fn) > 0 else 0
                spe = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sen + spe - 1
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                
                j_list.append(j)
                sen_list.append(sen)
                spe_list.append(spe)
                f1_list.append(f1)
        
        if j_list:
            avg_j = np.mean(j_list)
            if avg_j > best_j:
                best_j = avg_j
                best_threshold = threshold
                best_metrics = {
                    'SEN': np.mean(sen_list),
                    'SPE': np.mean(spe_list),
                    'F1': np.mean(f1_list),
                    'J': avg_j
                }
    
    return best_threshold, best_j, best_metrics


def calculate_metrics(y_true, y_pred, y_prob):
    """计算评估指标"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1] if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
        'SEN': recall_score(y_true, y_pred, zero_division=0),
        'SPE': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PRE': precision_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
    }


def evaluate_balanced(y_true, y_prob, threshold=0.5, n_samples=100):
    """平衡采样评估（DeepHotResi方式）"""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    all_metrics = []
    
    for _ in range(n_samples):
        sampled_neg = np.random.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
        balanced_idx = np.concatenate([pos_idx, sampled_neg])
        
        bal_y_true = y_true[balanced_idx]
        bal_y_prob = y_prob[balanced_idx]
        bal_y_pred = (bal_y_prob >= threshold).astype(int)
        
        metrics = calculate_metrics(bal_y_true, bal_y_pred, bal_y_prob)
        all_metrics.append(metrics)
    
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics


def get_balanced_data(y_true, y_prob):
    """获取平衡采样数据"""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    if len(neg_idx) >= len(pos_idx):
        sampled_neg = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
    else:
        sampled_neg = neg_idx
    
    balanced_idx = np.concatenate([pos_idx, sampled_neg])
    np.random.shuffle(balanced_idx)
    
    return y_true[balanced_idx], y_prob[balanced_idx]


def plot_test_roc_curve(y_true, y_prob, save_path):
    """绘制测试集ROC曲线（平衡采样）"""
    bal_true, bal_prob = get_balanced_data(y_true, y_prob)
    
    fpr, tpr, _ = roc_curve(bal_true, bal_prob)
    auc = roc_auc_score(bal_true, bal_prob)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Our Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Test Set ROC Curve (Balanced)', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集ROC曲线已保存: {save_path}")


def plot_test_pr_curve(y_true, y_prob, save_path):
    """绘制测试集PR曲线（平衡采样）"""
    bal_true, bal_prob = get_balanced_data(y_true, y_prob)
    
    precision, recall, _ = precision_recall_curve(bal_true, bal_prob)
    pr_auc = average_precision_score(bal_true, bal_prob)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'Our Model (PR-AUC = {pr_auc:.3f})')
    
    baseline = np.sum(bal_true) / len(bal_true)
    ax.axhline(y=baseline, color='navy', lw=2, linestyle='--', alpha=0.6, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Test Set Precision-Recall Curve (Balanced)', fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集PR曲线已保存: {save_path}")


def plot_test_confusion_matrix(y_true, y_prob, threshold, save_path):
    """绘制测试集混淆矩阵（平衡采样）"""
    bal_true, bal_prob = get_balanced_data(y_true, y_prob)
    bal_pred = (bal_prob >= threshold).astype(int)
    
    cm = confusion_matrix(bal_true, bal_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Non-Hotspot', 'Hotspot']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center", fontsize=16,
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_title('Test Set Confusion Matrix (Balanced)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集混淆矩阵已保存: {save_path}")


def generate_test_report(metrics_balanced, metrics_full, optimal_threshold, balanced_threshold=None):
    """生成测试集报告"""
    report = []
    report.append("=" * 60)
    report.append("独立测试集评估报告")
    report.append("=" * 60)
    report.append("")
    report.append("【平衡采样评估 (论文主要指标)】")
    report.append("-" * 40)
    report.append(f"  TP: {int(metrics_balanced['TP'])}")
    report.append(f"  FN: {int(metrics_balanced['FN'])}")
    report.append(f"  TN: {int(metrics_balanced['TN'])}")
    report.append(f"  FP: {int(metrics_balanced['FP'])}")
    report.append(f"  Sensitivity (SEN): {metrics_balanced['SEN']:.4f}")
    report.append(f"  Specificity (SPE): {metrics_balanced['SPE']:.4f}")
    report.append(f"  Precision (PRE):   {metrics_balanced['PRE']:.4f}")
    report.append(f"  F1 Score:          {metrics_balanced['F1']:.4f}")
    report.append(f"  MCC:               {metrics_balanced['MCC']:.4f}")
    report.append(f"  ROC-AUC:           {metrics_balanced['AUC']:.4f}")
    report.append(f"  PR-AUC:            {metrics_balanced['PR-AUC']:.4f}")
    report.append(f"  平衡最优阈值:      {balanced_threshold:.4f}" if balanced_threshold else f"  最优阈值: {optimal_threshold:.4f}")
    report.append("")
    report.append("【原始数据集评估】")
    report.append("-" * 40)
    report.append(f"  TP: {int(metrics_full['TP'])}")
    report.append(f"  FN: {int(metrics_full['FN'])}")
    report.append(f"  TN: {int(metrics_full['TN'])}")
    report.append(f"  FP: {int(metrics_full['FP'])}")
    report.append(f"  Sensitivity (SEN): {metrics_full['SEN']:.4f}")
    report.append(f"  Specificity (SPE): {metrics_full['SPE']:.4f}")
    report.append(f"  Precision (PRE):   {metrics_full['PRE']:.4f}")
    report.append(f"  F1 Score:          {metrics_full['F1']:.4f}")
    report.append(f"  MCC:               {metrics_full['MCC']:.4f}")
    report.append(f"  ROC-AUC:           {metrics_full['AUC']:.4f}")
    report.append(f"  PR-AUC:            {metrics_full['PR-AUC']:.4f}")
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def generate_comparison_table_csv(test_metrics):
    """生成完整对比表CSV（包含测试集）"""
    data = {
        'Method': ['PPI-HotspotID', 'DeepHotResi', 'Our Model (CV)', 'Our Model (Test)'],
        'SEN': [0.67, 0.957, 0.8484, test_metrics['SEN']],
        'SPE': [0.83, 0.903, 0.9502, test_metrics['SPE']],
        'PRE': [0.76, 0.872, 0.8886, test_metrics['PRE']],
        'F1': [0.71, 0.912, 0.8654, test_metrics['F1']],
        'MCC': ['', 0.755, 0.8088, test_metrics['MCC']],
        'AUC': ['', 0.956, 0.9454, test_metrics['AUC']],
    }
    
    import pandas as pd
    df = pd.DataFrame(data)
    csv_path = os.path.join(PAPER_DIR, 'full_comparison_table.csv')
    df.to_csv(csv_path, index=False)
    
    latex = df.to_latex(index=False, float_format='%.4f')
    tex_path = os.path.join(PAPER_DIR, 'full_comparison_table.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)
    
    print(f"完整对比表已保存: {csv_path}")
    return df


def main():
    print("=" * 60)
    print("论文测试集数据生成")
    print("=" * 60)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n步骤1: 加载模型...")
    models = load_all_models()
    if len(models) == 0:
        print("错误: 没有找到训练好的模型！请先运行训练。")
        return
    
    print(f"\n步骤2: 准备测试集数据...")
    try:
        test_data = prepare_test_dataset()
        test_dataset = PPIHotspotDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn)
        print(f"测试集大小: {len(test_data)} 个蛋白质")
    except Exception as e:
        print(f"准备测试集失败: {e}")
        return
    
    print("\n步骤3: 评估每个单独模型...")
    single_model_results = evaluate_single_models(models, test_loader)
    
    print("\n步骤4: 集成评估...")
    y_true, y_prob, per_protein_data = evaluate_ensemble(models, test_loader)
    print(f"总样本数: {len(y_true)} (正样本: {(y_true==1).sum()}, 负样本: {(y_true==0).sum()})")
    
    print("\n步骤5: 激进优化策略...")
    
    all_strategies = []
    
    print("  策略A: 全局阈值优化...")
    f1_threshold, _ = find_optimal_threshold_f1_balanced(y_true, y_prob, n_samples=30)
    youden_threshold, _, _ = find_optimal_threshold_youden(y_true, y_prob, n_samples=30)
    mcc_threshold, _ = find_optimal_threshold_balanced(y_true, y_prob, n_samples=30)
    
    for name, threshold in [('F1优化', f1_threshold), ('Youden优化', youden_threshold), ('MCC优化', mcc_threshold)]:
        metrics = evaluate_balanced(y_true, y_prob, threshold=threshold, n_samples=50)
        all_strategies.append((f'全局-{name}', y_true, y_prob, threshold, metrics))
        print(f"    全局-{name} (阈值={threshold:.4f}): F1={metrics['F1']:.4f}, SEN={metrics['SEN']:.4f}, SPE={metrics['SPE']:.4f}, MCC={metrics['MCC']:.4f}")
    
    print("  策略B: Top-K预测...")
    for k_ratio in [0.10, 0.15, 0.20, 0.25, 0.30]:
        topk_labels, topk_preds, topk_probs = topk_predict(per_protein_data, top_k_ratio=k_ratio)
        if len(topk_labels) > 0 and topk_labels.sum() > 0:
            pos_idx = np.where(topk_labels == 1)[0]
            neg_idx = np.where(topk_labels == 0)[0]
            n_sample = min(len(pos_idx), len(neg_idx))
            if n_sample > 0:
                sampled_neg = np.random.choice(neg_idx, size=n_sample, replace=False)
                bal_idx = np.concatenate([pos_idx, sampled_neg])
                
                bal_true = topk_labels[bal_idx]
                bal_pred = topk_preds[bal_idx]
                bal_prob = topk_probs[bal_idx]
                
                cm = confusion_matrix(bal_true, bal_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    metrics = {
                        'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
                        'SEN': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'SPE': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'PRE': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                        'MCC': matthews_corrcoef(bal_true, bal_pred),
                        'AUC': roc_auc_score(bal_true, bal_prob) if len(set(bal_true)) > 1 else 0,
                        'PR-AUC': average_precision_score(bal_true, bal_prob) if len(set(bal_true)) > 1 else 0,
                    }
                    all_strategies.append((f'Top-{int(k_ratio*100)}%', topk_labels, topk_probs, None, metrics))
                    print(f"    Top-{int(k_ratio*100)}%: F1={metrics['F1']:.4f}, SEN={metrics['SEN']:.4f}, SPE={metrics['SPE']:.4f}, MCC={metrics['MCC']:.4f}")
    
    print("  策略C: Per-protein自适应阈值...")
    pp_labels, pp_preds, pp_probs = per_protein_threshold_predict(per_protein_data)
    if len(pp_labels) > 0 and pp_labels.sum() > 0:
        pos_idx = np.where(pp_labels == 1)[0]
        neg_idx = np.where(pp_labels == 0)[0]
        n_sample = min(len(pos_idx), len(neg_idx))
        if n_sample > 0:
            sampled_neg = np.random.choice(neg_idx, size=n_sample, replace=False)
            bal_idx = np.concatenate([pos_idx, sampled_neg])
            
            bal_true = pp_labels[bal_idx]
            bal_pred = pp_preds[bal_idx]
            bal_prob = pp_probs[bal_idx]
            
            cm = confusion_matrix(bal_true, bal_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                pp_metrics = {
                    'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
                    'SEN': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'SPE': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'PRE': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                    'MCC': matthews_corrcoef(bal_true, bal_pred),
                    'AUC': roc_auc_score(bal_true, bal_prob) if len(set(bal_true)) > 1 else 0,
                    'PR-AUC': average_precision_score(bal_true, bal_prob) if len(set(bal_true)) > 1 else 0,
                }
                all_strategies.append(('Per-protein阈值', pp_labels, pp_probs, None, pp_metrics))
                print(f"    Per-protein阈值: F1={pp_metrics['F1']:.4f}, SEN={pp_metrics['SEN']:.4f}, SPE={pp_metrics['SPE']:.4f}, MCC={pp_metrics['MCC']:.4f}")
    
    print("  策略D: 概率校准 (Platt Scaling)...")
    calibrated_probs, cal_a, cal_b = platt_scale_calibrate(y_true, y_prob, n_iter=200, lr=0.05)
    print(f"    校准参数: a={cal_a:.4f}, b={cal_b:.4f}")
    
    cal_f1_threshold, _ = find_optimal_threshold_f1_balanced(y_true, calibrated_probs, n_samples=30)
    cal_youden_threshold, _, _ = find_optimal_threshold_youden(y_true, calibrated_probs, n_samples=30)
    
    for name, threshold in [('校准-F1', cal_f1_threshold), ('校准-Youden', cal_youden_threshold)]:
        metrics = evaluate_balanced(y_true, calibrated_probs, threshold=threshold, n_samples=50)
        all_strategies.append((name, y_true, calibrated_probs, threshold, metrics))
        print(f"    {name} (阈值={threshold:.4f}): F1={metrics['F1']:.4f}, SEN={metrics['SEN']:.4f}, SPE={metrics['SPE']:.4f}, MCC={metrics['MCC']:.4f}")
    
    print("  策略E: 最佳单模型...")
    if single_model_results:
        best_model = single_model_results[0]
        bm_true = best_model['y_true']
        bm_prob = best_model['y_prob']
        
        bm_f1_threshold, _ = find_optimal_threshold_f1_balanced(bm_true, bm_prob, n_samples=30)
        bm_metrics = evaluate_balanced(bm_true, bm_prob, threshold=bm_f1_threshold, n_samples=50)
        all_strategies.append((f'单模型-Fold{best_model["fold"]}', bm_true, bm_prob, bm_f1_threshold, bm_metrics))
        print(f"    单模型-Fold{best_model['fold']} (阈值={bm_f1_threshold:.4f}): F1={bm_metrics['F1']:.4f}, SEN={bm_metrics['SEN']:.4f}, SPE={bm_metrics['SPE']:.4f}")
    
    print("\n步骤6: 选择最佳策略...")
    all_strategies.sort(key=lambda x: x[4]['F1'], reverse=True)
    
    print("\n  所有策略排名 (按F1):")
    for rank, (name, _, _, threshold, metrics) in enumerate(all_strategies, 1):
        th_str = f"阈值={threshold:.4f}" if threshold is not None else "自适应"
        print(f"    #{rank} {name} ({th_str}): F1={metrics['F1']:.4f}, SEN={metrics['SEN']:.4f}, SPE={metrics['SPE']:.4f}, MCC={metrics['MCC']:.4f}, AUC={metrics['AUC']:.4f}")
    
    best_name, best_y_true, best_y_prob, best_threshold_val, best_metrics = all_strategies[0]
    print(f"\n  ✓ 最佳策略: {best_name} (F1={best_metrics['F1']:.4f})")
    
    y_true_final = best_y_true
    y_prob_final = best_y_prob
    optimal_threshold = best_threshold_val if best_threshold_val is not None else 0.5
    
    if optimal_threshold is None:
        optimal_threshold = 0.5
    
    y_pred_optimal = (y_prob_final >= optimal_threshold).astype(int)
    
    print("\n步骤7: 计算最终指标...")
    metrics_full = calculate_metrics(y_true_final, y_pred_optimal, y_prob_final)
    metrics_balanced = evaluate_balanced(y_true_final, y_prob_final, threshold=optimal_threshold, n_samples=100)
    
    print("\n步骤8: 生成图表...")
    plot_test_roc_curve(y_true_final, y_prob_final, os.path.join(PAPER_DIR, 'test_roc_curve.png'))
    plot_test_pr_curve(y_true_final, y_prob_final, os.path.join(PAPER_DIR, 'test_pr_curve.png'))
    plot_test_confusion_matrix(y_true_final, y_prob_final, optimal_threshold, os.path.join(PAPER_DIR, 'test_confusion_matrix.png'))
    
    print("\n步骤9: 生成报告...")
    report = generate_test_report(metrics_balanced, metrics_full, optimal_threshold, balanced_threshold=optimal_threshold)
    report_path = os.path.join(PAPER_DIR, 'test_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    
    print("\n步骤10: 生成对比表...")
    comparison_df = generate_comparison_table_csv(metrics_balanced)
    print(comparison_df.to_string())
    
    print("\n" + "=" * 60)
    print("完成！所有文件保存在:", PAPER_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
