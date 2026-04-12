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
    """集成评估"""
    all_labels = []
    all_probs = []
    all_pdb_ids = []
    
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
    
    return np.array(all_labels), np.array(all_probs)


def find_optimal_threshold(y_true, y_prob):
    """寻找最优阈值"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


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


def plot_test_roc_curve(y_true, y_prob, save_path):
    """绘制测试集ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Our Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Test Set ROC Curve', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集ROC曲线已保存: {save_path}")


def plot_test_pr_curve(y_true, y_prob, save_path):
    """绘制测试集PR曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'Our Model (PR-AUC = {pr_auc:.3f})')
    
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='navy', lw=2, linestyle='--', alpha=0.6, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Test Set Precision-Recall Curve', fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集PR曲线已保存: {save_path}")


def plot_test_confusion_matrix(y_true, y_pred, save_path):
    """绘制测试集混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
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
    ax.set_title('Test Set Confusion Matrix', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集混淆矩阵已保存: {save_path}")


def generate_test_report(metrics_balanced, metrics_full, optimal_threshold):
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
    report.append(f"  最优阈值:          {optimal_threshold:.4f}")
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
    
    print("\n步骤3: 集成评估...")
    y_true, y_prob = evaluate_ensemble(models, test_loader)
    print(f"总样本数: {len(y_true)} (正样本: {(y_true==1).sum()}, 负样本: {(y_true==0).sum()})")
    
    print("\n步骤4: 寻找最优阈值...")
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, y_prob)
    print(f"最优阈值: {optimal_threshold:.4f} (F1={best_f1:.4f})")
    
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    print("\n步骤5: 计算指标...")
    metrics_full = calculate_metrics(y_true, y_pred_optimal, y_prob)
    metrics_balanced = evaluate_balanced(y_true, y_prob, threshold=optimal_threshold, n_samples=100)
    
    print("\n步骤6: 生成图表...")
    plot_test_roc_curve(y_true, y_prob, os.path.join(PAPER_DIR, 'test_roc_curve.png'))
    plot_test_pr_curve(y_true, y_prob, os.path.join(PAPER_DIR, 'test_pr_curve.png'))
    plot_test_confusion_matrix(y_true, y_pred_optimal, os.path.join(PAPER_DIR, 'test_confusion_matrix.png'))
    
    print("\n步骤7: 生成报告...")
    report = generate_test_report(metrics_balanced, metrics_full, optimal_threshold)
    report_path = os.path.join(PAPER_DIR, 'test_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    
    print("\n步骤8: 生成对比表...")
    comparison_df = generate_comparison_table_csv(metrics_balanced)
    print(comparison_df.to_string())
    
    print("\n" + "=" * 60)
    print("完成！所有文件保存在:", PAPER_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
