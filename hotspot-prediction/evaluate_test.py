"""
独立测试集评估脚本
功能：
1. 加载训练好的模型
2. 在独立测试集 (data2) 上评估
3. 生成详细的评估报告
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import MODELS_DIR, RESULTS_DIR, DEVICE, INPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT
from dataset import prepare_test_dataset, PPIHotspotDataset, collate_fn
from model import create_model


def load_best_model(model_path, model_type='gat'):
    """加载训练好的模型"""
    model = create_model(model_type)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    return model


def evaluate_on_test_set(models, test_loader):
    """在测试集上评估模型（支持集成）"""
    for model in models:
        model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_pdb_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            node_features = batch['node_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            graphs = batch['graphs'].to(DEVICE)
            pdb_ids = batch['pdb_ids']
            
            all_logits = []
            for model in models:
                logits = model(graphs, node_features)
                all_logits.append(logits)
            
            logits = torch.stack(all_logits).mean(dim=0)
            
            labels_flat = labels.flatten()
            valid_mask = labels_flat >= 0
            
            probs = F.softmax(logits, dim=1)
            
            for i, pdb_id in enumerate(pdb_ids):
                start_idx = sum(len(batch['labels'][j]) for j in range(i))
                end_idx = start_idx + len(batch['labels'][i])
                
                protein_labels = labels_flat[start_idx:end_idx]
                protein_probs = probs[start_idx:end_idx, 1]
                
                valid_protein_mask = protein_labels >= 0
                
                all_pdb_ids.extend([pdb_id] * valid_protein_mask.sum().item())
                all_labels.extend(protein_labels[valid_protein_mask].cpu().numpy())
                all_probs.extend(protein_probs[valid_protein_mask].cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), all_pdb_ids


def calculate_metrics(labels, probs, threshold=0.5):
    """计算各种评估指标"""
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(labels, probs),
        'pr_auc': average_precision_score(labels, probs),
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'mcc': matthews_corrcoef(labels, preds),
    }
    
    cm = confusion_matrix(labels, preds)
    metrics['tn'] = cm[0, 0] if cm.shape == (2, 2) else 0
    metrics['fp'] = cm[0, 1] if cm.shape == (2, 2) else 0
    metrics['fn'] = cm[1, 0] if cm.shape == (2, 2) else 0
    metrics['tp'] = cm[1, 1] if cm.shape == (2, 2) else 0
    
    specificity = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
    metrics['specificity'] = specificity
    
    return metrics


def calculate_balanced_metrics(labels, probs, threshold=0.5):
    """计算平衡评估指标（类似DeepHotResi）"""
    preds = (probs >= threshold).astype(int)
    
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return None
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    if n_neg >= n_pos:
        sampled_neg = np.random.choice(neg_indices, size=n_pos, replace=False)
    else:
        sampled_neg = neg_indices
        pos_indices = np.random.choice(pos_indices, size=n_neg, replace=False)
    
    balanced_indices = np.concatenate([pos_indices, sampled_neg])
    balanced_labels = labels[balanced_indices]
    balanced_probs = probs[balanced_indices]
    balanced_preds = (balanced_probs >= threshold).astype(int)
    
    cm = confusion_matrix(balanced_labels, balanced_preds)
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'roc_auc': roc_auc_score(balanced_labels, balanced_probs),
        'pr_auc': average_precision_score(balanced_labels, balanced_probs),
        'accuracy': accuracy_score(balanced_labels, balanced_preds),
        'precision': precision_score(balanced_labels, balanced_preds, zero_division=0),
        'recall': recall_score(balanced_labels, balanced_preds, zero_division=0),
        'specificity': specificity,
        'f1': f1_score(balanced_labels, balanced_preds, zero_division=0),
        'mcc': matthews_corrcoef(balanced_labels, balanced_preds),
    }
    
    return metrics


def find_optimal_threshold(labels, probs):
    """找到最优阈值"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def plot_roc_curve(labels, probs, save_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存: {save_path}")


def plot_pr_curve(labels, probs, save_path):
    """绘制PR曲线"""
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存: {save_path}")


def generate_report(metrics, balanced_metrics, save_path, model_path, threshold):
    """生成评估报告"""
    report = []
    report.append("=" * 60)
    report.append("独立测试集评估报告")
    report.append("=" * 60)
    report.append(f"\n模型路径: {model_path}")
    report.append(f"使用阈值: {threshold:.4f}")
    report.append("\n" + "-" * 40)
    report.append("标准评估指标:")
    report.append("-" * 40)
    report.append(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    report.append(f"PR-AUC:      {metrics['pr_auc']:.4f}")
    report.append(f"Accuracy:    {metrics['accuracy']:.4f}")
    report.append(f"Precision:   {metrics['precision']:.4f}")
    report.append(f"Recall (SEN): {metrics['recall']:.4f}")
    report.append(f"F1 Score:    {metrics['f1']:.4f}")
    report.append(f"Specificity (SPE): {metrics['specificity']:.4f}")
    report.append(f"MCC:         {metrics['mcc']:.4f}")
    
    if balanced_metrics:
        report.append("\n" + "-" * 40)
        report.append("平衡评估指标 (类似DeepHotResi):")
        report.append("-" * 40)
        report.append(f"ROC-AUC:     {balanced_metrics['roc_auc']:.4f}")
        report.append(f"PR-AUC:      {balanced_metrics['pr_auc']:.4f}")
        report.append(f"Precision:   {balanced_metrics['precision']:.4f}")
        report.append(f"Recall (SEN): {balanced_metrics['recall']:.4f}")
        report.append(f"Specificity (SPE): {balanced_metrics['specificity']:.4f}")
        report.append(f"F1 Score:    {balanced_metrics['f1']:.4f}")
        report.append(f"MCC:         {balanced_metrics['mcc']:.4f}")
    
    report.append("\n" + "-" * 40)
    report.append("混淆矩阵:")
    report.append("-" * 40)
    report.append(f"True Negative (TN):  {metrics['tn']}")
    report.append(f"False Positive (FP): {metrics['fp']}")
    report.append(f"False Negative (FN): {metrics['fn']}")
    report.append(f"True Positive (TP):  {metrics['tp']}")
    report.append("\n" + "-" * 40)
    report.append("样本统计:")
    report.append("-" * 40)
    report.append(f"总样本数:    {metrics['tn'] + metrics['fp'] + metrics['fn'] + metrics['tp']}")
    report.append(f"正样本数:    {metrics['fn'] + metrics['tp']}")
    report.append(f"负样本数:    {metrics['tn'] + metrics['fp']}")
    report.append(f"正样本比例:  {(metrics['fn'] + metrics['tp']) / (metrics['tn'] + metrics['fp'] + metrics['fn'] + metrics['tp']) * 100:.2f}%")
    report.append("\n" + "=" * 60)
    
    report_text = "\n".join(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存: {save_path}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='独立测试集评估')
        parser.add_argument('--model', type=str, default='gat', help='模型类型')
        parser.add_argument('--model-path', type=str, default=None, help='模型路径')
        parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
        parser.add_argument('--find-threshold', action='store_true', help='自动寻找最优阈值')
        parser.add_argument('--force-reload', action='store_true', help='强制重新处理数据')
        args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("独立测试集评估")
    print("=" * 60)
    
    model_paths = []
    if args.model_path is None:
        for fold in range(1, 6):
            fold_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}.pth')
            if os.path.exists(fold_path):
                model_paths.append(fold_path)
        
        if len(model_paths) == 0:
            single_path = os.path.join(MODELS_DIR, f'best_model_{args.model}.pth')
            if os.path.exists(single_path):
                model_paths.append(single_path)
        
        if len(model_paths) == 0:
            print(f"错误: 找不到模型文件")
            print(f"请先运行训练: python main.py --mode train")
            return
    else:
        model_paths = [args.model_path]
    
    print(f"\n找到 {len(model_paths)} 个模型文件:")
    for p in model_paths:
        print(f"  - {p}")
    
    models = [load_best_model(p, args.model) for p in model_paths]
    
    print("\n准备测试数据集...")
    test_data = prepare_test_dataset(force_reload=args.force_reload)
    
    if len(test_data) == 0:
        print("错误: 测试数据集为空")
        return
    
    print(f"测试数据集大小: {len(test_data)} 个蛋白质")
    
    test_dataset = PPIHotspotDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print("\n评估模型...")
    labels, probs, pdb_ids = evaluate_on_test_set(models, test_loader)
    
    if args.find_threshold:
        threshold = find_optimal_threshold(labels, probs)
        print(f"\n自动找到最优阈值: {threshold:.4f}")
    else:
        threshold = args.threshold
    
    metrics = calculate_metrics(labels, probs, threshold)
    
    balanced_metrics = calculate_balanced_metrics(labels, probs, threshold)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    report_path = os.path.join(RESULTS_DIR, 'test_evaluation_report.txt')
    roc_path = os.path.join(RESULTS_DIR, 'test_roc_curve.png')
    pr_path = os.path.join(RESULTS_DIR, 'test_pr_curve.png')
    
    generate_report(metrics, balanced_metrics, report_path, model_paths[0], threshold)
    plot_roc_curve(labels, probs, roc_path)
    plot_pr_curve(labels, probs, pr_path)
    
    np.save(os.path.join(RESULTS_DIR, 'test_predictions.npy'), {
        'labels': labels,
        'probs': probs,
        'pdb_ids': pdb_ids
    })
    
    print("\n评估完成!")


if __name__ == '__main__':
    main()
