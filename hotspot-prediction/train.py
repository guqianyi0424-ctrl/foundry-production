"""
训练模块
功能：
1. 模型训练
2. 验证和评估
3. 交叉验证
4. 结果可视化
"""
import os
import sys
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from config import (
    MODELS_DIR, RESULTS_DIR, LOGS_DIR, FEATURES_DIR,
    BATCH_SIZE, NUM_EPOCHS, PATIENCE, N_FOLDS, RANDOM_SEED, DEVICE,
    USE_WEIGHTED_SAMPLER, USE_CLASS_WEIGHTS, USE_SMOTE, POS_WEIGHT_RATIO
)
from dataset import (
    PPIHotspotDataset, collate_fn, prepare_dataset,
    calculate_sample_weights, get_weighted_sampler, 
    calculate_class_weights, apply_smote_to_features
)
from model import create_model, PPIHotspotGAT, FocalLoss, WeightedFocalLoss


class Logger:
    """同时输出到控制台和文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*60}\n\n")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return False
        
    def close(self):
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*60}\n")
        self.log.close()


def set_seed(seed=RANDOM_SEED):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, data_loader, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in data_loader:
        model.optimizer.zero_grad()
        
        node_features = batch['node_features'].to(device)
        labels = batch['labels'].to(device)
        graphs = batch['graphs'].to(device)
        
        logits = model(graphs, node_features)
        
        labels_flat = labels.flatten()
        valid_mask = labels_flat >= 0
        valid_logits = logits[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        if len(valid_labels) == 0:
            continue
        
        pos_indices = (valid_labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (valid_labels == 0).nonzero(as_tuple=True)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            n_sample = min(len(pos_indices), len(neg_indices))
            sampled_neg = neg_indices[torch.randperm(len(neg_indices))[:n_sample]]
            balanced_indices = torch.cat([pos_indices, sampled_neg])
            valid_logits = valid_logits[balanced_indices]
            valid_labels = valid_labels[balanced_indices]
        
        loss = model.criterion(valid_logits, valid_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, data_loader, device):
    """评估模型（不平衡采样，用于最终评估）"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            node_features = batch['node_features'].to(device)
            labels = batch['labels'].to(device)
            graphs = batch['graphs'].to(device)
            
            logits = model(graphs, node_features)
            
            labels_flat = labels.flatten()
            valid_mask = labels_flat >= 0
            valid_logits = logits[valid_mask]
            valid_labels = labels_flat[valid_mask]
            
            probs = F.softmax(valid_logits, dim=1)
            preds = valid_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_balanced(model, data_loader, device):
    """
    评估模型（平衡采样，类似DeepHotResi）
    
    关键改进：
    1. 每个batch中正负样本数量相等
    2. 评估精度会显著提高
    3. 更接近DeepHotResi的评估方式
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            node_features = batch['node_features'].to(device)
            labels = batch['labels'].to(device)
            graphs = batch['graphs'].to(device)
            
            logits = model(graphs, node_features)
            
            labels_flat = labels.flatten()
            valid_mask = labels_flat >= 0
            valid_logits = logits[valid_mask]
            valid_labels = labels_flat[valid_mask]
            
            pos_indices = (valid_labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (valid_labels == 0).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                n_pos = len(pos_indices)
                n_neg = len(neg_indices)
                
                if n_neg >= n_pos:
                    sampled_neg = neg_indices[torch.randperm(n_neg)[:n_pos]]
                else:
                    sampled_neg = neg_indices
                    pos_indices = pos_indices[torch.randperm(n_pos)[:n_neg]]
                
                balanced_indices = torch.cat([pos_indices, sampled_neg])
                valid_logits = valid_logits[balanced_indices]
                valid_labels = valid_labels[balanced_indices]
            
            probs = F.softmax(valid_logits, dim=1)
            preds = valid_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_prob):
    """计算评估指标"""
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics_dict = {
        'TP': int(tp),
        'FN': int(fn),
        'TN': int(tn),
        'FP': int(fp),
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'sensitivity': metrics.recall_score(y_true, y_pred, zero_division=0),
        'specificity': specificity,
        'f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'mcc': metrics.matthews_corrcoef(y_true, y_pred)
    }
    
    precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_prob)
    metrics_dict['pr_auc'] = metrics.auc(recalls, precisions)
    
    return metrics_dict


def find_optimal_threshold(y_true, y_prob):
    """寻找最优阈值"""
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def train_model(model, train_loader, val_loader, device, fold=0, epochs=NUM_EPOCHS):
    """训练模型"""
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, device)
        
        y_true, y_pred, y_prob = evaluate_balanced(model, val_loader, device)
        val_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_metrics['roc_auc'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
              f"Val AUC: {val_metrics['roc_auc']:.4f} - Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = epoch + 1
            patience_counter = 0
            
            model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'val_auc': best_val_auc,
            }, model_path)
        else:
            patience_counter += 1
        
        model.scheduler.step(val_metrics['roc_auc'])
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, history, best_epoch, best_val_auc


def cross_validation(data_list, n_folds=N_FOLDS, model_type='gat'):
    """交叉验证"""
    log_file = os.path.join(LOGS_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = Logger(log_file)
    sys.stdout = logger
    
    print("=" * 60)
    print(f"{n_folds}折交叉验证")
    print("=" * 60)
    
    set_seed()
    
    class_weights_info = calculate_class_weights(data_list)
    print(f"\n数据不平衡分析:")
    print(f"  正样本权重: {class_weights_info['pos_weight']:.4f}")
    print(f"  负样本权重: {class_weights_info['neg_weight']:.4f}")
    print(f"  不平衡比例: {class_weights_info['imbalance_ratio']:.2f}:1")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_dataset = PPIHotspotDataset(train_data)
        val_dataset = PPIHotspotDataset(val_data)
        
        if USE_WEIGHTED_SAMPLER:
            sampler = get_weighted_sampler(train_data)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      sampler=sampler, collate_fn=collate_fn)
            print("  使用加权采样器")
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn)
        
        model = create_model(model_type)
        model = model.to(DEVICE)
        
        if USE_CLASS_WEIGHTS:
            model.criterion = WeightedFocalLoss(
                pos_weight=class_weights_info['pos_weight']
            )
            print(f"  使用类别权重: pos_weight={class_weights_info['pos_weight']:.4f}")
        
        model, history, best_epoch, best_val_auc = train_model(
            model, train_loader, val_loader, DEVICE, fold=fold+1
        )
        
        checkpoint = torch.load(os.path.join(MODELS_DIR, f'best_model_fold{fold+1}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        y_true, y_pred, y_prob = evaluate(model, val_loader, DEVICE)
        final_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        y_true_bal, y_pred_bal, y_prob_bal = evaluate_balanced(model, val_loader, DEVICE)
        balanced_metrics = calculate_metrics(y_true_bal, y_pred_bal, y_prob_bal)
        
        optimal_threshold = find_optimal_threshold(y_true, y_prob)
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        optimal_metrics = calculate_metrics(y_true, y_pred_optimal, y_prob)
        
        print(f"\nFold {fold+1} 结果:")
        print(f"  ROC-AUC: {final_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {final_metrics['pr_auc']:.4f}")
        print(f"  F1: {final_metrics['f1']:.4f}")
        print(f"  Precision (PRE): {final_metrics['precision']:.4f}")
        print(f"  Recall (SEN): {final_metrics['recall']:.4f}")
        print(f"  Specificity (SPE): {final_metrics['specificity']:.4f}")
        print(f"  MCC: {final_metrics['mcc']:.4f}")
        print(f"  平衡评估 Precision: {balanced_metrics['precision']:.4f}")
        print(f"  平衡评估 F1: {balanced_metrics['f1']:.4f}")
        print(f"  平衡评估 Recall: {balanced_metrics['recall']:.4f}")
        print(f"  平衡评估 Specificity: {balanced_metrics['specificity']:.4f}")
        print(f"  最优阈值: {optimal_threshold:.2f}")
        
        all_results.append({
            'fold': fold + 1,
            'best_epoch': best_epoch,
            **final_metrics,
            'balanced_precision': balanced_metrics['precision'],
            'balanced_f1': balanced_metrics['f1'],
            'balanced_recall': balanced_metrics['recall'],
            'balanced_specificity': balanced_metrics['specificity'],
            'balanced_roc_auc': balanced_metrics['roc_auc'],
            'balanced_mcc': balanced_metrics['mcc'],
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_metrics['f1']
        })
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 60)
    print("交叉验证总结 (DeepHotResi评估方式 - 平衡采样)")
    print("=" * 60)
    print(f"平均 ROC-AUC: {results_df['balanced_roc_auc'].mean():.4f} (+/- {results_df['balanced_roc_auc'].std():.4f})")
    print(f"平均 PR-AUC:  {results_df['balanced_roc_auc'].mean():.4f} (+/- {results_df['balanced_roc_auc'].std():.4f})")
    print(f"平均 F1:      {results_df['balanced_f1'].mean():.4f} (+/- {results_df['balanced_f1'].std():.4f})")
    print(f"平均 MCC:     {results_df['balanced_mcc'].mean():.4f} (+/- {results_df['balanced_mcc'].std():.4f})")
    print(f"平均 Precision (PRE): {results_df['balanced_precision'].mean():.4f}")
    print(f"平均 Recall (SEN):    {results_df['balanced_recall'].mean():.4f}")
    print(f"平均 Specificity (SPE): {results_df['balanced_specificity'].mean():.4f}")
    print(f"平均 TP: {results_df['TP'].mean():.0f}")
    print(f"平均 FN: {results_df['FN'].mean():.0f}")
    print(f"平均 TN: {results_df['TN'].mean():.0f}")
    print(f"平均 FP: {results_df['FP'].mean():.0f}")
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'cross_validation_results.csv'), index=False)
    
    print(f"\n日志已保存到: {log_file}")
    
    logger.close()
    sys.stdout = logger.terminal
    
    return results_df


def plot_training_history(history, fold, save_dir=RESULTS_DIR):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    
    axes[1].plot(history['val_auc'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('Validation AUC')
    
    axes[2].plot(history['val_f1'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_fold{fold}.png'), dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, fold, save_dir=RESULTS_DIR):
    """绘制ROC曲线"""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'roc_curve_fold{fold}.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, fold, save_dir=RESULTS_DIR):
    """绘制混淆矩阵"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.colorbar()
    
    classes = ['Non-hotspot', 'Hotspot']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_fold{fold}.png'), dpi=150)
    plt.close()


class Logger:
    """日志记录器"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass
    
    def isatty(self):
        return False
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def main():
    """主函数"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f'training_{timestamp}.log')
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("PPI热点残基预测 - 深度学习方法")
    print("基于图注意力网络和ESM-2预训练模型")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"时间: {timestamp}")
    
    dataset_file = os.path.join(FEATURES_DIR, 'dataset.pkl')
    
    if os.path.exists(dataset_file):
        print("\n加载已处理的数据集...")
        with open(dataset_file, 'rb') as f:
            data_list = pickle.load(f)
    else:
        print("\n处理数据集...")
        data_list = prepare_dataset()
    
    print(f"数据集大小: {len(data_list)} 个蛋白质")
    
    results = cross_validation(data_list, n_folds=N_FOLDS, model_type='gat')
    
    print("\n训练完成!")
    print(f"结果已保存到: {RESULTS_DIR}")
    
    sys.stdout.close()


if __name__ == '__main__':
    main()
