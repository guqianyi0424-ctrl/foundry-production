"""
论文数据生成脚本
功能：
1. 特征消融实验 (类似DeepHotResi Table 4)
2. ROC曲线 (5折 + 平均, 类似DeepHotResi Figure 3)
3. PR曲线
4. 混淆矩阵
5. 注意力权重可视化
6. 与基线方法对比表
7. 独立测试集评估
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
    precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    MODELS_DIR, RESULTS_DIR, FEATURES_DIR, LOGS_DIR,
    BATCH_SIZE, NUM_EPOCHS, PATIENCE, N_FOLDS, RANDOM_SEED, DEVICE,
    USE_WEIGHTED_SAMPLER, USE_CLASS_WEIGHTS,
    INPUT_DIM, ESM2_DIM, PSSM_DIM, HMM_DIM, TRADITIONAL_DIM,
    HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING, USE_LABEL_SMOOTHING,
    NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY
)
from dataset import (
    PPIHotspotDataset, collate_fn, prepare_dataset,
    calculate_sample_weights, get_weighted_sampler,
    calculate_class_weights
)
from model import PPIHotspotGAT, SELayer, FocalLoss, WeightedFocalLoss, FocalLossWithLabelSmoothing

PAPER_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(PAPER_DIR, exist_ok=True)

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5


def set_seed(seed=RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
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
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
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


def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    result = {
        'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
        'SEN': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'SPE': specificity,
        'PRE': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    result['PR-AUC'] = metrics.auc(recalls, precisions)
    return result


def train_fold(model, train_loader, val_loader, device, fold, epochs=NUM_EPOCHS):
    best_val_auc = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
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
            loss = model.criterion(valid_logits, valid_labels)
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        y_true, y_pred, y_prob = evaluate_balanced(model, val_loader, device)
        val_metrics = calc_metrics(y_true, y_pred, y_prob)
        
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
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
        
        model.scheduler.step(val_metrics['AUC'])
        
        if patience_counter >= PATIENCE:
            break
    
    checkpoint = torch.load(os.path.join(MODELS_DIR, f'best_model_fold{fold}.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def plot_roc_curves_five_fold(all_fpr, all_tpr, all_auc, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(len(all_fpr)):
        ax.plot(all_fpr[i], all_tpr[i], color=colors[i], lw=1.5,
                label=f'Fold {i+1} (AUC = {all_auc[i]:.3f})', alpha=0.8)
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for i in range(len(all_fpr)):
        interp_tpr = np.interp(mean_fpr, all_fpr[i], all_tpr[i])
        interp_tpr[0] = 0.0
        mean_tpr += interp_tpr
    mean_tpr /= len(all_fpr)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(all_auc)
    
    ax.plot(mean_fpr, mean_tpr, color='red', lw=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.2f}$\\pm${std_auc:.3f})')
    
    std_tpr = np.zeros_like(mean_fpr)
    for i in range(len(all_fpr)):
        interp_tpr = np.interp(mean_fpr, all_fpr[i], all_tpr[i])
        interp_tpr[0] = 0.0
        std_tpr += (interp_tpr - mean_tpr) ** 2
    std_tpr = np.sqrt(std_tpr / len(all_fpr))
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='red', alpha=0.15)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve - Five-fold Cross Validation', fontsize=16)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存: {save_path}")


def plot_pr_curves_five_fold(all_recall, all_precision, all_pr_auc, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(len(all_recall)):
        ax.plot(all_recall[i], all_precision[i], color=colors[i], lw=1.5,
                label=f'Fold {i+1} (AUC = {all_pr_auc[i]:.3f})', alpha=0.8)
    
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i in range(len(all_recall)):
        interp_prec = np.interp(mean_recall, all_recall[i][::-1], all_precision[i][::-1])
        mean_precision += interp_prec
    mean_precision /= len(all_recall)
    mean_pr_auc = np.mean(all_pr_auc)
    std_pr_auc = np.std(all_pr_auc)
    
    ax.plot(mean_recall, mean_precision, color='red', lw=2.5,
            label=f'Mean PR (AUC = {mean_pr_auc:.2f}$\\pm${std_pr_auc:.3f})')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve - Five-fold Cross Validation', fontsize=16)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=16)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    classes = ['Non-hotspot', 'Hotspot']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > thresh else 'black')
    
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_attention_weights(model, data_loader, device, save_path, top_n=20):
    model.eval()
    all_attn = []
    all_labels = []
    all_residue_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            node_features = batch['node_features'].to(device)
            labels = batch['labels'].to(device)
            graphs = batch['graphs'].to(device)
            
            x = node_features.float()
            x_se = model.se(x)
            h = model.input_proj(x_se)
            
            for i, gat_layer in enumerate(model.gat_layers):
                attn = gat_layer(g, h)
                h_new = attn.flatten(1)
                h_new = model.layer_norms[i](h_new)
                h_new = F.relu(h_new)
                h = h + h_new
            
            labels_flat = labels.flatten()
            valid_mask = labels_flat >= 0
            
            pos_mask = (labels_flat == 1) & valid_mask
            if pos_mask.sum() > 0:
                h_norm = torch.norm(h, dim=1)
                all_attn.extend(h_norm[pos_mask].cpu().numpy())
                all_labels.extend(labels_flat[pos_mask].cpu().numpy())
    
    if len(all_attn) == 0:
        print("没有正样本，跳过注意力可视化")
        return
    
    attn_array = np.array(all_attn)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(attn_array, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Attention Weight (Feature Norm)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of Attention Weights for Hotspot Residues', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"注意力权重分布已保存: {save_path}")


def feature_ablation_study(data_list, device):
    print("\n" + "=" * 60)
    print("特征消融实验 (类似DeepHotResi Table 4)")
    print("=" * 60)
    
    ablation_configs = {
        'Full Model': {'use_esm2': True, 'use_pssm': True, 'use_hmm': True, 'use_trad': True},
        'w/o ESM-2': {'use_esm2': False, 'use_pssm': True, 'use_hmm': True, 'use_trad': True},
        'w/o PSSM': {'use_esm2': True, 'use_pssm': False, 'use_hmm': True, 'use_trad': True},
        'w/o HMM': {'use_esm2': True, 'use_pssm': True, 'use_hmm': False, 'use_trad': True},
        'w/o Traditional': {'use_esm2': True, 'use_pssm': True, 'use_hmm': True, 'use_trad': False},
    }
    
    feature_dims = {
        'esm2': ESM2_DIM,
        'pssm': PSSM_DIM,
        'hmm': HMM_DIM,
        'trad': TRADITIONAL_DIM,
    }
    
    all_results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"\n--- {config_name} ---")
        
        input_dim = 0
        if config['use_esm2']:
            input_dim += feature_dims['esm2']
        if config['use_pssm']:
            input_dim += feature_dims['pssm']
        if config['use_hmm']:
            input_dim += feature_dims['hmm']
        if config['use_trad']:
            input_dim += feature_dims['trad']
        
        if input_dim == 0:
            continue
        
        modified_data = []
        for item in data_list:
            new_item = dict(item)
            features = item['node_features']
            start = 0
            selected_features = []
            
            if config['use_esm2']:
                selected_features.append(features[:, start:start+ESM2_DIM])
            start += ESM2_DIM
            
            if config['use_pssm']:
                selected_features.append(features[:, start:start+PSSM_DIM])
            start += PSSM_DIM
            
            if config['use_hmm']:
                selected_features.append(features[:, start:start+HMM_DIM])
            start += HMM_DIM
            
            if config['use_trad']:
                selected_features.append(features[:, start:start+TRADITIONAL_DIM])
            
            if selected_features:
                new_item['node_features'] = np.concatenate(selected_features, axis=1)
            modified_data.append(new_item)
        
        set_seed()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(modified_data)):
            train_data = [modified_data[i] for i in train_idx]
            val_data = [modified_data[i] for i in val_idx]
            
            train_dataset = PPIHotspotDataset(train_data)
            val_dataset = PPIHotspotDataset(val_data)
            
            sampler = get_weighted_sampler(train_data)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      sampler=sampler, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn)
            
            model = PPIHotspotGAT(input_dim=input_dim)
            model = model.to(device)
            
            class_weights_info = calculate_class_weights(train_data)
            model.criterion = WeightedFocalLoss(pos_weight=class_weights_info['pos_weight'])
            
            model = train_fold(model, train_loader, val_loader, device, fold+1)
            
            y_true, y_pred, y_prob = evaluate_balanced(model, val_loader, device)
            fold_metrics = calc_metrics(y_true, y_pred, y_prob)
            fold_results.append(fold_metrics)
        
        avg_results = {}
        for key in ['SEN', 'SPE', 'PRE', 'F1', 'MCC', 'AUC']:
            values = [r[key] for r in fold_results]
            avg_results[key] = np.mean(values)
            avg_results[f'{key}_std'] = np.std(values)
        
        all_results[config_name] = avg_results
        print(f"  AUC: {avg_results['AUC']:.4f} ± {avg_results['AUC_std']:.4f}")
        print(f"  F1:  {avg_results['F1']:.4f} ± {avg_results['F1_std']:.4f}")
        print(f"  MCC: {avg_results['MCC']:.4f} ± {avg_results['MCC_std']:.4f}")
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df[['SEN', 'SPE', 'PRE', 'F1', 'MCC', 'AUC']]
    results_df.to_csv(os.path.join(PAPER_DIR, 'feature_ablation_results.csv'))
    
    print("\n" + "=" * 60)
    print("特征消融实验结果汇总")
    print("=" * 60)
    print(results_df.to_string(float_format='%.4f'))
    
    return results_df


def run_five_fold_with_curves(data_list, device):
    print("\n" + "=" * 60)
    print("5折交叉验证 + 生成论文图表")
    print("=" * 60)
    
    set_seed()
    class_weights_info = calculate_class_weights(data_list)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    all_fold_metrics = []
    all_fpr, all_tpr, all_auc = [], [], []
    all_recall, all_precision, all_pr_auc = [], [], []
    all_y_true_bal, all_y_pred_bal, all_y_prob_bal = [], [], []
    all_y_true_full, all_y_prob_full = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_dataset = PPIHotspotDataset(train_data)
        val_dataset = PPIHotspotDataset(val_data)
        
        sampler = get_weighted_sampler(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  sampler=sampler, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn)
        
        model = PPIHotspotGAT()
        model = model.to(device)
        model.criterion = WeightedFocalLoss(pos_weight=class_weights_info['pos_weight'])
        
        model = train_fold(model, train_loader, val_loader, device, fold+1)
        
        y_true_bal, y_pred_bal, y_prob_bal = evaluate_balanced(model, val_loader, device)
        fold_metrics = calc_metrics(y_true_bal, y_pred_bal, y_prob_bal)
        all_fold_metrics.append(fold_metrics)
        
        all_y_true_bal.extend(y_true_bal)
        all_y_pred_bal.extend(y_pred_bal)
        all_y_prob_bal.extend(y_prob_bal)
        
        y_true_full, y_pred_full, y_prob_full = evaluate_model(model, val_loader, device)
        all_y_true_full.extend(y_true_full)
        all_y_prob_full.extend(y_prob_full)
        
        fpr, tpr, _ = roc_curve(y_true_bal, y_prob_bal)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(fold_metrics['AUC'])
        
        prec, rec, _ = precision_recall_curve(y_true_bal, y_prob_bal)
        all_recall.append(rec)
        all_precision.append(prec)
        all_pr_auc.append(fold_metrics['PR-AUC'])
        
        plot_confusion_matrix(
            y_true_bal, y_pred_bal,
            os.path.join(PAPER_DIR, f'confusion_matrix_fold{fold+1}.png'),
            title=f'Confusion Matrix - Fold {fold+1}'
        )
        
        plot_attention_weights(
            model, val_loader, device,
            os.path.join(PAPER_DIR, f'attention_weights_fold{fold+1}.png')
        )
    
    plot_roc_curves_five_fold(
        all_fpr, all_tpr, all_auc,
        os.path.join(PAPER_DIR, 'roc_curve_five_fold.png')
    )
    
    plot_pr_curves_five_fold(
        all_recall, all_precision, all_pr_auc,
        os.path.join(PAPER_DIR, 'pr_curve_five_fold.png')
    )
    
    plot_confusion_matrix(
        np.array(all_y_true_bal), np.array(all_y_pred_bal),
        os.path.join(PAPER_DIR, 'confusion_matrix_overall.png'),
        title='Confusion Matrix - Overall (Balanced)'
    )
    
    print("\n" + "=" * 60)
    print("5折交叉验证结果汇总 (平衡评估)")
    print("=" * 60)
    
    summary = {}
    for key in ['SEN', 'SPE', 'PRE', 'F1', 'MCC', 'AUC', 'PR-AUC']:
        values = [r[key] for r in all_fold_metrics]
        summary[key] = {'mean': np.mean(values), 'std': np.std(values)}
        print(f"  {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    
    summary_df = pd.DataFrame([
        {
            'Metric': key,
            'Mean': val['mean'],
            'Std': val['std']
        }
        for key, val in summary.items()
    ])
    summary_df.to_csv(os.path.join(PAPER_DIR, 'five_fold_summary.csv'), index=False)
    
    return all_fold_metrics


def generate_comparison_table(our_metrics):
    print("\n" + "=" * 60)
    print("与基线方法对比表 (类似DeepHotResi Table 2/3)")
    print("=" * 60)
    
    ppihotspotid_metrics = {
        'SEN': 0.67, 'SPE': 0.83, 'PRE': 0.76, 'F1': 0.71, 'MCC': None, 'AUC': None
    }
    
    deephotresi_metrics = {
        'SEN': 0.957, 'SPE': 0.903, 'PRE': 0.872, 'F1': 0.912, 'MCC': 0.755, 'AUC': 0.956
    }
    
    our_sen = np.mean([r['SEN'] for r in our_metrics])
    our_spe = np.mean([r['SPE'] for r in our_metrics])
    our_pre = np.mean([r['PRE'] for r in our_metrics])
    our_f1 = np.mean([r['F1'] for r in our_metrics])
    our_mcc = np.mean([r['MCC'] for r in our_metrics])
    our_auc = np.mean([r['AUC'] for r in our_metrics])
    
    comparison = pd.DataFrame({
        'Method': ['PPI-HotspotID', 'DeepHotResi', 'Our Model'],
        'SEN': [ppihotspotid_metrics['SEN'], deephotresi_metrics['SEN'], our_sen],
        'SPE': [ppihotspotid_metrics['SPE'], deephotresi_metrics['SPE'], our_spe],
        'PRE': [ppihotspotid_metrics['PRE'], deephotresi_metrics['PRE'], our_pre],
        'F1': [ppihotspotid_metrics['F1'], deephotresi_metrics['F1'], our_f1],
        'MCC': [ppihotspotid_metrics['MCC'], deephotresi_metrics['MCC'], our_mcc],
        'AUC': [ppihotspotid_metrics['AUC'], deephotresi_metrics['AUC'], our_auc],
    })
    
    comparison.to_csv(os.path.join(PAPER_DIR, 'comparison_table.csv'), index=False)
    
    print(comparison.to_string(index=False, float_format='%.4f'))
    
    print("\n" + "=" * 60)
    print("对比表 (LaTeX格式)")
    print("=" * 60)
    
    latex = comparison.to_latex(index=False, float_format='%.4f')
    print(latex)
    
    with open(os.path.join(PAPER_DIR, 'comparison_table.tex'), 'w') as f:
        f.write(latex)
    
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description='论文数据生成')
    parser.add_argument('--skip-ablation', action='store_true', help='跳过特征消融实验')
    parser.add_argument('--skip-curves', action='store_true', help='跳过曲线生成')
    parser.add_argument('--force-reload', action='store_true', help='强制重新加载数据')
    args = parser.parse_args()
    
    print("=" * 60)
    print("论文数据生成脚本")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"输出目录: {PAPER_DIR}")
    
    dataset_file = os.path.join(FEATURES_DIR, 'dataset.pkl')
    if os.path.exists(dataset_file) and not args.force_reload:
        print("\n加载已处理的数据集...")
        with open(dataset_file, 'rb') as f:
            data_list = pickle.load(f)
    else:
        print("\n处理数据集...")
        data_list = prepare_dataset()
    
    print(f"数据集大小: {len(data_list)} 个蛋白质")
    
    if not args.skip_curves:
        fold_metrics = run_five_fold_with_curves(data_list, DEVICE)
    else:
        print("\n跳过曲线生成，使用已有模型...")
        fold_metrics = []
        set_seed()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        class_weights_info = calculate_class_weights(data_list)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
            val_data = [data_list[i] for i in val_idx]
            val_dataset = PPIHotspotDataset(val_data)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn)
            
            model_path = os.path.join(MODELS_DIR, f'best_model_fold{fold+1}.pth')
            if not os.path.exists(model_path):
                print(f"  模型文件不存在: {model_path}")
                continue
            
            model = PPIHotspotGAT()
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
            
            y_true, y_pred, y_prob = evaluate_balanced(model, val_loader, DEVICE)
            fold_metrics.append(calc_metrics(y_true, y_pred, y_prob))
    
    if fold_metrics:
        generate_comparison_table(fold_metrics)
    
    if not args.skip_ablation:
        feature_ablation_study(data_list, DEVICE)
    
    print("\n" + "=" * 60)
    print("论文数据生成完成!")
    print(f"所有图表和数据已保存到: {PAPER_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
