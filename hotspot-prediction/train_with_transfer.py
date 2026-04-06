"""
整合标签转移策略的训练脚本
功能：
1. 加载数据
2. 应用标签转移策略增强数据
3. 结合Focal Loss + 类别权重 + 平衡采样
4. 训练模型
5. 评估模型
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    MODELS_DIR, RESULTS_DIR, DEVICE, INPUT_DIM, HIDDEN_DIM, 
    NUM_HEADS, NUM_LAYERS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, N_FOLDS
)
from dataset import (
    load_raw_data, download_pdb_files, extract_sequence_and_coords,
    prepare_dataset, PPIHotspotDataset, collate_fn, calculate_class_weights
)
from model import create_model, FocalLoss, WeightedFocalLoss
from label_transfer import LabelTransfer, prepare_protein_data_for_transfer


def apply_label_transfer(data_list, use_external_tools=False):
    """
    应用标签转移策略
    
    Args:
        data_list: 原始数据列表
        use_external_tools: 是否使用外部工具（BLAST, TM-align）
        
    Returns:
        增强后的数据列表
    """
    print("\n" + "=" * 60)
    print("应用标签转移策略")
    print("=" * 60)
    
    protein_data = {}
    
    for item in data_list:
        pdb_id = item.get('pdb_id', '')
        chain = item.get('chain', 'A')
        chain_id = f"{pdb_id}_{chain}"
        
        hotspots = item.get('hotspots', [])
        if isinstance(hotspots, str):
            try:
                hotspots = eval(hotspots)
            except:
                hotspots = []
        
        protein_data[chain_id] = {
            'sequence': item.get('sequence', ''),
            'coords': item.get('coords'),
            'hotspots': hotspots,
            'pdb_file': item.get('pdb_file')
        }
    
    transfer = LabelTransfer(
        seq_similarity_threshold=0.4,
        tm_score_threshold=0.5
    )
    
    enhanced_data, stats = transfer.augment_dataset(protein_data, use_external_tools)
    
    enhanced_list = []
    for chain_id, data in enhanced_data.items():
        pdb_id, chain = chain_id.rsplit('_', 1)
        
        original_item = None
        for item in data_list:
            if item.get('pdb_id', '') == pdb_id and item.get('chain', 'A') == chain:
                original_item = item
                break
        
        if original_item:
            new_item = original_item.copy()
            new_item['hotspots'] = data['hotspots']
            new_item['transferred'] = data.get('transferred', False)
            enhanced_list.append(new_item)
    
    return enhanced_list, stats


def train_epoch_with_balance(model, data_loader, device):
    """训练一个epoch（带平衡采样）"""
    model.train()
    total_loss = 0.0
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
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_balanced(model, data_loader, device):
    """评估模型（带平衡采样）"""
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
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        metrics['specificity'] = 0
    
    return metrics


def train_with_label_transfer(args):
    """
    使用标签转移策略训练模型
    """
    print("\n" + "=" * 70)
    print("整合标签转移策略的训练")
    print("=" * 70)
    
    print("\n加载数据...")
    df = load_raw_data()
    
    print(f"原始数据: {len(df)} 条记录")
    
    print("\n准备数据集...")
    data_list = prepare_dataset(force_reload=args.force_reload)
    
    if args.use_label_transfer:
        print("\n应用标签转移策略...")
        data_list, transfer_stats = apply_label_transfer(
            data_list, 
            use_external_tools=args.use_external_tools
        )
        print(f"标签转移完成: 热点残基增加 {transfer_stats['increase_ratio']:.2f}%")
    
    print(f"\n数据集大小: {len(data_list)} 个蛋白质")
    
    class_weights_info = calculate_class_weights(data_list)
    print(f"\n类别权重信息:")
    print(f"  正样本权重: {class_weights_info['pos_weight']:.4f}")
    print(f"  负样本权重: {class_weights_info['neg_weight']:.4f}")
    print(f"  不平衡比例: {class_weights_info['imbalance_ratio']:.2f}:1")
    
    n_total = len(data_list)
    fold_size = n_total // N_FOLDS
    indices = np.random.permutation(n_total)
    
    results = []
    
    for fold in range(N_FOLDS):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'=' * 60}")
        
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < N_FOLDS - 1 else n_total
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        
        train_dataset = PPIHotspotDataset(train_data)
        val_dataset = PPIHotspotDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        model = create_model('gat')
        model = model.to(DEVICE)
        
        if args.use_focal_loss:
            model.criterion = WeightedFocalLoss(
                pos_weight=class_weights_info['pos_weight'],
                gamma=2.0
            )
            print(f"  使用 Weighted Focal Loss (pos_weight={class_weights_info['pos_weight']:.4f})")
        
        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch_with_balance(model, train_loader, DEVICE)
            
            y_true, y_pred, y_prob = evaluate_balanced(model, val_loader, DEVICE)
            metrics = calculate_metrics(y_true, y_pred, y_prob)
            
            if metrics['roc_auc'] > best_val_auc:
                best_val_auc = metrics['roc_auc']
                best_epoch = epoch + 1
                patience_counter = 0
                
                os.makedirs(MODELS_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'metrics': metrics
                }, os.path.join(MODELS_DIR, f'best_model_fold{fold + 1}_transfer.pth'))
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, "
                      f"Val AUC={metrics['roc_auc']:.4f}, "
                      f"Val F1={metrics['f1']:.4f}")
            
            if patience_counter >= 20:
                print(f"  早停: {patience_counter} epochs 无改善")
                break
        
        checkpoint = torch.load(os.path.join(MODELS_DIR, f'best_model_fold{fold + 1}_transfer.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        y_true, y_pred, y_prob = evaluate_balanced(model, val_loader, DEVICE)
        final_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        results.append({
            'fold': fold + 1,
            'best_epoch': best_epoch,
            **final_metrics
        })
        
        print(f"\n  Fold {fold + 1} 最终结果:")
        print(f"    ROC-AUC: {final_metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:  {final_metrics['pr_auc']:.4f}")
        print(f"    F1:      {final_metrics['f1']:.4f}")
        print(f"    MCC:     {final_metrics['mcc']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("交叉验证总结")
    print("=" * 70)
    print(f"平均 ROC-AUC: {results_df['roc_auc'].mean():.4f} (+/- {results_df['roc_auc'].std():.4f})")
    print(f"平均 PR-AUC:  {results_df['pr_auc'].mean():.4f} (+/- {results_df['pr_auc'].std():.4f})")
    print(f"平均 F1:      {results_df['f1'].mean():.4f} (+/- {results_df['f1'].std():.4f})")
    print(f"平均 MCC:     {results_df['mcc'].mean():.4f} (+/- {results_df['mcc'].std():.4f})")
    print(f"平均 Precision: {results_df['precision'].mean():.4f}")
    print(f"平均 Recall:    {results_df['recall'].mean():.4f}")
    print(f"平均 Specificity: {results_df['specificity'].mean():.4f}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'cross_validation_with_transfer.csv'), index=False)
    
    print(f"\n结果已保存到: {os.path.join(RESULTS_DIR, 'cross_validation_with_transfer.csv')}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='整合标签转移策略的训练')
    parser.add_argument('--use-label-transfer', action='store_true', 
                        help='使用标签转移策略')
    parser.add_argument('--use-external-tools', action='store_true',
                        help='使用外部工具（BLAST, TM-align）')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='使用Focal Loss')
    parser.add_argument('--force-reload', action='store_true',
                        help='强制重新处理数据')
    args = parser.parse_args()
    
    results = train_with_label_transfer(args)
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
