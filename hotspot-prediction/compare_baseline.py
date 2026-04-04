"""
与Baseline (ppihotspotid) 对比评估模块
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt

from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, DEVICE
from dataset import PPIHotspotDataset, collate_fn
from model import create_model


def load_deep_model(checkpoint_path):
    """加载深度学习模型"""
    model = create_model('gat')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_deep_model(model, data_loader):
    """评估深度学习模型"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            node_features = batch['node_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            graphs = batch['graphs'].to(DEVICE)
            
            logits = model(graphs, node_features)
            
            valid_mask = labels >= 0
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            
            probs = torch.softmax(valid_logits, dim=1)[:, 1]
            preds = valid_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_baseline(X_test, y_test, model_path):
    """评估baseline模型 (AutoGluon)"""
    try:
        from autogluon.tabular import TabularPredictor
        
        predictor = TabularPredictor.load(model_path)
        y_pred = predictor.predict(X_test)
        y_prob = predictor.predict_proba(X_test).iloc[:, 1]
        
        return y_test.values, y_pred.values, y_prob.values
    except Exception as e:
        print(f"加载baseline模型失败: {e}")
        return None, None, None


def compare_models(deep_results, baseline_results, save_path):
    """对比两个模型的结果"""
    y_true_deep, y_pred_deep, y_prob_deep = deep_results
    y_true_base, y_pred_base, y_prob_base = baseline_results
    
    metrics_deep = {
        'ROC-AUC': metrics.roc_auc_score(y_true_deep, y_prob_deep),
        'PR-AUC': metrics.average_precision_score(y_true_deep, y_prob_deep),
        'F1': metrics.f1_score(y_true_deep, y_pred_deep),
        'Precision': metrics.precision_score(y_true_deep, y_pred_deep),
        'Recall': metrics.recall_score(y_true_deep, y_pred_deep),
        'Accuracy': metrics.accuracy_score(y_true_deep, y_pred_deep),
        'MCC': metrics.matthews_corrcoef(y_true_deep, y_pred_deep)
    }
    
    metrics_baseline = {
        'ROC-AUC': metrics.roc_auc_score(y_true_base, y_prob_base),
        'PR-AUC': metrics.average_precision_score(y_true_base, y_prob_base),
        'F1': metrics.f1_score(y_true_base, y_pred_base),
        'Precision': metrics.precision_score(y_true_base, y_pred_base),
        'Recall': metrics.recall_score(y_true_base, y_pred_base),
        'Accuracy': metrics.accuracy_score(y_true_base, y_pred_base),
        'MCC': metrics.matthews_corrcoef(y_true_base, y_pred_base)
    }
    
    comparison_df = pd.DataFrame({
        'Metric': list(metrics_deep.keys()),
        'Deep Learning (Ours)': list(metrics_deep.values()),
        'Baseline (AutoGluon)': list(metrics_baseline.values()),
        'Improvement': [metrics_deep[k] - metrics_baseline[k] for k in metrics_deep.keys()]
    })
    
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(os.path.join(save_path, 'model_comparison.csv'), index=False)
    
    return comparison_df


def plot_comparison_roc(deep_results, baseline_results, save_path):
    """绘制ROC对比曲线"""
    y_true_deep, _, y_prob_deep = deep_results
    y_true_base, _, y_prob_base = baseline_results
    
    fpr_deep, tpr_deep, _ = metrics.roc_curve(y_true_deep, y_prob_deep)
    fpr_base, tpr_base, _ = metrics.roc_curve(y_true_base, y_prob_base)
    
    auc_deep = metrics.auc(fpr_deep, tpr_deep)
    auc_base = metrics.auc(fpr_base, tpr_base)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_deep, tpr_deep, label=f'Deep Learning (AUC = {auc_deep:.4f})', linewidth=2)
    plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {auc_base:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'roc_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_pr(deep_results, baseline_results, save_path):
    """绘制PR对比曲线"""
    y_true_deep, _, y_prob_deep = deep_results
    y_true_base, _, y_prob_base = baseline_results
    
    prec_deep, rec_deep, _ = metrics.precision_recall_curve(y_true_deep, y_prob_deep)
    prec_base, rec_base, _ = metrics.precision_recall_curve(y_true_base, y_prob_base)
    
    ap_deep = metrics.average_precision_score(y_true_deep, y_prob_deep)
    ap_base = metrics.average_precision_score(y_true_base, y_prob_base)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rec_deep, prec_deep, label=f'Deep Learning (AP = {ap_deep:.4f})', linewidth=2)
    plt.plot(rec_base, prec_base, label=f'Baseline (AP = {ap_base:.4f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'pr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("对比评估模块")
    print("请先运行训练脚本生成模型")
