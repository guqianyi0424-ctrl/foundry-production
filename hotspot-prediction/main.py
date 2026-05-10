"""
PPI热点残基预测 - 主程序入口
基于图注意力网络和ESM-2预训练模型
"""
import os
import sys
import argparse
import pickle
import torch

from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, DEVICE
from dataset import prepare_dataset, prepare_test_dataset, PPIHotspotDataset, collate_fn
from model import create_model
from train import cross_validation, train_model, evaluate, calculate_metrics
from torch.utils.data import DataLoader


def run_pipeline(args):
    """运行完整流程"""
    print("=" * 70)
    print("PPI热点残基预测 - 深度学习方法")
    print("基于图注意力网络和ESM-2预训练模型")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    
    if args.mode == 'eval_test':
        print("\n独立测试集评估模式...")
        from evaluate_test import main as eval_test_main
        eval_test_args = type('Args', (), {
            'model': args.model,
            'model_path': None,
            'threshold': 0.5,
            'find_threshold': False,
            'force_reload': args.reprocess,
            'predictions_csv': None,
            'metrics_csv': None,
        })()
        eval_test_main(eval_test_args)
        return
    
    dataset_file = os.path.join(FEATURES_DIR, 'dataset.pkl')
    
    if os.path.exists(dataset_file) and not args.reprocess:
        print("\n加载已处理的数据集...")
        with open(dataset_file, 'rb') as f:
            data_list = pickle.load(f)
    else:
        print("\n处理数据集...")
        data_list = prepare_dataset()
    
    print(f"数据集大小: {len(data_list)} 个蛋白质")
    
    if args.mode == 'train':
        print("\n开始训练...")
        results = cross_validation(data_list, n_folds=args.n_folds, model_type=args.model)
        
    elif args.mode == 'test':
        print("\n测试模式...")
        model = create_model(args.model)
        model_path = os.path.join(MODELS_DIR, args.checkpoint)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型: {model_path}")
        else:
            print(f"模型文件不存在: {model_path}")
            return
        
        model = model.to(DEVICE)
        model.eval()
        
    elif args.mode == 'inference':
        print("\n推理模式...")
        print("请提供PDB文件进行预测")
    
    print("\n完成!")


def main():
    parser = argparse.ArgumentParser(description='PPI热点残基预测')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'inference', 'eval_test'],
                       help='运行模式: train=训练, test=测试, inference=推理, eval_test=独立测试集评估')
    parser.add_argument('--model', type=str, default='gat',
                       choices=['gat', 'gat_v2', 'ensemble'],
                       help='模型类型')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--reprocess', action='store_true',
                       help='重新处理数据集')
    parser.add_argument('--checkpoint', type=str, default='best_model_fold1.pth',
                       help='模型检查点文件')
    
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
