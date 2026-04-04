"""
模型模块
基于图注意力网络和ESM-2预训练模型的PPI热点残基预测
借鉴DeepHotResi架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv, GraphConv
import numpy as np

from config import (
    INPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
    NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY, FOCAL_ALPHA, FOCAL_GAMMA, DEVICE,
    LABEL_SMOOTHING, USE_LABEL_SMOOTHING
)


class SELayer(nn.Module):
    """Squeeze-and-Excitation层"""
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.size()
        y = self.fc(x)
        return x * y.expand_as(x)


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss - 防止过拟合"""
    
    def __init__(self, classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FocalLossWithLabelSmoothing(nn.Module):
    """Focal Loss with Label Smoothing - 组合策略"""
    
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, 
                 smoothing=LABEL_SMOOTHING, classes=NUM_CLASSES):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.classes = classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        pred = inputs.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        ce_loss = torch.sum(-true_dist * pred, dim=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class WeightedFocalLoss(nn.Module):
    """加权 Focal Loss - 类别权重 + Focal Loss"""
    
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, 
                 pos_weight=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        weights = torch.ones_like(targets, dtype=torch.float)
        if self.pos_weight is not None:
            weights[targets == 1] = self.pos_weight
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss * weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PPIHotspotGAT(nn.Module):
    """PPI热点残基预测模型 - 图注意力网络版本"""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, 
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(PPIHotspotGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.se = SELayer(input_dim, reduction=16)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, 
                           num_heads=num_heads, feat_drop=dropout, 
                           allow_zero_in_degree=True)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads,
                           num_heads=num_heads, feat_drop=dropout,
                           allow_zero_in_degree=True)
                )
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        if USE_LABEL_SMOOTHING:
            self.criterion = FocalLossWithLabelSmoothing(
                alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, 
                smoothing=LABEL_SMOOTHING, classes=num_classes
            )
        else:
            self.criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
    
    def forward(self, g, node_features):
        x = node_features.float()
        
        x_se = self.se(x)
        
        h = self.input_proj(x_se)
        h = F.relu(h)
        
        for i, gat_layer in enumerate(self.gat_layers):
            h_res = h
            h = gat_layer(g, h).flatten(1)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if i > 0:
                h = h + h_res
        
        h = torch.cat([h, x_se], dim=1)
        
        logits = self.classifier(h)
        
        return logits


class PPIHotspotGAT_v2(nn.Module):
    """PPI热点残基预测模型 - 增强版"""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(PPIHotspotGAT_v2, self).__init__()
        
        self.se = SELayer(input_dim, reduction=16)
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.gat_layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                       num_heads=num_heads, feat_drop=dropout,
                       allow_zero_in_degree=True,
                       residual=True)
            )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
        self.criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
    
    def forward(self, g, node_features):
        x = node_features.float()
        
        x = self.input_bn(x)
        x_se = self.se(x)
        
        h = self.input_proj(x_se)
        
        all_h = [h]
        
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(g, h).flatten(1)
            h_new = self.layer_norms[i](h_new)
            h_new = F.relu(h_new)
            h = h + h_new
            all_h.append(h)
        
        h_stack = torch.stack(all_h, dim=0)
        attn_weights = F.softmax(self.attention_pool(h_stack), dim=0)
        h_pooled = (h_stack * attn_weights).sum(dim=0)
        
        h_final = torch.cat([h_pooled, h, x_se], dim=1)
        
        logits = self.classifier(h_final)
        
        return logits


class PPIHotspotEnsemble(nn.Module):
    """集成模型"""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_heads=NUM_HEADS, num_classes=NUM_CLASSES):
        super(PPIHotspotEnsemble, self).__init__()
        
        self.model1 = PPIHotspotGAT(input_dim, hidden_dim, num_heads, 
                                    num_layers=2, dropout=0.3)
        self.model2 = PPIHotspotGAT(input_dim, hidden_dim, num_heads,
                                    num_layers=3, dropout=0.5)
        self.model3 = PPIHotspotGAT(input_dim, hidden_dim, num_heads,
                                    num_layers=4, dropout=0.5)
        
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
        self.criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    
    def forward(self, g, node_features):
        out1 = self.model1(g, node_features)
        out2 = self.model2(g, node_features)
        out3 = self.model3(g, node_features)
        
        weights = F.softmax(self.weights, dim=0)
        
        logits = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
        
        return logits


def create_model(model_type='gat', **kwargs):
    """创建模型工厂函数"""
    if model_type == 'gat':
        return PPIHotspotGAT(**kwargs)
    elif model_type == 'gat_v2':
        return PPIHotspotGAT_v2(**kwargs)
    elif model_type == 'ensemble':
        return PPIHotspotEnsemble(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    model = create_model('gat')
    print(model)
    
    num_nodes = 100
    input_dim = INPUT_DIM
    
    g = dgl.rand_graph(num_nodes, num_nodes * 5)
    x = torch.randn(num_nodes, input_dim)
    
    output = model(g, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
