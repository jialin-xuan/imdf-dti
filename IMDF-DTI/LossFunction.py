# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((labels.shape[0], 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)

def get_mask_l1_loss(model):
    """
    计算模型中所有 FeatureMask 层的 L1 正则项 (Innovation Point 2)
    公式: ||σ(δ)||_1 [cite: 105]
    """
    l1_loss = 0
    # 遍历模型中定义的四个掩码层
    masks = [
        model.mask_2d_drug, model.mask_2d_prot,
        model.mask_1d_drug, model.mask_1d_prot
    ]
    for mask_layer in masks:
        # 获取当前掩码值 σ(δ)
        m = torch.sigmoid(mask_layer.delta)
        l1_loss += torch.norm(m, p=1)
    return l1_loss
