# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-29 13:59
LastEditTime: 2022-11-23 15:33
LastEditors: MrZQAQ
Description: Prepare Data for main process
FilePath: /MCANet/utils/DataPrepare.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import numpy as np

def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:# 当前折既不是第 0 折（第一折），也不是最后一折（k-1 折）
        val_end = (i + 1) * fold_size#计算验证集的结束索引；
        validset = datasets[val_start:val_end]#截取当前折的验证集（列表切片，左闭右开）；
        trainset = datasets[0:val_start] + datasets[val_end:]#训练集 = 验证集之前的样本 + 验证集之后的样本；
    elif i == 0:# 第 0 折（第一折），验证集为第 0 到 fold_size 个样本
        val_end = fold_size# 第 0 折（第一折），验证集为第 0 到 fold_size 个样本
        validset = datasets[val_start:val_end]# 截取第 0 到 fold_size 个样本作为验证集
        trainset = datasets[val_end:]# 训练集 = 验证集之后的样本
    else:# 最后一折（k-1 折），验证集为第 (k-1)*fold_size 到最后一个样本
        validset = datasets[val_start:] # 截取第 (k-1)*fold_size 到最后一个样本作为验证集
        trainset = datasets[0:val_start]# 训练集 = 验证集之前的样本

    return trainset, validset#返回当前折的训练集和验证集，顺序固定（训练集在前，验证集在后）

def shuffle_dataset(dataset, seed):
    '''
    Description: 随机打乱数据集，确保不同随机种子下的实验结果可复现
    Args:
        dataset: 待打乱的数据集（列表或数组）
        seed: 随机种子（整数）
    Returns:
        dataset: 打乱后的数据集（列表或数组）
    '''
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
