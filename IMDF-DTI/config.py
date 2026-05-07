# -*- coding:utf-8 -*-
class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4  # 学习率
        self.Epoch = 200           # 训练轮数
        self.Batch_size = 128      # 批次大小
        self.Patience = 40         # 早停机制的耐心值，即多少轮loss不下降则停止训练l
        self.decay_interval = 10   # 学习率衰减间隔
        self.lr_decay = 0.5        # 学习率衰减率
        self.weight_decay = 1e-4   # 权重衰减（L2正则化）
        self.embed_dim = 64        # 嵌入维度
        self.protein_kernel = [4, 8, 12] # 蛋白质CNN卷积核大小
        self.drug_kernel = [4, 6, 8]     # 药物CNN卷积核大小
        self.conv = 50             # 卷积层输出通道数
        self.char_dim = 64         # 字符嵌入维度
        self.loss_epsilon = 1.0      # 损失函数epsilon参数 1
        self.lambda_l1 = 0.002   # L1 正则化强度（惩罚力度越大，特征越稀疏）
        self.warmup_epochs = 10   # 预热期，建议设为总 Epoch 的 10%-20%
        self.lambda_balance = 0.01  #0.05  # 平衡约束系数
        self.gate_lr_ratio = 0.1   # gate 学习率与主模型学习率的比例
        self.gate_update_interval = 5 # gate 每隔多少轮更新一次
        self.gate_freeze_epochs = 10  # 前多少轮完全冻结 gate
