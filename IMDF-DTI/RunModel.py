# -*- coding:utf-8 -*-

import os
import random
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import MIFDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
# from LossFunction import CELoss, PolyLoss
# # 将 get_mask_l1_loss 添加到导入语句中
from LossFunction import CELoss, PolyLoss, get_mask_l1_loss
from utils.TestModel import test_model
from utils.ShowResult import show_result
from utils import protein_init, ligand_init, ProteinMoleculeDataset
import torch_geometric.loader as pyg_loader

import json
import csv


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_MIF_model(SEED, DATASET, MODEL, K_Fold, LOSS, device):
    '''设置随机种子'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''初始化超参数'''
    hp = hyperparameter()

    '''从文本文件加载数据集'''
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''设置损失函数权重'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    if DATASET == "BD2D":
        split_pos = 52010
        train_data_list = data_list[:split_pos]
        test_data_list = data_list[split_pos:]
        '''打乱数据'''
        print("data shuffle")
        train_data_list = shuffle_dataset(train_data_list, SEED)
    else:
        '''打乱数据'''
        print("data shuffle")
        data_list = shuffle_dataset(data_list, SEED)

        '''将数据集分割为训练集&验证集和测试集'''
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''数据预处理与加载'''
    # 1. 蛋白质序列转图并缓存
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein-new.pkl'# 蛋白图缓存路径
    if os.path.exists(protein_path):# 若缓存文件存在，直接加载
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)# 反序列化：返回字典（key=蛋白序列，value=蛋白图数据）
    else:# 若缓存文件不存在，重新转换并保存
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))# 提取所有唯一蛋白序列（去重，避免重复转换）
        protein_dict = protein_init(protein_seqs)# 自定义函数：蛋白序列→图结构数据，返回字典
        joblib.dump(protein_dict,protein_path)# 序列化：保存为pkl文件
    # 2. 配体SMILES转图并缓存（逻辑与蛋白完全一致）
    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi-new.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()

    '''初始化评价指标列表'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    # 解析：分别保存准确率、ROC-AUC、PR-AUC、召回率、精确率，均为DTI任务核心评价指标；
    # 每折测试完成后，将指标append到对应列表

    for i_fold in range(K_Fold):# 从0到K_Fold-1循环，共K折
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)# 打印折数，分隔实验日志，方便查看

        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold) # 自定义函数：将train_data_list按K折划分为当前折的训练集和验证集
        train_dataset = ProteinMoleculeDataset(train_dataset, ligand_dict, protein_dict, device=device) # 初始化DTI专属图数据集：将样本映射为蛋白-配体图对，传入设备
        valid_dataset = ProteinMoleculeDataset(valid_dataset, ligand_dict, protein_dict, device=device) # 初始化DTI专属图数据集：将样本映射为蛋白-配体图对，传入设备
        test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device) # 初始化DTI专属图数据集：将样本映射为蛋白-配体图对，传入设备
        train_size = len(train_dataset)# 训练集样本数，用于设置学习率调度器的step_size_up参数（每个epoch的迭代次数）

        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)# 初始化训练集数据加载器：批量大小、是否打乱、遵循批量处理、是否丢弃最后一个不完整批次
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)# 初始化验证集数据加载器：批量大小、是否打乱、遵循批量处理、是否丢弃最后一个不完整批次
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)# 初始化测试集数据加载器：批量大小、是否打乱、遵循批量处理、是否丢弃最后一个不完整批次
                                    
        """ create model"""
        model = MODEL(device=device)

        """Initialize weights"""
        weight_p, bias_p = [], []# 分别保存模型参数中的权重和偏置项
        for p in model.parameters():# 遍历模型所有参数
            if p.dim() > 1:# 若参数维度大于1（即权重矩阵）
                nn.init.xavier_uniform_(p)# 采用Xavier均匀初始化，保持输入输出方差一致，避免梯度消失或爆炸
        for name, p in model.named_parameters():# 遍历模型所有参数，包含参数名
            if 'bias' in name:# 若参数名包含'bias'（即偏置项）
                bias_p += [p]# 将偏置项添加到bias_p列表
            else:# 若参数名不包含'bias'（即权重矩阵）
                weight_p += [p]# 将权重矩阵添加到weight_p列表

        """create optimizer and scheduler"""
#         optimizer = optim.AdamW(
#             [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)# 初始化AdamW优化器：包含权重和偏置项，分别设置不同的权重衰减系数（0.01和0.0），学习率为hp.Learning_rate

# # 循环学习率调度器：让学习率在[base_lr, max_lr]之间循环变化，提升模型泛化能力
#         scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
#                                                 step_size_up=train_size // hp.Batch_size)# 初始化循环学习率调度器：基础学习率为hp.Learning_rate，最大学习率为hp.Learning_rate*10，每个epoch迭代train_size // hp.Batch_size次
#         # if LOSS == 'PolyLoss':
#         #     Loss = PolyLoss(weight_loss=weight_loss,
#         #                     DEVICE=device, epsilon=hp.loss_epsilon)
#         # else:
        gate_p, base_p = [], []

        for name, p in model.named_parameters():
            if "dfgu" in name.lower() or "gate" in name.lower():
                gate_p.append(p)
            else:
                base_p.append(p)

        optimizer = optim.AdamW(
            [
                {'params': base_p, 'weight_decay': hp.weight_decay, 'lr': hp.Learning_rate},
                {'params': gate_p, 'weight_decay': hp.weight_decay, 'lr': hp.Learning_rate * hp.gate_lr_ratio},
            ]
        )
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=[
                hp.Learning_rate,
                hp.Learning_rate * hp.gate_lr_ratio
            ],
            max_lr=[
                hp.Learning_rate * 10,
                hp.Learning_rate * hp.gate_lr_ratio * 10
            ],
            cycle_momentum=False,
            step_size_up=train_size // hp.Batch_size
        )
        Loss = CELoss(weight_CE=weight_loss, DEVICE=device)

        """Output files"""
        save_path = "./" + DATASET + "/{}".format(i_fold+1)# 定义当前折的输出路径，用于保存模型、训练日志、测试结果等
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        # --- 📍 新增：保存超参数到 hyperparameters.json ---
        # 将 hp 对象转为字典并保存
        hp_dict = {k: v for k, v in hp.__dict__.items() if not k.startswith('__')}
        with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
            json.dump(hp_dict, f, indent=4)

        # --- 📍 新增：初始化 training_results.csv ---
        csv_file = os.path.join(save_path, 'training_results.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Epoch', 'Train_Loss', 'Valid_Loss', 'Valid_AUC', 'Valid_PRC', 'Accuracy', 'Drug_Weight_2D', 'Prot_Weight_2D'])

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)# 初始化早停机制：若验证集loss在hp.Patience轮内未下降，则停止训练，防止过拟合

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):# 遍历每个epoch
            if early_stopping.early_stop == True:
                break

            """train"""
            train_losses_in_epoch = []# 用于记录当前epoch内每个batch的训练loss
            epoch_w2d, epoch_w1d = [], []# 用于记录当前epoch内每个batch的权重2d和权重1d
            model.train()# 切换模型为训练模式，启用dropout等训练时特有的操作
            # 前 gate_freeze_epochs 轮冻结 gate，之后每 gate_update_interval 轮更新一次
            if epoch <= hp.gate_freeze_epochs:
                train_gate = False
            else:
                train_gate = ((epoch - hp.gate_freeze_epochs - 1) % hp.gate_update_interval == 0)

            for name, param in model.named_parameters():
                if "dfgu" in name.lower() or "gate" in name.lower():
                    param.requires_grad = train_gate

            if train_gate:
                print(f"🔓 Epoch {epoch}: update gate/dfgu")
            else:
                print(f"🔒 Epoch {epoch}: freeze gate/dfgu")
            for data in train_loader:# 遍历训练集数据加载器，每个batch包含hp.Batch_size个样本
                optimizer.zero_grad()# 清空当前batch的梯度，避免梯度累加

                data = data.to(device)# 将当前batch的数据移动到指定设备（CPU或GPU）
                predicted_y= model(data)# 前向传播：计算当前batch的预测输出
                # --- 📍 插入这些行：采集当前 Batch 的动态权重 ---
                with torch.no_grad():
                    # 只有在 2.0 模型中才有这些门控单元
                    w2d = model.dfgu_2d.latest_weights.cpu().numpy()
                    w1d = model.dfgu_1d.latest_weights.cpu().numpy()
                    epoch_w2d.append(w2d)
                    epoch_w1d.append(w1d)
                # # 1. 计算基础预测损失 (CE 或 PolyLoss)
                # pred_loss = Loss(predicted_y, data.cls_y)

                # # --- 📍 新增：计算模态平衡损失 (确保梯度回传) ---
                # # 获取 DFGU 内部的 Tensor 权重 (Shape 通常为 [Batch, 2])
                # w2d_tensor = model.dfgu_2d.latest_weights 
                # w1d_tensor = model.dfgu_1d.latest_weights
                
                # # 计算两路权重的差值绝对值。如果是 [Batch, 2]，则取第 0 列(Drug)和第 1 列(Prot)
                # # 这样写无论是一维还是二维都能正确运行
                # balance_loss_2d = torch.abs(w2d_tensor.narrow(-1, 0, 1) - w2d_tensor.narrow(-1, 1, 1)).mean()
                # balance_loss_1d = torch.abs(w1d_tensor[:, 0] - w1d_tensor[:, 1]).mean()
                
                # # 2. 初始总损失 = 预测损失 + 平衡约束 (系数 0.2 可以根据效果调整)
                # train_loss = pred_loss + 0.2 * (balance_loss_2d + balance_loss_1d)

                # #消融实验（原）
                # # 2. 两阶段策略:判断是否开启 L1 惩罚 [cite: 46, 83]
                # # if epoch > hp.warmup_epochs:
                # #     l1_penalty = get_mask_l1_loss(model)
                # #     # 总损失 = 预测损失 + λ * L1 惩罚 [cite: 105]
                # #     train_loss = pred_loss + hp.lambda_l1 * l1_penalty
                # # else:
                # #     train_loss = pred_loss
                # if epoch > hp.warmup_epochs:
                #     l1_penalty = get_mask_l1_loss(model)
                #     progress = min(1.0, (epoch - hp.warmup_epochs) / 10.0)  # 10轮内线性升温
                #     lambda_t = hp.lambda_l1 * progress
                #     train_loss = pred_loss + lambda_t * l1_penalty
                # else:
                #     l1_penalty = torch.tensor(0.0, device=device)
                #     lambda_t = 0.0
                #     train_loss = pred_loss
                # 1. 基础预测损失
                pred_loss = Loss(predicted_y, data.cls_y)

                # 2. 模态平衡损失
                w2d_tensor = model.dfgu_2d.latest_weights   # [B, 2]
                w1d_tensor = model.dfgu_1d.latest_weights   # [B, 2]

                balance_loss_2d = torch.abs(
                    w2d_tensor.narrow(-1, 0, 1) - w2d_tensor.narrow(-1, 1, 1)
                ).mean()

                balance_loss_1d = torch.abs(
                    w1d_tensor.narrow(-1, 0, 1) - w1d_tensor.narrow(-1, 1, 1)
                ).mean()

                balance_loss = balance_loss_2d + balance_loss_1d

                # 3. 总损失
                train_loss = pred_loss

                if epoch > hp.warmup_epochs:
                    # 3.1 只有 gate 更新轮才加模态平衡损失
                    if train_gate:
                        train_loss = train_loss + hp.lambda_balance * balance_loss

                    # 3.2 L1 mask 惩罚正常开启
                    l1_penalty = get_mask_l1_loss(model)
                    progress = min(1.0, (epoch - hp.warmup_epochs) / 10.0)
                    lambda_t = hp.lambda_l1 * progress
                    train_loss = train_loss + lambda_t * l1_penalty
                else:
                    l1_penalty = torch.tensor(0.0, device=device)
                    lambda_t = 0.0
                #结束
                # 消融实验：禁用 L1 掩码惩罚项
                # train_loss = pred_loss
                #如果恢复
                # train_loss = Loss(predicted_y, data.cls_y)# 计算当前batch的训练loss
                train_losses_in_epoch.append(train_loss.item())# 将当前batch的训练loss添加到train_losses_in_epoch列表
                train_loss.backward()# 反向传播：计算当前batch的梯度
                optimizer.step()# 更新模型参数：根据当前batch的梯度，使用优化器更新模型参数
                scheduler.step()# 更新学习率：根据循环学习率调度器，更新当前epoch的学习率
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            valid_losses_in_epoch = []# 用于记录当前epoch内每个batch的验证loss
            model.eval()# 切换模型为评估模式，禁用dropout等训练时特有的操作
            Y, P, S = [], [], []# 分别用于记录验证集真实标签、预测标签、预测分数
            with torch.no_grad():# 禁用梯度计算，节省内存和计算时间
                for data in valid_loader:# 遍历验证集数据加载器，每个batch包含hp.Batch_size个样本

                    data = data.to(device)# 将当前batch的数据移动到指定设备（CPU或GPU）

                    valid_scores = model(data)# 前向传播：计算当前batch的验证输出
                    
                    valid_labels = data.cls_y# 获取当前batch的验证标签
                    valid_loss = Loss(valid_scores, valid_labels)# 计算当前batch的验证loss
                    valid_losses_in_epoch.append(valid_loss.item())# 将当前batch的验证loss添加到valid_losses_in_epoch列表
                    valid_labels = valid_labels.to('cpu').data.numpy()# 将当前batch的验证标签从GPU移动到CPU，并转换为NumPy数组
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()# 将当前batch的验证输出从GPU移动到CPU，并转换为NumPy数组，同时应用softmax函数归一化
                    valid_predictions = np.argmax(valid_scores, axis=1)# 对当前batch的验证输出进行预测，取概率最大的类别作为预测标签
                    valid_scores = valid_scores[:, 1]# 提取当前batch的验证输出中类别为1的概率，作为预测分数

                    Y.extend(valid_labels)# 将当前batch的验证标签添加到Y列表
                    P.extend(valid_predictions)# 将当前batch的验证预测标签添加到P列表
                    S.extend(valid_scores)# 将当前batch的验证预测分数添加到S列表

            Precision_dev = precision_score(Y, P)# 计算当前epoch的验证集精度（Precision）精确率：TP/(TP+FP)
            Reacll_dev = recall_score(Y, P)# 计算当前epoch的验证集召回率（Reacll）
            Accuracy_dev = accuracy_score(Y, P)# 计算当前epoch的验证集准确率（Accuracy）
            AUC_dev = roc_auc_score(Y, S)# 计算当前epoch的验证集AUC（Area Under the ROC Curve）
            tpr, fpr, _ = precision_recall_curve(Y, S)# 计算当前epoch的验证集PRC（Precision-Recall Curve）
            PRC_dev = auc(fpr, tpr)# 计算当前epoch的验证集PRC下的面积（AUC）
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)# 一次epoch的平均验证loss

            epoch_len = len(str(hp.Epoch))# 计算epoch数的字符串长度，用于格式化输出
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')# 格式化输出当前epoch的训练loss、验证loss、AUC、PRC、准确率、精确率、召回率
            print(print_msg)
            # --- 📍 插入这些行：计算并输出本轮平均权重 ---
            avg_w2d = np.mean(epoch_w2d, axis=0)
            avg_w1d = np.mean(epoch_w1d, axis=0)
            print(f"   📊 Modality Weights -> [2D] Drug:{avg_w2d[0]:.3f}, Prot:{avg_w2d[1]:.3f} | [1D] Drug:{avg_w1d[0]:.3f}, Prot:{avg_w1d[1]:.3f}")

            # --- 📍 修改 3：将本轮结果实时追加到 CSV ---
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    f"{train_loss_a_epoch:.5f}", 
                    f"{valid_loss_a_epoch:.5f}", 
                    f"{AUC_dev:.5f}", 
                    f"{PRC_dev:.5f}", 
                    f"{Accuracy_dev:.5f}",
                    f"{avg_w2d[0]:.3f}", # 记录 2D 药物权重
                    f"{avg_w2d[1]:.3f}"  # 记录 2D 蛋白权重
                ])

            '''save checkpoint and make decision when early stop'''
            early_stopping(AUC_dev, model, epoch)# 调用早停函数，判断是否需要早停并保存最佳模型

        '''load best checkpoint'''
        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{str(device).replace(":", "_")}.pth', weights_only=True))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1, MIF=True)# 测试训练集，返回测试结果、准确率、精确率、召回率、AUC、PRC
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1, MIF=True)# 测试验证集，返回测试结果、准确率、精确率、召回率、AUC、PRC
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1, MIF=True)# 测试测试集，返回测试结果、准确率、精确率、召回率、AUC、PRC
        AUC_List_stable.append(AUC_test)# 将当前测试集的AUC添加到AUC_List_stable列表
        Accuracy_List_stable.append(Accuracy_test)# 将当前测试集的准确率添加到Accuracy_List_stable列表
        AUPR_List_stable.append(PRC_test)# 将当前测试集的PRC添加到AUPR_List_stable列表
        Recall_List_stable.append(Recall_test)# 将当前测试集的召回率添加到Recall_List_stable列表
        Precision_List_stable.append(Precision_test)# 将当前测试集的精确率添加到Precision_List_stable列表
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)
    

def ensemble_run_MIF_model(SEED, DATASET, K_Fold, device):# 集成运行MIF模型的函数

    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    '''Data Preparation'''
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()  
    
          
    

    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
    
    # test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                                  collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

    model = []
    for i in range(K_Fold):
        model.append(MIFDTI().to(device))
        '''MIF-DTI K-Fold train process is necessary'''
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + f'/valid_best_checkpoint-{device}.pth', map_location=torch.device(device)))   #加载对应权重
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'MIFDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=device, epsilon=hp.loss_epsilon)

#   testdataset_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
#       model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", save=True, FOLD_NUM=K_Fold)
    
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold, MIF=True)
    
    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)



    