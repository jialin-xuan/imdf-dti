# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import os  # 📍 必须添加，用于路径处理
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

# ... (test_precess 和 test_MIF_precess 函数保持不变) ...

def test_MIF_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    # 切换模型为评估模式
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()

    test_losses = []
    Y, P, S = [], [], []
    
    with torch.no_grad():
        for i, data in pbar:
            '''数据准备：直接将整个 Batch 移动到设备'''
            # 📍 修正：不再尝试解包 data
            data = data.to(DEVICE)
            
            # 兼容单模型和集成模型逻辑
            if isinstance(MODEL, list):
                # 初始化预测分数为 batch_size x 类别数
                # 这里的 2 需要根据你的分类类别数确定
                predicted_scores = torch.zeros(len(data.cls_y), 2).to(DEVICE)
                for m_idx in range(len(MODEL)):
                    # 📍 修正：传入 data 对象，并显式设置 training=False 触发 2.1 架构逻辑
                    predicted_scores = predicted_scores + MODEL[m_idx](data)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                # 📍 修正：直接传入 data 对象，并设置 training=False
                predicted_scores = MODEL(data)

            # 获取标签：在 PyG Batch 中，标签存放在 .cls_y 属性里
            labels = data.cls_y
            
            # 计算 Loss
            loss = LOSS(predicted_scores, labels)
            
            # 采集预测结果
            correct_labels = labels.to('cpu').data.numpy()
            predicted_probs = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_probs, axis=1)
            positive_scores = predicted_probs[:, 1] # 提取正类概率

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(positive_scores)
            test_losses.append(loss.item())

    # 计算指标
    Precision = precision_score(Y, P, zero_division=0)
    Recall = recall_score(Y, P, zero_division=0)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)

    # 📍 确保返回 8 个值，顺序与 test_model 接收处一致
    # 注意：test_model 接收的是 T, P (预测概率), loss... 而非预测标签
    # 因此这里返回 S (预测分数/概率)
    return Y, S, np.average(test_losses), Accuracy, Precision, Recall, AUC, PRC
     # 实际代码请保留你原有的 test_MIF_precess 内容

def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    # 📍 修正：直接重用 test_MIF_precess 的逻辑，不要重复写循环耗尽 pbar
    return test_MIF_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM)
# def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
#     if isinstance(MODEL, list):
#         for item in MODEL:
#             item.eval()
#     else:
#         MODEL.eval()
#     test_losses = []
#     Y, P, S = [], [], []
#     with torch.no_grad():
#         for i, data in pbar:
#             '''data preparation '''
#             data = data.to(DEVICE)
#             if isinstance(MODEL, list):
#                 predicted_scores = torch.zeros(2).to(DEVICE)
#                 for i in range(len(MODEL)):
#                     predicted_scores = predicted_scores + \
#                         MODEL[i](data)
#                 predicted_scores = predicted_scores / FOLD_NUM
#             else:
#                 predicted_scores = MODEL(data)
#             labels = data.cls_y
#             loss = LOSS(predicted_scores, labels)
#             correct_labels = labels.to('cpu').data.numpy()
#             predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
#             predicted_labels = np.argmax(predicted_scores, axis=1)
#             predicted_scores = predicted_scores[:, 1]

#             Y.extend(correct_labels)
#             P.extend(predicted_labels)
#             S.extend(predicted_scores)
#             test_losses.append(loss.item())
#     Precision = precision_score(Y, P)
#     Recall = recall_score(Y, P)
#     AUC = roc_auc_score(Y, S)
#     tpr, fpr, _ = precision_recall_curve(Y, S)
#     PRC = auc(fpr, tpr)
#     Accuracy = accuracy_score(Y, P)
#     test_loss = np.average(test_losses)
#     return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC

def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1, MIF=False):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    
    # 执行测试流程
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_MIF_precess(MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM) if MIF else test_precess(MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    
    # 保存预测结果文本
    if save:
        if FOLD_NUM == 1:
            filepath = os.path.join(save_path, "{}_{}_prediction.txt".format(DATASET, dataset_class))
        else:
            filepath = os.path.join(save_path, "{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class))
        
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
                
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)

    # --- 📍 核心修改：IMDF-DTI 解释报告输出逻辑 ---
    # 限制条件：必须是MIF模型、必须是单模型模式(FOLD_NUM==1)、必须是最终测试集(Test)
    if MIF and FOLD_NUM == 1 and dataset_class == "Test":
        try:
            # 1. 检查模型类型，防止集成模式报错
            if isinstance(MODEL, list):
                print("Notice: Interpretability maps are generated from the first model in ensemble.")
                target_model = MODEL[0]
            else:
                target_model = MODEL

            print(f"🚀 Generating interpretability reports for {DATASET}...")
            
            # 2. 提取掩码权重
            importance_weights = target_model.get_explainability_weights()
            
            # 3. 动态构建保存路径：例如 ./Davis/1/interpretability_report/
            interp_path = os.path.join(save_path, "interpretability_report")
            
            # 4. 调用可视化函数
            from utils.Visualization import plot_feature_importance
            plot_feature_importance(importance_weights, save_path=interp_path)
            
            print(f"✅ Interpretability reports (Heatmaps) saved to: {interp_path}")
            
        except Exception as e:
            print(f"⚠️ Interpretability output skipped or failed: {e}")

    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test