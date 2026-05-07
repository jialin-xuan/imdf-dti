# -*- coding:utf-8 -*-
import torch
import numpy as np
import os
import joblib
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             accuracy_score, precision_score, recall_score)
import torch_geometric.loader as pyg_loader
from datetime import datetime

# 导入项目原有模块
from model import MIFDTI
from config import hyperparameter
from utils.DataPrepare import shuffle_dataset
from utils import ProteinMoleculeDataset

def run_ensemble():
    hp = hyperparameter()
    
    # 📍 配置区域
    SEED = 114514 
    DATASET = "Davis" #"DrugBank" "Davis" "BIOSNAP"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEV_STR = str(DEVICE).replace(":", "_")

        # 手动指定要集成的模型编号
    # loaded_fold_ids = [1, 2, 3, 4, 5]

    # fold_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)
    # fold_weights = fold_weights / fold_weights.sum()

    
    print(f"⭐ IMDF-DTI 深度集成系统 | 数据集: {DATASET} | 设备: {DEVICE}")

    # 1. 数据对齐
    print("步骤 [1/7]: 正在同步数据切分逻辑...")
    dir_input = f'./DataSets/{DATASET}.txt'
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    
    data_list = shuffle_dataset(data_list, SEED)
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print(f"✅ 数据对齐完成，独立测试集样本数: {len(test_data_list)}")

    # 2. 加载图字典
    print("步骤 [2/7]: 正在加载预处理字典...")
    try:
        protein_path = f'./DataSets/Preprocessed/{DATASET}-protein-new.pkl'
        ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi-new.pkl'
        protein_dict = joblib.load(protein_path)
        ligand_dict = joblib.load(ligand_path)
        print("   - 字典文件已成功载入内存")
    except Exception as e:
        print(f"❌ 警告：字典加载失败: {e}")
        return

    # 3. 初始化数据加载器
    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=DEVICE)
    test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, 
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

    # 4. 初始化模型并提取静态掩码
    print("步骤 [3/7]: 正在载入 5 折模型权重...")
    models = []
    all_mask_weights = {'mask_2d_drug': [], 'mask_2d_prot': [], 'mask_1d_drug': [], 'mask_1d_prot': []}
    
    # 动态门控权重缓存
    batch_dfgu_2d = []
    batch_dfgu_1d = []
    
    for i in range(1, 6):
        m = MIFDTI(depth=3, device=DEVICE).to(DEVICE)
        path = f"./{DATASET}/{i}/valid_best_checkpoint-{DEV_STR}.pth"
        
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.eval()
            models.append(m)
            
            # 提取 4 路特征掩码 (模型中 get_explainability_weights 定义的方法)
            exp_w = m.get_explainability_weights()
            for k in all_mask_weights.keys():
                all_mask_weights[k].append(exp_w[k])
            print(f"   [Fold {i}] 权重载入成功")
        else:
            print(f"   [Fold {i}] ❌ 忽略缺失文件: {path}")

    if not models:
        print("❌ 错误：未加载到任何权重，程序终止。")
        return

    # 5. 执行集成推理
    print(f"步骤 [4/7]: 正在执行多模型联合表决 (Soft Voting)...")
    all_labels, all_probs = [], []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="推理进度"):
            data = data.to(DEVICE)
            model_probs = []
            
            for m in models:
                outputs = m(data)
                probs = F.softmax(outputs, dim=1)[:, 1]
                model_probs.append(probs.cpu().numpy())
                
                # 📍 关键：在这里提取 latest_weights，因为 forward 刚执行完，属性已存在
                if hasattr(m.dfgu_2d, 'latest_weights'):
                    batch_dfgu_2d.append(m.dfgu_2d.latest_weights.cpu().numpy())
                    batch_dfgu_1d.append(m.dfgu_1d.latest_weights.cpu().numpy())
            
            ensemble_prob = np.mean(model_probs, axis=0)
            # model_probs = np.stack(model_probs, axis=0)
            # ensemble_prob = np.sum(model_probs * fold_weights[:, None], axis=0)
            all_probs.extend(ensemble_prob)
            all_labels.extend(data.cls_y.cpu().numpy())

    # 6. 计算全量指标
    print("步骤 [5/7]: 正在结算全量集成指标...")
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.50).astype(int)
    
    final_auc = roc_auc_score(y_true, y_prob)
    final_prc = average_precision_score(y_true, y_prob)
    final_acc = accuracy_score(y_true, y_pred)
    final_prec = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)

    # 7. 结果保存
    print("步骤 [6/7]: 正在保存结果文件...")
    save_dir = f"./{DATASET}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 写入文件报告
    res_path = os.path.join(save_dir, "ensemble_results.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(res_path, "a") as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"数据集: {DATASET} | 随机种子: {SEED}\n")
        f.write(f"ROC-AUC  : {final_auc:.5f}\n")
        f.write(f"PRC/AUPR : {final_prc:.5f}\n")
        f.write(f"Accuracy : {final_acc:.5f}\n")
        f.write(f"Precision: {final_prec:.5f}\n")
        f.write(f"Recall   : {final_recall:.5f}\n")
        f.write(f"{'='*40}\n")

    # 保存原始预测数据
    np.save(os.path.join(save_dir, "ensemble_y_prob.npy"), y_prob)
    np.save(os.path.join(save_dir, "ensemble_y_true.npy"), y_true)

    # 步骤 7: 导出集成平均权重数据
    print("步骤 [7/7]: 正在导出集成平均解释性权重...")
    ensemble_explain = {
        'mask_2d_drug': np.mean(all_mask_weights['mask_2d_drug'], axis=0),
        'mask_2d_prot': np.mean(all_mask_weights['mask_2d_prot'], axis=0),
        'mask_1d_drug': np.mean(all_mask_weights['mask_1d_drug'], axis=0),
        'mask_1d_prot': np.mean(all_mask_weights['mask_1d_prot'], axis=0),
        'dfgu_2d_mean': np.mean(batch_dfgu_2d, axis=0) if batch_dfgu_2d else None,
        'dfgu_1d_mean': np.mean(batch_dfgu_1d, axis=0) if batch_dfgu_1d else None
    }
    np.save(os.path.join(save_dir, "ensemble_weights.npy"), ensemble_explain)

    # 🏆 终端战报输出 (满足您的格式要求)
    print("\n" + "🏆" * 20)
    print(f"🔥 {DATASET} 最终集成战报 🔥")
    print(f"   ROC-AUC  : {final_auc:.5f}")
    print(f"   PRC/AUPR : {final_prc:.5f}")
    print(f"   Accuracy : {final_acc:.5f}")
    print(f"   Precision: {final_prec:.5f}")
    print(f"   Recall   : {final_recall:.5f}")
    print(f"✅ 所有结果、原始预测数据及集成权重已保存至: {save_dir}")
    print("🏆" * 20)

if __name__ == "__main__":
    run_ensemble()