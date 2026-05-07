# -*- coding:utf-8 -*-
import os
import joblib
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import torch_geometric.loader as pyg_loader

# 导入项目中定义的模块
from model import MIFDTI
from config import hyperparameter
from utils import ProteinMoleculeDataset


def build_target_loader(target_name, hp, device):
    """
    构建目标数据集的 DataLoader
    """
    print(f"步骤 [1/5]: 正在加载目标数据集 {target_name} 的图字典...")
    protein_path = f'./DataSets/Preprocessed/{target_name}-protein-new.pkl'
    ligand_path = f'./DataSets/Preprocessed/{target_name}-ligand-hi-new.pkl'

    if not os.path.exists(protein_path):
        raise FileNotFoundError(f"未找到蛋白字典文件: {protein_path}")
    if not os.path.exists(ligand_path):
        raise FileNotFoundError(f"未找到配体字典文件: {ligand_path}")

    protein_dict = joblib.load(protein_path)
    ligand_dict = joblib.load(ligand_path)

    data_txt_path = f'./DataSets/{target_name}.txt'
    if not os.path.exists(data_txt_path):
        raise FileNotFoundError(f"未找到目标数据集文本文件: {data_txt_path}")

    with open(data_txt_path, "r", encoding="utf-8") as f:
        target_data_list = f.read().strip().split('\n')

    print(f"步骤 [2/5]: 正在构建目标数据集 DataLoader...")
    test_dataset = ProteinMoleculeDataset(
        target_data_list,
        ligand_dict,
        protein_dict,
        device=device
    )

    test_loader = pyg_loader.DataLoader(
        test_dataset,
        batch_size=hp.Batch_size,
        shuffle=False,
        follow_batch=['mol_x', 'clique_x', 'prot_node_aa']  # 必须与训练保持一致
    )

    print(f"✅ 目标数据集加载完成，共 {len(test_dataset)} 个样本")
    return test_loader


def get_checkpoint_path(source_name, fold_index, device):
    """
    自动尝试多种 checkpoint 命名方式，提高兼容性
    """
    device_str = str(device).replace(':', '_')

    candidate_paths = [
        f'./{source_name}/{fold_index}/valid_best_checkpoint-{device_str}.pth',
        f'./{source_name}/{fold_index}/valid_best_checkpoint-cuda_0.pth',
        f'./{source_name}/{fold_index}/valid_best_checkpoint-cpu.pth',
        f'./{source_name}/{fold_index}/valid_best_checkpoint.pth',
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    return None


def load_model_checkpoint(model, checkpoint_path, device):
    """
    加载模型权重，兼容普通 state_dict 和 DataParallel 保存格式
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 兼容 {"model_state_dict": ...} 或直接 state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # 兼容 DataParallel 保存的 module. 前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model


def evaluate_one_fold(source_name, target_name, fold_index, test_loader, device):
    """
    单折跨数据集验证
    """
    print(f"\n🔍 启动单折跨数据集验证：[{source_name}] fold={fold_index} -> [{target_name}]")

    print(f"步骤 [3/5]: 正在加载源数据集 {source_name} 第 {fold_index} 折模型权重...")
    model = MIFDTI().to(device)

    checkpoint_path = get_checkpoint_path(source_name, fold_index, device)
    if checkpoint_path is None:
        print(f"❌ 错误：未找到第 {fold_index} 折模型权重文件")
        return None

    print(f"尝试加载路径: {checkpoint_path}")
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    print(f"步骤 [4/5]: 正在目标域执行推理...")
    all_probs, all_labels = [], []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Fold-{fold_index} Testing"):
            data = data.to(device)
            output = model(data)

            # 默认假设模型输出为 [B, 2]，并使用 CrossEntropyLoss 训练
            probs = torch.softmax(output, dim=1)[:, 1]

            labels = data.cls_y.view(-1)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    print(f"步骤 [5/5]: 正在计算指标...")
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0)
    }

    print("\n" + "⭐" * 20)
    print(f"🔥 单折跨数据集验证结果 ({source_name} fold {fold_index} -> {target_name})")
    for k, v in metrics.items():
        print(f"   {k:10}: {v:.5f}")
    print("⭐" * 20)

    # 保存单折结果
    save_dir = f"./Cross_Val_Results/{source_name}_to_{target_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"fold_{fold_index}_results.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Source Dataset: {source_name}\n")
        f.write(f"Target Dataset: {target_name}\n")
        f.write(f"Fold Index: {fold_index}\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.5f}\n")

    print(f"✅ 单折结果已保存至: {save_path}")
    return metrics


def summarize_five_folds(source_name, target_name, all_fold_metrics):
    """
    汇总五折结果并保存
    """
    if len(all_fold_metrics) == 0:
        print("❌ 没有可汇总的折结果")
        return

    metric_names = ['ROC-AUC', 'AUPRC', 'Accuracy', 'Precision', 'Recall']
    save_dir = f"./Cross_Val_Results/{source_name}_to_{target_name}"
    os.makedirs(save_dir, exist_ok=True)

    summary_path = os.path.join(save_dir, "five_fold_summary.txt")

    print("\n" + "=" * 60)
    print(f"📊 五折跨数据集验证汇总结果 ({source_name} -> {target_name})")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Five-Fold Cross-Dataset Validation Summary\n")
        f.write(f"Source Dataset: {source_name}\n")
        f.write(f"Target Dataset: {target_name}\n")
        f.write("=" * 60 + "\n\n")

        for fold_result in all_fold_metrics:
            fold_id = fold_result["Fold"]
            print(f"Fold {fold_id}:")
            f.write(f"Fold {fold_id}:\n")
            for metric_name in metric_names:
                value = fold_result[metric_name]
                print(f"   {metric_name:10}: {value:.5f}")
                f.write(f"   {metric_name}: {value:.5f}\n")
            print("-" * 40)
            f.write("-" * 40 + "\n")

        print("Mean ± Std:")
        f.write("\nMean ± Std:\n")

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)

            line = f"{metric_name:10}: {mean_val:.5f} ± {std_val:.5f}"
            print(line)
            f.write(line + "\n")

    print("=" * 60)
    print(f"✅ 五折汇总结果已保存至: {summary_path}")


def run_cross_dataset_validation(source_name="BIOSNAP",
                                 target_name="Davis",
                                 mode="single",
                                 fold_index=1):
    """
    跨数据集验证统一入口

    参数:
        source_name: 源数据集名称（训练权重来源）
        target_name: 目标数据集名称（测试集）
        mode: "single" 表示单折验证, "five_fold" 表示五折验证
        fold_index: 单折模式下指定折号
    """
    hp = hyperparameter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"🚀 启动跨数据集验证")
    print(f"   Source Dataset : {source_name}")
    print(f"   Target Dataset : {target_name}")
    print(f"   Mode           : {mode}")
    print(f"   Device         : {device}")
    print("=" * 70)

    # 目标域数据只加载一次
    try:
        test_loader = build_target_loader(target_name, hp, device)
    except Exception as e:
        print(f"❌ 构建目标数据集失败: {e}")
        return None

    if mode == "single":
        return evaluate_one_fold(
            source_name=source_name,
            target_name=target_name,
            fold_index=fold_index,
            test_loader=test_loader,
            device=device
        )

    elif mode == "five_fold":
        all_fold_metrics = []

        for fold in range(1, 6):
            metrics = evaluate_one_fold(
                source_name=source_name,
                target_name=target_name,
                fold_index=fold,
                test_loader=test_loader,
                device=device
            )
            if metrics is not None:
                metrics["Fold"] = fold
                all_fold_metrics.append(metrics)

        summarize_five_folds(source_name, target_name, all_fold_metrics)
        return all_fold_metrics

    else:
        print(f"❌ 不支持的 mode: {mode}，请使用 'single' 或 'five_fold'")
        return None


if __name__ == "__main__":
    # =========================
    # 用法 1：单折跨数据集验证
    # =========================
    run_cross_dataset_validation(#"DrugBank" "Davis" "BIOSNAP"
        source_name="BIOSNAP",
        target_name="Davis",
        mode="single",
        fold_index=3
    )

    # =========================
    # 用法 2：五折跨数据集验证
    # =========================
    # run_cross_dataset_validation(#"DrugBank" "Davis" "BIOSNAP"
    #     source_name="BIOSNAP",
    #     target_name="Davis",
    #     mode="five_fold"
    # )