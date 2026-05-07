import argparse
import torch
from RunModel import run_MIF_model,ensemble_run_MIF_model
from model import MIFDTI

# 创建命令行参数解析器
parser = argparse.ArgumentParser(
    prog='MIF-DTI',
    description='MIF-DTI is model in paper: \"multimodal information fusion method for drug-target interaction prediction\"',
    epilog='Model config set by config.py')

# 添加数据集参数，必须在["DrugBank", "Davis", "BIOSNAP", "BD2D"]中选择
parser.add_argument('dataSetName', choices=[
                    "DrugBank", "Davis", "BIOSNAP","BD2D"], help='Enter which dataset to use for the experiment')
# 添加模型参数，默认为"MIF-DTI"
parser.add_argument('-m', '--model', choices=['MIF-DTI', 'MIF-DTI-B'],
                    default='MIF-DTI', help='Which model to use, \"MIF-DTI\" is used by default')
# 添加随机种子参数，默认为114514
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
# 添加K折交叉验证参数，默认为5
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
# 添加GPU设备参数，默认为0
parser.add_argument('-g', '--gpu', type=int, default=0,
                    help='cuda number, the default is 0')

# 解析命令行参数
args = parser.parse_args()
# 设置设备，如果有GPU则使用指定的GPU，否则使用CPU
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# 如果选择的模型是MIF-DTI
if args.model == 'MIF-DTI':
    # 运行MIF-DTI模型
    run_MIF_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=MIFDTI, K_Fold=args.fold, LOSS='PolyLoss', device=device)
# 如果选择的模型是MIF-DTI-B
if args.model == 'MIF-DTI-B':
    # 运行MIF-DTI-B集成模型
    ensemble_run_MIF_model(SEED=args.seed, DATASET=args.dataSetName, K_Fold=args.fold, device=device)
