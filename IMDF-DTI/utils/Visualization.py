import matplotlib.pyplot as plt
import numpy as np
import os

def plot_feature_importance(weights_dict, save_path):
    """
    IMDF-DTI 专用可视化工具
    功能：将 1D/2D 掩码权重输出为高分辨率热力图，并自动分类归档
    """
    # 1. 路径预处理：确保文件夹存在且格式正确
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # 设置全局字体样式（可选，增强学术感）
    plt.rcParams['font.family'] = 'serif'

    for name, weight in weights_dict.items():
        # 2. 画布比例优化：15x3 的比例更适合展示药物原子或蛋白长序列
        plt.figure(figsize=(15, 3))
        
        # 3. 配色方案：使用 'YlOrRd' (黄-橙-红)
        # 这种配色在打印和论文查阅时比纯 'Reds' 更有层次感，能清晰分辨 0.1-0.3 之间的微小差异
        im = plt.imshow(weight.reshape(1, -1), aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        
        # 4. 细节修饰
        cbar = plt.colorbar(im, pad=0.02)
        cbar.set_label('Sensitivity Score', fontsize=10)
        
        plt.title(f"IMDF-DTI Interpretability: {name.replace('_', ' ').title()}", fontsize=12, pad=15)
        plt.xlabel("Atom / Amino Acid Index (Spatial-Sequence Dimension)", fontsize=10)
        plt.yticks([])  # 隐藏纵轴索引，因为是 1D 展开
        
        # 5. 高质量保存
        # dpi=300 是 SCI 论文投稿的最低标准
        # bbox_inches='tight' 确保标题和颜色条不会被切掉
        save_filename = os.path.join(save_path, f"{name}.png")
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"📊 解释性报告已成功导出至: {os.path.abspath(save_path)}")

