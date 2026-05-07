# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool
                                )#GATConv：图注意力卷积层，基于注意力机制捕捉节点间的依赖关系。SAGPooling：自注意力图池化层，对图节点进行动态池化（保留重要节点）。LayerNorm：适配图数据的层归一化（按 batch 内的图独立归一化）。global_add_pool：全局图池化层，将每个图的节点特征进行求和，得到图级别的表示。
from config import hyperparameter

class MIF_conv_block(nn.Module):# MIF卷积块类，用于构建MIF模型中的卷积层
    """
    MIF卷积块：包含GATConv层，LayerNorm和SAGPooling
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):# 初始化MIF卷积块
        super(MIF_conv_block, self).__init__()# 调用父类初始化方法
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 图注意力卷积层
        self.conv = GATConv(self.in_channels, self.out_channels//self.num_heads, self.num_heads, dropout=self.dropout)
        # 层归一化
        self.norm = LayerNorm(self.in_channels)
        # 自注意力图池化
        self.readout = SAGPooling(self.out_channels, min_score=-1)

    def forward(self, x, edge_index, batch, edge_attr=None):# 前向传播函数，定义数据在网络中的流动
        # 归一化后经过ELU激活函数
        x = F.elu(self.norm(x, batch))
        # 经过GAT卷积
        x = self.conv(x, edge_index, edge_attr)
        # 池化操作，获取全局图嵌入
        x, _, _, x_batch, _, _ = self.readout(x, edge_index, edge_attr=edge_attr, batch=batch)# 对节点特征x进行池化操作，返回池化后的节点特征x、注意力权重、None、新的批次索引x_batch、None、None
        global_graph_emb = global_add_pool(x, x_batch)# 对池化后的节点特征x进行全局图池化，返回每个图的全局表示global_graph_emb
        return x, global_graph_emb


class MIFBlock(nn.Module):# MIF核心模块类，用于处理药物和蛋白质的相互作用
    """
    MIF核心模块：处理药物和蛋白质的相互作用
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=5, dropout=0.4):# 初始化MIF核心模块
        super(MIFBlock, self).__init__()
        
        self.hidden_channels = out_channels // (num_heads*2)
        # 药物内部卷积
        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)
        # 蛋白质内部卷积
        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.3)
        # 交互卷积（药物与蛋白质之间）
        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=dropout)
        # 层归一化
        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)
        # 池化层
        self.drug_pool = GATConv(out_channels, out_channels//num_heads, num_heads)
        self.prot_pool = SAGPooling(out_channels, min_score=-1)
        # self.prot_pool = GATConv(out_channels, out_channels//num_heads, num_heads)

    def forward(self, atom_x, atom_edge_index, bond_x, atom_batch, \
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):# 前向传播函数，定义数据在网络中的流动
        
        # 保存残差连接的输入
        atom_x_res = atom_x
        aa_x_res = aa_x

        # 药物内部特征提取
        atom_intra_x = self.drug_conv(atom_x, atom_edge_index, bond_x)
        # 药物-蛋白质交互特征提取
        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])
        # 拼接内部和交互特征
        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)
        # 归一化和激活
        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))

        # 蛋白质内部特征提取
        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)
        # 蛋白质-药物交互特征提取
        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)
        # 拼接内部和交互特征
        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)
        # 归一化和激活
        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))

        # 池化操作
        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        aa_x, _, _, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        # aa_x, aa_edge_index, aa_edge_attr, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        # aa_x = self.prot_pool(aa_x, aa_edge_index, aa_edge_attr)
        
        # 残差连接和Dropout
        atom_x = F.dropout(atom_x_res+F.elu(atom_x), 0.1, self.training)
        aa_x = F.dropout(aa_x_res+F.elu(aa_x), 0.1, self.training)
        
        # 全局池化，获取图级别的表示
        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr

class MIFBlock_1D(nn.Module):# 1D MIF模块类，用于处理序列数据（如药物SMILES和蛋白质序列）
    """
    1D MIF模块：处理序列数据（如药物SMILES和蛋白质序列）
    """
    def __init__(self, input_dim=200, conv=50, drug_kernel=[4, 6, 8], prot_kernel=[4, 8, 12]):# 初始化1D MIF模块
        super(MIFBlock_1D, self).__init__()# 调用父类nn.Module的初始化方法
        self.attention_dim = conv * 4# 注意力层的维度，等于卷积层输出通道数的4倍
        self.mix_attention_head = 5# 混合注意力头数

        # CNN层，用于提取局部特征
        self.Drug_CNNs = get_CNNs(input_dim, conv, drug_kernel)
        self.Protein_CNNs = get_CNNs(input_dim, conv, prot_kernel)

        # 多头注意力层，用于特征融合
        self.mix_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head, batch_first=True, dropout=0.3)

    def forward(self, drugembed, proteinembed):

        # 调整维度以适应CNN: [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len] 
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # 经过CNN提取特征
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # 调整维度以适应Attention: [batch_size, embed_dim, seq_len] -> [batch_size, seq_len, embed_dim]
        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)

        # 交叉注意力 (Cross Attention)
        drug_att, _ = self.mix_attention_layer(drugConv, proteinConv, proteinConv)
        protein_att, _ = self.mix_attention_layer(proteinConv, drugConv, drugConv)

        # 残差连接
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        # 最大池化
        drugPool, _ = torch.max(drugConv, dim=1)
        proteinPool, _ = torch.max(proteinConv, dim=1)

        return drugConv, proteinConv, drugPool, proteinPool


class MIFDTI(nn.Module):# MIF-DTI主模型类，用于药物-蛋白质交互预测
    """
    MIF-DTI主模型
    """
    def __init__(self, depth=3, device='cuda:0'):# 初始化MIF-DTI模型
        super(MIFDTI, self).__init__()# 调用父类nn.Module的初始化方法

        self.drug_in_channels = 43# 药物原子特征维度
        self.prot_in_channels = 33# 蛋白质氨基酸特征维度
        self.prot_evo_in_channels = 1280# 蛋白质进化信息维度
        self.hidden_channels = 200# 隐藏层维度
        self.depth = depth# 编码器层数
        self.device = device# 设备（CPU或GPU）

        # 分子特征编码器
        # 原子类型嵌入
        self.atom_type_encoder = Embedding(20, self.hidden_channels)
        # 原子特征MLP
        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        # 化学键嵌入
        self.bond_encoder = Embedding(10, self.hidden_channels)

        # 蛋白质特征编码器
        # 进化信息MLP
        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        # 氨基酸特征MLP
        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 

        # 编码器模块列表（图神经网络部分）
        self.blocks = nn.ModuleList([MIFBlock() for _ in range(depth)])

        # 序列嵌入
        self.drug_seq_emb = nn.Embedding(65, self.hidden_channels, padding_idx=0)
        self.prot_seq_emb = nn.Embedding(26, self.hidden_channels, padding_idx=0)
        # 编码器模块列表（序列部分）
        self.blocks_1D = nn.ModuleList([MIFBlock_1D() for _ in range(depth)])

        # 最终的预测层（RESCAL）
        # self.attn = RESCAL(self.hidden_channels, self.depth*2)
        self.attn = RESCAL_Hybrid(self.hidden_channels, self.depth*2, alpha=0.5)
        
        # 2D 序列分支掩码
        self.mask_2d_drug = FeatureMask(self.hidden_channels)
        self.mask_2d_prot = FeatureMask(self.hidden_channels)
        # 1D 序列分支掩码
        self.mask_1d_drug = FeatureMask(self.hidden_channels)
        self.mask_1d_prot = FeatureMask(self.hidden_channels)
        
        # 动态门控单元
        self.dfgu_2d = DFGU()
        self.dfgu_1d = DFGU()
        # self.attn = PoolAttention(self.hidden_channels)

        self.to(device)


    def forward(self,data):

        # 获取分子数据
        atom_x, atom_x_feat, smiles_x, atom_edge_index, bond_x, mol_node_levels = \
            data.mol_x, data.mol_x_feat, data.mol_smiles_x, data.mol_edge_index, data.mol_edge_attr, data.mol_node_levels
        # 获取蛋白质数据 (氨基酸)
        aa_x, aa_evo_x, seq_x, aa_edge_index, aa_edge_weight = \
            data.prot_node_aa, data.prot_node_evo, data.prot_seq_x, data.prot_edge_index, data.prot_edge_weight, \
        # 获取Batch信息
        atom_batch, aa_batch = data.mol_x_batch, data.prot_node_aa_batch
        # 双向图边索引
        m2p_edge_index = data.m2p_edge_index

        # 分子特征初始化
        atom_x = self.atom_type_encoder(atom_x.squeeze()) + self.atom_feat_encoder(atom_x_feat)
        bond_x = self.bond_encoder(bond_x)
                
        # 蛋白质特征初始化
        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)
        aa_edge_attr = rbf(aa_edge_weight, D_max=1.0, D_count=self.hidden_channels, device=self.device)

        atom_x, m_2d_d = self.mask_2d_drug(atom_x)
        aa_x, m_2d_p = self.mask_2d_prot(aa_x)
        # 消融实验：禁用 2D 掩码过滤
        # atom_x, m_2d_d = self.mask_2d_drug(atom_x) 
        # aa_x, m_2d_p = self.mask_2d_prot(aa_x)

        # # （改）模拟全通掩码（全 1），确保 DFGU 逻辑不崩溃
        # m_2d_d = torch.ones(self.hidden_channels).to(self.device)
        # m_2d_p = torch.ones(self.hidden_channels).to(self.device)

        w_2d = self.dfgu_2d(m_2d_d, m_2d_p)
        # 原代码: w_2d = self.dfgu_2d(m_2d_d, m_2d_p)
        # 消融修改: 强行固定为 0.5/0.5 平分
        # w_2d = torch.tensor([0.5, 0.5], device=self.device)


        # 编码过程 (Encoding)
        drug_repr = []
        prot_repr = []
        # 图神经网络部分
        for i in range(self.depth):
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            drug_global_repr = atom_x[mol_node_levels==2]
            # drug_repr.append(drug_global_repr)
            # prot_repr.append(prot_global_repr)
            drug_repr.append(atom_x[mol_node_levels==2] * w_2d[0])
            prot_repr.append(prot_global_repr * w_2d[1])

        # 序列部分
        atom_x_seq = self.drug_seq_emb(smiles_x)
        aa_x_seq = self.prot_seq_emb(seq_x)
        atom_x_seq, m_1d_d = self.mask_1d_drug(atom_x_seq)
        aa_x_seq, m_1d_p = self.mask_1d_prot(aa_x_seq)
        # 消融实验：禁用 1D 掩码过滤
        # atom_x_seq, m_1d_d = self.mask_1d_drug(atom_x_seq)
        # aa_x_seq, m_1d_p = self.mask_1d_prot(aa_x_seq)

        # # 模拟全通掩码（全 1）
        # m_1d_d = torch.ones(self.hidden_channels).to(self.device)
        # m_1d_p = torch.ones(self.hidden_channels).to(self.device)
        # 计算 1D 模态的样本特异性权重 [cite: 63]
        w_1d = self.dfgu_1d(m_1d_d, m_1d_p)
        # 原代码: w_1d = self.dfgu_1d(m_1d_d, m_1d_p)
        # 消融修改: 强行固定为 0.5/0.5 平分
        # w_1d = torch.tensor([0.5, 0.5], device=self.device)
        for i in range(self.depth):
            out_seq = self.blocks_1D[i](atom_x_seq, aa_x_seq)
            atom_x_seq, aa_x_seq, drug_seq_pool, prot_seq_pool = out_seq
            drug_repr.append(drug_seq_pool * w_1d[0]) # 应用药物 1D 权重 [cite: 63]
            prot_repr.append(prot_seq_pool * w_1d[1]) # 应用蛋白 1D 权重 [cite: 63]

        # 堆叠所有深度的表示
        drug_repr = torch.stack(drug_repr, dim=-2)
        prot_repr = torch.stack(prot_repr, dim=-2)

        # 联合注意力机制 (Co-attn) 计算最终得分
        # scores = self.attn(drug_repr, prot_repr)
        # --- 📍 插入逻辑：综合 1D 和 2D 的敏感度权重作为先验 ---
        # --- 📍 插入逻辑：综合 1D 和 2D 的敏感度权重作为先验 ---
        # 将 1D 和 2D 模态权重取平均，作为该样本对药物和靶点的整体敏感度先验
        s_drug_avg = (w_2d[0] + w_1d[0]) / 2
        s_prot_avg = (w_2d[1] + w_1d[1]) / 2

        # 调用混合动力算子，传入先验权重 [cite: 87, 110]
        scores = self.attn(drug_repr, prot_repr, s_drug=s_drug_avg, s_prot=s_prot_avg)

        return scores
    def get_explainability_weights(self):
        """
        提取推理阶段的可解释性权重 (Innovation Point 3) [cite: 88, 92]
        返回: 2D药物、2D蛋白、1D药物、1D蛋白的掩码分值
        """
        with torch.no_grad():
            weights = {
                "mask_2d_drug": torch.sigmoid(self.mask_2d_drug.delta).cpu().numpy(),
                "mask_2d_prot": torch.sigmoid(self.mask_2d_prot.delta).cpu().numpy(),
                "mask_1d_drug": torch.sigmoid(self.mask_1d_drug.delta).cpu().numpy(),
                "mask_1d_prot": torch.sigmoid(self.mask_1d_prot.delta).cpu().numpy()
            }
        return weights

def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):# 从Batch中获取药物-蛋白质交互边索引
    """
    从Batch中获取药物-蛋白质交互边索引
    """

    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)  # (num_a_nodes, num_b_nodes) 的bool矩阵
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list
