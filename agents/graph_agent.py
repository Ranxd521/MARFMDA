"""
Graph Agent
基于图结构进行推理的 Agent
重构版本：只负责特征提取，不调用LLM
"""
import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_tools import compute_graph_features


class GraphAgent:
    """图结构推理 Agent"""
    
    def __init__(self, data_loader):
        """
        初始化 Graph Agent
        
        Args:
            data_loader: DataLoader 实例
        """
        self.data_loader = data_loader
    
    def compute_evidence(self, mirna_idx: int, disease_idx: int) -> Dict:
        """
        提取图结构特征（不调用LLM）
        
        修复说明:
        1. 使用 .copy() 创建矩阵副本，避免修改全局数据
        2. 仅 Mask 掉 (mirna_idx, disease_idx) 这一条特定边
        3. 确保保留 Disease 节点的其他连接 (Degree 不会变成 0)
        """
        # 1. 获取训练集矩阵 (基础数据)
        # 这是为了防止数据泄露，默认只应包含训练集的边
        train_md_matrix = self.data_loader.get_train_matrix()
        
        # 2. 创建副本进行操作 (关键步骤)
        # 必须使用 copy()，否则修改会影响到内存中的原始矩阵，干扰并行运行的其他 Agent
        md_matrix_masked = train_md_matrix.copy()
        
        # 3. 动态 Masking: 仅移除目标边
        # 逻辑：假装这条边不存在，看还能提取什么特征。
        # 如果矩阵中该位置本身就是 1 (训练集正样本)，这里必须置 0。
        # 如果本身是 0，置 0 也没影响。
        if md_matrix_masked[mirna_idx, disease_idx] != 0:
            md_matrix_masked[mirna_idx, disease_idx] = 0
            
            # ⚠️ 注意：绝对不能写成 md_matrix_masked[:, disease_idx] = 0
            # 那样会删除该疾病的所有连接，导致 Disease Degree = 0
        
        # 4. 计算图特征
        # 将处理好的 masked 矩阵传递给工具函数
        # compute_graph_features 内部会计算 np.sum(md_matrix_masked[:, disease_idx]) 作为度
        graph_features = compute_graph_features(
            md_matrix_masked,
            self.data_loader.ml_matrix,
            self.data_loader.dl_matrix,
            mirna_idx,
            disease_idx
        )
        
        # 提取关键特征
        direct_link = graph_features.get("direct_link", 0)
        path_stats = graph_features.get("path_statistics", {})
        num_paths = path_stats.get("num_paths", 0)
        max_strength = path_stats.get("max_strength", 0.0)
        mean_strength = path_stats.get("mean_strength", 0.0)
        mirna_degree = graph_features.get("mirna_degree", 0)
        disease_degree = graph_features.get("disease_degree", 0)
        
        # 构建文本描述
        text_evidence = f"Direct link: {bool(direct_link)}. "
        if num_paths > 0:
            text_evidence += f"2-hop paths: {num_paths} (max strength: {max_strength:.3f}, mean: {mean_strength:.3f}). "
        else:
            text_evidence += "No 2-hop paths. "
            
        # 此时 disease_degree 应该反映该疾病在训练集中与其他 miRNA 的连接数
        text_evidence += f"Connectivity: miRNA degree={mirna_degree}, disease degree={disease_degree}."
        
        return {
            "agent": "graph",
            "features": {
                "direct_link": int(direct_link),
                "num_paths": num_paths,
                "max_strength": float(max_strength),
                "mean_strength": float(mean_strength),
                "mirna_degree": mirna_degree,
                "disease_degree": disease_degree
            },
            "text_evidence": text_evidence
        }

