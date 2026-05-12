"""
RWR Agent (Random Walk with Restart)
基于 Random Walk with Restart 算法进行推理的 Agent
重构版本：只负责特征提取，不调用LLM
"""
import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RWRAgent:
    """Random Walk with Restart 推理 Agent"""
    
    def __init__(self, data_loader):
        """
        初始化 RWR Agent
        
        Args:
            data_loader: DataLoader 实例
        """
        self.data_loader = data_loader
    
    def compute_evidence(self, mirna_idx: int, disease_idx: int, restart_prob: float = 0.3, max_iter: int = 100) -> Dict:
        """
        提取RWR特征（不调用LLM）
        
        Args:
            mirna_idx: miRNA 索引
            disease_idx: 疾病索引
            restart_prob: 重启概率
            max_iter: 最大迭代次数
        
        Returns:
            包含特征的字典
        """
        # 使用训练集矩阵（修复数据泄露：排除测试集边）
        train_md_matrix = self.data_loader.get_train_matrix()
        
        # 构建异构网络矩阵
        # 合并 miRNA-disease, miRNA-lncRNA, disease-lncRNA 矩阵
        n_mirna = train_md_matrix.shape[0]
        n_disease = train_md_matrix.shape[1]
        n_lncrna = self.data_loader.ml_matrix.shape[1]
        
        # 构建异构网络邻接矩阵
        # [miRNA, Disease, lncRNA] 的顺序
        total_nodes = n_mirna + n_disease + n_lncrna
        
        adj_matrix = np.zeros((total_nodes, total_nodes))
        
        # miRNA-disease 关联（使用训练集矩阵）
        adj_matrix[:n_mirna, n_mirna:n_mirna+n_disease] = train_md_matrix
        adj_matrix[n_mirna:n_mirna+n_disease, :n_mirna] = train_md_matrix.T
        
        # miRNA-lncRNA 关联
        adj_matrix[:n_mirna, n_mirna+n_disease:] = self.data_loader.ml_matrix
        adj_matrix[n_mirna+n_disease:, :n_mirna] = self.data_loader.ml_matrix.T
        
        # disease-lncRNA 关联
        adj_matrix[n_mirna:n_mirna+n_disease, n_mirna+n_disease:] = self.data_loader.dl_matrix
        adj_matrix[n_mirna+n_disease:, n_mirna:n_mirna+n_disease] = self.data_loader.dl_matrix.T
        
        # 归一化（按行归一化）
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        transition_matrix = adj_matrix / row_sums
        
        # RWR: 从 miRNA 节点开始随机游走
        start_node = mirna_idx
        p = np.zeros(total_nodes)
        p[start_node] = 1.0
        
        # 迭代直到收敛
        for _ in range(max_iter):
            p_new = (1 - restart_prob) * transition_matrix.T @ p + restart_prob * p
            if np.linalg.norm(p_new - p) < 1e-6:
                break
            p = p_new
        
        # 提取到达 disease 节点的概率
        disease_start = n_mirna
        disease_end = n_mirna + n_disease
        disease_probs = p[disease_start:disease_end]
        target_disease_prob = disease_probs[disease_idx]
        
        # 计算排名
        rank = int(np.sum(disease_probs > target_disease_prob)) + 1
        max_prob = float(np.max(disease_probs))
        mean_prob = float(np.mean(disease_probs))
        
        # 构建文本描述
        text_evidence = f"RWR probability: {target_disease_prob:.6e} (rank: {rank}/{len(disease_probs)}). Max: {max_prob:.6e}, Mean: {mean_prob:.6e}."
        
        return {
            "agent": "rwr",
            "features": {
                "rwr_probability": float(target_disease_prob),
                "rank": rank,
                "max_probability": max_prob,
                "mean_probability": mean_prob,
                "total_diseases": len(disease_probs)
            },
            "text_evidence": text_evidence
        }

