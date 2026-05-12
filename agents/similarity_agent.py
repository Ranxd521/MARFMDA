"""
Similarity Agent
基于相似度矩阵进行推理的 Agent
重构版本：只负责特征提取，不调用LLM
"""
import numpy as np
from typing import Dict, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimilarityAgent:
    """相似度推理 Agent"""
    
    def __init__(self, data_loader):
        """
        初始化 Similarity Agent
        
        Args:
            data_loader: DataLoader 实例
        """
        self.data_loader = data_loader
    
    def compute_evidence(self, mirna_idx: int, disease_idx: int) -> Dict:
        """
        提取相似度特征（不调用LLM）
        
        Args:
            mirna_idx: miRNA 索引
            disease_idx: 疾病索引
        
        Returns:
            包含特征的字典
        """
        # 提取相似度矩阵特征
        mm_seq = self.data_loader.mm_seq
        dd_sem = self.data_loader.dd_sem
        
        # 获取 miRNA 的相似度行（与其他 miRNA 的相似度）
        mirna_similarity_row = mm_seq[mirna_idx, :]
        
        # 获取 disease 的相似度行（与其他 disease 的相似度）
        disease_similarity_row = dd_sem[disease_idx, :]
        
        # 获取与当前 miRNA 相似的 miRNA 关联的疾病
        similar_mirnas = np.argsort(mirna_similarity_row)[::-1][:10]  # top-10
        similar_mirnas_scores = mirna_similarity_row[similar_mirnas]
        
        # 使用训练集矩阵（修复数据泄露：排除测试集边）
        train_md_matrix = self.data_loader.get_train_matrix()
        
        # 计算这些相似 miRNA 与当前疾病的关联（只考虑训练集）
        similar_mirnas_disease_assoc = train_md_matrix[similar_mirnas, disease_idx]
        
        # 获取与当前疾病相似的疾病关联的 miRNA
        similar_diseases = np.argsort(disease_similarity_row)[::-1][:10]  # top-10
        similar_diseases_scores = disease_similarity_row[similar_diseases]
        
        # 计算当前 miRNA 与这些相似疾病的关联（只考虑训练集）
        mirna_similar_diseases_assoc = train_md_matrix[mirna_idx, similar_diseases]
        
        # 计算统计特征
        mirna_assoc_count = int(np.sum(similar_mirnas_disease_assoc))
        disease_assoc_count = int(np.sum(mirna_similar_diseases_assoc))
        top_mirna_sim = float(np.mean(similar_mirnas_scores[:5])) if len(similar_mirnas_scores) > 0 else 0.0
        top_disease_sim = float(np.mean(similar_diseases_scores[:5])) if len(similar_diseases_scores) > 0 else 0.0
        
        return {
            "agent": "similarity",
            "features": {
                "mirna_assoc_count": mirna_assoc_count,
                "disease_assoc_count": disease_assoc_count,
                "top_mirna_sim": top_mirna_sim,
                "top_disease_sim": top_disease_sim,
                "similar_mirnas": len(similar_mirnas),
                "similar_diseases": len(similar_diseases)
            },
            # 用于批处理prompt的文本描述
            "text_evidence": f"Similar miRNAs: {mirna_assoc_count}/10 associate with target disease (avg sim: {top_mirna_sim:.3f}). Similar diseases: {disease_assoc_count}/10 associate with target miRNA (avg sim: {top_disease_sim:.3f})."
        }

