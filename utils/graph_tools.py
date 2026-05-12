"""
图工具函数
使用 numpy 进行图计算，不做预测，只提供数值特征
"""
import numpy as np
from typing import Dict, List, Tuple


def get_direct_link(md_matrix: np.ndarray, mirna_idx: int, disease_idx: int) -> float:
    """
    获取 miRNA-disease 直接链接
    
    Args:
        md_matrix: miRNA-disease 关联矩阵
        mirna_idx: miRNA 索引
        disease_idx: 疾病索引
    
    Returns:
        直接链接强度（0 或 1）
    """
    return float(md_matrix[mirna_idx, disease_idx])


def get_two_hop_paths(
    ml_matrix: np.ndarray,
    dl_matrix: np.ndarray,
    mirna_idx: int,
    disease_idx: int
) -> List[Tuple[int, float]]:
    """
    获取 miRNA → lncRNA → disease 的 2-hop 路径
    
    Args:
        ml_matrix: miRNA-lncRNA 关联矩阵
        dl_matrix: disease-lncRNA 关联矩阵
        mirna_idx: miRNA 索引
        disease_idx: 疾病索引
    
    Returns:
        路径列表，每个元素为 (lncRNA_idx, path_strength)
    """
    paths = []
    
    # miRNA → lncRNA
    mirna_lncrna = ml_matrix[mirna_idx, :]
    
    # lncRNA → disease (dl_matrix shape: (n_disease, n_lncrna))
    lncrna_disease = dl_matrix[disease_idx, :]
    
    # 计算路径强度：min(miRNA-lncRNA, lncRNA-disease)
    path_strengths = np.minimum(mirna_lncrna, lncrna_disease)
    
    # 找出所有非零路径
    nonzero_indices = np.where(path_strengths > 0)[0]
    
    for lncrna_idx in nonzero_indices:
        strength = float(path_strengths[lncrna_idx])
        paths.append((int(lncrna_idx), strength))
    
    # 按强度排序
    paths.sort(key=lambda x: x[1], reverse=True)
    
    return paths


def get_path_statistics(paths: List[Tuple[int, float]]) -> Dict:
    """
    计算路径统计信息
    
    Args:
        paths: 路径列表
    
    Returns:
        包含统计信息的字典
    """
    if len(paths) == 0:
        return {
            "num_paths": 0,
            "max_strength": 0.0,
            "min_strength": 0.0,
            "mean_strength": 0.0,
            "sum_strength": 0.0,
            "top_k_strengths": []
        }
    
    strengths = [p[1] for p in paths]
    
    return {
        "num_paths": len(paths),
        "max_strength": float(np.max(strengths)),
        "min_strength": float(np.min(strengths)),
        "mean_strength": float(np.mean(strengths)),
        "sum_strength": float(np.sum(strengths)),
        "top_k_strengths": [float(s) for s in sorted(strengths, reverse=True)[:10]]
    }


def compute_graph_features(
    md_matrix: np.ndarray,
    ml_matrix: np.ndarray,
    dl_matrix: np.ndarray,
    mirna_idx: int,
    disease_idx: int
) -> Dict:
    """
    计算图特征
    
    Args:
        md_matrix: miRNA-disease 关联矩阵
        ml_matrix: miRNA-lncRNA 关联矩阵
        dl_matrix: disease-lncRNA 关联矩阵
        mirna_idx: miRNA 索引
        disease_idx: 疾病索引
    
    Returns:
        包含所有图特征的字典
    """
    # 直接链接
    direct_link = get_direct_link(md_matrix, mirna_idx, disease_idx)
    
    # 2-hop 路径
    two_hop_paths = get_two_hop_paths(ml_matrix, dl_matrix, mirna_idx, disease_idx)
    path_stats = get_path_statistics(two_hop_paths)
    
    # miRNA 的度（关联的疾病数）
    mirna_degree = int(np.sum(md_matrix[mirna_idx, :]))
    
    # disease 的度（关联的 miRNA 数）
    disease_degree = int(np.sum(md_matrix[:, disease_idx]))
    
    # miRNA 通过 lncRNA 关联的疾病数
    mirna_lncrna_indices = np.where(ml_matrix[mirna_idx, :] > 0)[0]
    diseases_via_lncrna = set()
    for lncrna_idx in mirna_lncrna_indices:
        disease_indices = np.where(dl_matrix[:, lncrna_idx] > 0)[0]
        diseases_via_lncrna.update(disease_indices)
    
    return {
        "direct_link": direct_link,
        "two_hop_paths": two_hop_paths,
        "path_statistics": path_stats,
        "mirna_degree": mirna_degree,
        "disease_degree": disease_degree,
        "num_diseases_via_lncrna": len(diseases_via_lncrna),
        "disease_in_lncrna_paths": 1 if disease_idx in diseases_via_lncrna else 0
    }

