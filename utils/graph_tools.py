"""Graph utility functions for numeric feature extraction."""
import numpy as np
from typing import Dict, List, Tuple


def get_direct_link(md_matrix: np.ndarray, mirna_idx: int, disease_idx: int) -> float:
    """Get direct link."""
    return float(md_matrix[mirna_idx, disease_idx])


def get_two_hop_paths(
    ml_matrix: np.ndarray,
    dl_matrix: np.ndarray,
    mirna_idx: int,
    disease_idx: int
) -> List[Tuple[int, float]]:
    """Get two hop paths."""
    paths = []
    
    # miRNA -> lncRNA
    mirna_lncrna = ml_matrix[mirna_idx, :]
    
    # lncRNA -> disease (dl_matrix shape: (n_disease, n_lncrna))
    lncrna_disease = dl_matrix[disease_idx, :]
    
    path_strengths = np.minimum(mirna_lncrna, lncrna_disease)
    
    nonzero_indices = np.where(path_strengths > 0)[0]
    
    for lncrna_idx in nonzero_indices:
        strength = float(path_strengths[lncrna_idx])
        paths.append((int(lncrna_idx), strength))
    
    paths.sort(key=lambda x: x[1], reverse=True)
    
    return paths


def get_path_statistics(paths: List[Tuple[int, float]]) -> Dict:
    """Get path statistics."""
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
    """Graph utility functions for numeric feature extraction."""
    direct_link = get_direct_link(md_matrix, mirna_idx, disease_idx)
    
    two_hop_paths = get_two_hop_paths(ml_matrix, dl_matrix, mirna_idx, disease_idx)
    path_stats = get_path_statistics(two_hop_paths)
    
    mirna_degree = int(np.sum(md_matrix[mirna_idx, :]))
    
    disease_degree = int(np.sum(md_matrix[:, disease_idx]))
    
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

