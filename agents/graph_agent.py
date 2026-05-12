"""Graph feature extraction agent."""
import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_tools import compute_graph_features


class GraphAgent:
    """GraphAgent."""
    
    def __init__(self, data_loader):
        """  init  ."""
        self.data_loader = data_loader
    
    def compute_evidence(self, mirna_idx: int, disease_idx: int) -> Dict:
        """Compute evidence."""
        train_md_matrix = self.data_loader.get_train_matrix()
        
        md_matrix_masked = train_md_matrix.copy()
        
        if md_matrix_masked[mirna_idx, disease_idx] != 0:
            md_matrix_masked[mirna_idx, disease_idx] = 0
            
        
        graph_features = compute_graph_features(
            md_matrix_masked,
            self.data_loader.ml_matrix,
            self.data_loader.dl_matrix,
            mirna_idx,
            disease_idx
        )
        
        direct_link = graph_features.get("direct_link", 0)
        path_stats = graph_features.get("path_statistics", {})
        num_paths = path_stats.get("num_paths", 0)
        max_strength = path_stats.get("max_strength", 0.0)
        mean_strength = path_stats.get("mean_strength", 0.0)
        mirna_degree = graph_features.get("mirna_degree", 0)
        disease_degree = graph_features.get("disease_degree", 0)
        
        text_evidence = f"Direct link: {bool(direct_link)}. "
        if num_paths > 0:
            text_evidence += f"2-hop paths: {num_paths} (max strength: {max_strength:.3f}, mean: {mean_strength:.3f}). "
        else:
            text_evidence += "No 2-hop paths. "
            
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

