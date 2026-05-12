"""Random Walk with Restart feature extraction agent."""
import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RWRAgent:
    """RWRAgent."""
    
    def __init__(self, data_loader):
        """  init  ."""
        self.data_loader = data_loader
    
    def compute_evidence(self, mirna_idx: int, disease_idx: int, restart_prob: float = 0.3, max_iter: int = 100) -> Dict:
        """Compute evidence."""
        train_md_matrix = self.data_loader.get_train_matrix()
        
        n_mirna = train_md_matrix.shape[0]
        n_disease = train_md_matrix.shape[1]
        n_lncrna = self.data_loader.ml_matrix.shape[1]
        
        total_nodes = n_mirna + n_disease + n_lncrna
        
        adj_matrix = np.zeros((total_nodes, total_nodes))
        
        adj_matrix[:n_mirna, n_mirna:n_mirna+n_disease] = train_md_matrix
        adj_matrix[n_mirna:n_mirna+n_disease, :n_mirna] = train_md_matrix.T
        
        adj_matrix[:n_mirna, n_mirna+n_disease:] = self.data_loader.ml_matrix
        adj_matrix[n_mirna+n_disease:, :n_mirna] = self.data_loader.ml_matrix.T
        
        adj_matrix[n_mirna:n_mirna+n_disease, n_mirna+n_disease:] = self.data_loader.dl_matrix
        adj_matrix[n_mirna+n_disease:, n_mirna:n_mirna+n_disease] = self.data_loader.dl_matrix.T
        
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = adj_matrix / row_sums
        
        start_node = mirna_idx
        p = np.zeros(total_nodes)
        p[start_node] = 1.0
        
        for _ in range(max_iter):
            p_new = (1 - restart_prob) * transition_matrix.T @ p + restart_prob * p
            if np.linalg.norm(p_new - p) < 1e-6:
                break
            p = p_new
        
        disease_start = n_mirna
        disease_end = n_mirna + n_disease
        disease_probs = p[disease_start:disease_end]
        target_disease_prob = disease_probs[disease_idx]
        
        rank = int(np.sum(disease_probs > target_disease_prob)) + 1
        max_prob = float(np.max(disease_probs))
        mean_prob = float(np.mean(disease_probs))
        
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

