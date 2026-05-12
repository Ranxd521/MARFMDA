"""Similarity feature extraction agent."""
import numpy as np
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimilarityAgent:
    """SimilarityAgent."""
    
    def __init__(self, data_loader):
        """  init  ."""
        self.data_loader = data_loader
    
    def compute_evidence(self, mirna_idx: int, disease_idx: int) -> Dict:
        """Compute evidence."""
        mm_seq = self.data_loader.mm_seq
        dd_sem = self.data_loader.dd_sem
        
        mirna_similarity_row = mm_seq[mirna_idx, :]
        
        disease_similarity_row = dd_sem[disease_idx, :]
        
        similar_mirnas = np.argsort(mirna_similarity_row)[::-1][:10]  # top-10
        similar_mirnas_scores = mirna_similarity_row[similar_mirnas]
        
        train_md_matrix = self.data_loader.get_train_matrix()
        
        similar_mirnas_disease_assoc = train_md_matrix[similar_mirnas, disease_idx]
        
        similar_diseases = np.argsort(disease_similarity_row)[::-1][:10]  # top-10
        similar_diseases_scores = disease_similarity_row[similar_diseases]
        
        mirna_similar_diseases_assoc = train_md_matrix[mirna_idx, similar_diseases]
        
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
            "text_evidence": f"Similar miRNAs: {mirna_assoc_count}/10 associate with target disease (avg sim: {top_mirna_sim:.3f}). Similar diseases: {disease_assoc_count}/10 associate with target miRNA (avg sim: {top_disease_sim:.3f})."
        }

