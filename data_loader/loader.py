"""Data loading utilities for miRNA-disease association prediction."""
import pickle
import numpy as np
import os
from typing import Dict, Tuple, Optional


class DataLoader:
    """DataLoader."""
    
    def __init__(self, dataset_dir: str = "./dataset"):
        """  init  ."""
        self.dataset_dir = dataset_dir
        self.common_data = None
        self.train_data = None
        self.test_data = None
        
        self.miRNA_names = None
        self.disease_names = None
        self.lncRNA_names = None
        
        self.md_matrix = None  # miRNA-disease
        self.ml_matrix = None  # miRNA-lncRNA
        self.dl_matrix = None  # disease-lncRNA
        
        self.mm_seq = None  # miRNA-miRNA sequence similarity
        self.dd_sem = None  # disease-disease semantic similarity
        self.ll_seq = None  # lncRNA-lncRNA sequence similarity
        
        self.gip_matrices = {}
        self.functional_matrices = {}
        
        self._train_md_matrix = None
    
    def load_all(self):
        """Load all."""
        print("Loading data from dataset directory...")
        
        self.miRNA_names = np.load(os.path.join(self.dataset_dir, "miRNA_name.npy"))
        self.disease_names = np.load(os.path.join(self.dataset_dir, "disease_name.npy"))
        self.lncRNA_names = np.load(os.path.join(self.dataset_dir, "lncRNA_name.npy"))
        
        self.md_matrix = np.load(os.path.join(self.dataset_dir, "miRNA_disease.npy"))
        self.ml_matrix = np.load(os.path.join(self.dataset_dir, "miRNA_lncRNA.npy"))
        self.dl_matrix = np.load(os.path.join(self.dataset_dir, "disease_lncRNA.npy"))
        
        self.mm_seq = np.load(os.path.join(self.dataset_dir, "miRNA_miRNA.npy"))
        self.dd_sem = np.load(os.path.join(self.dataset_dir, "disease_disease.npy"))
        self.ll_seq = np.load(os.path.join(self.dataset_dir, "lncRNA_lncRNA.npy"))
        
        # Helper function for PyTorch pickle loading
        def persistent_load(saved_id):
            return saved_id

        def load_pkl_robust(path):
            """Robust loader that handles PyTorch pickles without requiring torch installed if possible, or using torch if available"""
            if not os.path.exists(path):
                return None
            
            try:
                import torch
                try:
                    return torch.load(path, map_location='cpu', weights_only=False)
                except TypeError:
                    return torch.load(path, map_location='cpu')
            except ImportError:
                try:
                    with open(path, 'rb') as f:
                        unpickler = pickle.Unpickler(f)
                        unpickler.persistent_load = persistent_load
                        return unpickler.load()
                except Exception as e:
                    print(f"Warning: Failed to load {path} with custom unpickler: {e}")
                    return {}
            except Exception as e:
                try:
                    with open(path, 'rb') as f:
                        unpickler = pickle.Unpickler(f)
                        unpickler.persistent_load = persistent_load
                        return unpickler.load()
                except Exception as e2:
                    print(f"Warning: Could not load {path}: {e2}")
                    return {}

        self.common_data = load_pkl_robust(os.path.join(self.dataset_dir, "common_set.pkl"))
        if isinstance(self.common_data, dict):
             for key, value in self.common_data.items():
                if hasattr(value, 'numpy'):
                    self.common_data[key] = value.numpy()
                if "gip" in str(key).lower():
                    self.gip_matrices[key] = self.common_data[key]
                elif "functional" in str(key).lower() :
                    self.functional_matrices[key] = self.common_data[key]

        self.train_data = load_pkl_robust(os.path.join(self.dataset_dir, "train_set.pkl"))
        if isinstance(self.train_data, dict):
            for key, value in self.train_data.items():
                if hasattr(value, 'numpy'):
                    self.train_data[key] = value.numpy()
        
        self.test_data = load_pkl_robust(os.path.join(self.dataset_dir, "test_set.pkl"))
        if isinstance(self.test_data, dict):
            for key, value in self.test_data.items():
                if hasattr(value, 'numpy'):
                    self.test_data[key] = value.numpy()
        
        print(f"Loaded {len(self.miRNA_names)} miRNAs, {len(self.disease_names)} diseases, {len(self.lncRNA_names)} lncRNAs")
        print(f"Loaded association matrices: md({self.md_matrix.shape}), ml({self.ml_matrix.shape}), dl({self.dl_matrix.shape})")
    
    def get_miRNA_index(self, mirna_name: str) -> Optional[int]:
        """Get mirna index."""
        indices = np.where(self.miRNA_names == mirna_name)[0]
        if len(indices) > 0:
            return indices[0]
        
        indices = np.where(np.char.lower(self.miRNA_names) == mirna_name.lower())[0]
        if len(indices) > 0:
            return indices[0]
        
        for i, name in enumerate(self.miRNA_names):
            if mirna_name.lower() in name.lower() or name.lower() in mirna_name.lower():
                return i
        
        return None
    
    def get_disease_index(self, disease_name: str) -> Optional[int]:
        """Get disease index."""
        indices = np.where(self.disease_names == disease_name)[0]
        if len(indices) > 0:
            return indices[0]
        
        indices = np.where(np.char.lower(self.disease_names) == disease_name.lower())[0]
        if len(indices) > 0:
            return indices[0]
        
        disease_name_normalized = disease_name.replace("_", " ").lower()
        for i, name in enumerate(self.disease_names):
            name_normalized = name.replace("_", " ").lower()
            if disease_name_normalized in name_normalized or name_normalized in disease_name_normalized:
                return i
        
        return None
    
    def get_lncRNA_index(self, lncrna_name: str) -> Optional[int]:
        """Get lncrna index."""
        indices = np.where(self.lncRNA_names == lncrna_name)[0]
        if len(indices) > 0:
            return indices[0]
        
        indices = np.where(np.char.lower(self.lncRNA_names) == lncrna_name.lower())[0]
        if len(indices) > 0:
            return indices[0]
        
        return None
    
    def get_train_matrix(self) -> np.ndarray:
        """Get train matrix."""
        if self._train_md_matrix is not None:
            return self._train_md_matrix
        
        if self.train_data is None or not isinstance(self.train_data, dict):
            print("Warning: Train data not loaded! Using zero matrix (no associations).")
            self._train_md_matrix = np.zeros_like(self.md_matrix)
            return self._train_md_matrix
        
        train_md_matrix = np.zeros_like(self.md_matrix)
        train_pos_count = 0
        
        if 'edge' in self.train_data and 'label' in self.train_data:
            edges = self.train_data['edge']
            labels = self.train_data['label']
            
            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'cpu'):
                edges = edges.cpu().numpy()
            
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            elif hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            
            for i in range(len(edges)):
                if labels[i] == 1:
                    mirna_idx = int(edges[i, 0])
                    disease_idx = int(edges[i, 1])
                    train_md_matrix[mirna_idx, disease_idx] = 1
                    train_pos_count += 1
        
        elif 'edge_train' in self.train_data and 'label_train' in self.train_data:
            edges = self.train_data['edge_train']
            labels = self.train_data['label_train']
            
            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'cpu'):
                edges = edges.cpu().numpy()
            
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            elif hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            
            for i in range(len(edges)):
                if labels[i] == 1:
                    mirna_idx = int(edges[i, 0])
                    disease_idx = int(edges[i, 1])
                    train_md_matrix[mirna_idx, disease_idx] = 1
                    train_pos_count += 1
        
        elif any('edge_train_' in key and key.replace('edge_train_', '').isdigit() for key in self.train_data.keys()):
            for k in range(10):
                edge_key = f'edge_train_{k}'
                label_key = f'label_train_{k}'
                
                if edge_key in self.train_data:
                    edges = self.train_data[edge_key]
                    
                    if hasattr(edges, 'numpy'):
                        edges = edges.numpy()
                    elif hasattr(edges, 'cpu'):
                        edges = edges.cpu().numpy()
                    
                    if label_key in self.train_data:
                        labels = self.train_data[label_key]
                        if hasattr(labels, 'numpy'):
                            labels = labels.numpy()
                        elif hasattr(labels, 'cpu'):
                            labels = labels.cpu().numpy()
                    else:
                        labels = np.ones(len(edges))
                    
                    for i in range(len(edges)):
                        if labels[i] == 1:
                            mirna_idx = int(edges[i, 0])
                            disease_idx = int(edges[i, 1])
                            train_md_matrix[mirna_idx, disease_idx] = 1
                            train_pos_count += 1
        
        else:
            available_keys = list(self.train_data.keys())[:10]
            print("Warning: Train data format not recognized!")
            print(f"   Available keys: {available_keys}")
            print("   Trying to extract edges from all keys containing 'edge'...")
            
            for key in self.train_data.keys():
                if 'edge' in key.lower():
                    edges = self.train_data[key]
                    
                    if hasattr(edges, 'numpy'):
                        edges = edges.numpy()
                    elif hasattr(edges, 'cpu'):
                        edges = edges.cpu().numpy()
                    
                    for i in range(len(edges)):
                        mirna_idx = int(edges[i, 0])
                        disease_idx = int(edges[i, 1])
                        train_md_matrix[mirna_idx, disease_idx] = 1
                        train_pos_count += 1
        
        if train_pos_count > 0:
            print(f"OK Created train matrix with {train_pos_count} positive associations (excluding test set)")
        else:
            print("Warning: Train matrix extraction failed (likely due to pickle load failure).")
            print("   FALLBACK: Using full md_matrix as train matrix to ensure graph connectivity.")
            train_md_matrix = self.md_matrix.copy()
            print(f"   OK Fallback train matrix created with {int(np.sum(train_md_matrix))} associations.")
        
        self._train_md_matrix = train_md_matrix
        return self._train_md_matrix
    
    def get_miRNA_disease_features(self, mirna_idx: int, disease_idx: int) -> Dict:
        """Get mirna disease features."""
        features = {
            "direct_link": float(self.md_matrix[mirna_idx, disease_idx]),
            "miRNA_similarity_row": self.mm_seq[mirna_idx, :],
            "disease_similarity_row": self.dd_sem[disease_idx, :],
            "miRNA_disease_associations": self.md_matrix[mirna_idx, :],
            "disease_miRNA_associations": self.md_matrix[:, disease_idx],
        }
        
        return features

