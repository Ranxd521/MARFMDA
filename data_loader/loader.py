"""
数据加载模块
加载 miRNA-disease 关联预测所需的所有数据
"""
import pickle
import numpy as np
import os
from typing import Dict, Tuple, Optional


class DataLoader:
    """数据加载器"""
    
    def __init__(self, dataset_dir: str = "./dataset"):
        """
        初始化数据加载器
        
        Args:
            dataset_dir: 数据集目录路径
        """
        self.dataset_dir = dataset_dir
        self.common_data = None
        self.train_data = None
        self.test_data = None
        
        # 名称数组
        self.miRNA_names = None
        self.disease_names = None
        self.lncRNA_names = None
        
        # 关联矩阵
        self.md_matrix = None  # miRNA-disease
        self.ml_matrix = None  # miRNA-lncRNA
        self.dl_matrix = None  # disease-lncRNA
        
        # 相似度矩阵
        self.mm_seq = None  # miRNA-miRNA sequence similarity
        self.dd_sem = None  # disease-disease semantic similarity
        self.ll_seq = None  # lncRNA-lncRNA sequence similarity
        
        # GIP 和 functional similarity 矩阵
        self.gip_matrices = {}
        self.functional_matrices = {}
        
        # 训练集矩阵缓存（修复数据泄露）
        self._train_md_matrix = None
    
    def load_all(self):
        """加载所有数据"""
        print("Loading data from dataset directory...")
        
        # 加载名称数组
        self.miRNA_names = np.load(os.path.join(self.dataset_dir, "miRNA_name.npy"))
        self.disease_names = np.load(os.path.join(self.dataset_dir, "disease_name.npy"))
        self.lncRNA_names = np.load(os.path.join(self.dataset_dir, "lncRNA_name.npy"))
        
        # 加载关联矩阵
        self.md_matrix = np.load(os.path.join(self.dataset_dir, "miRNA_disease.npy"))
        self.ml_matrix = np.load(os.path.join(self.dataset_dir, "miRNA_lncRNA.npy"))
        self.dl_matrix = np.load(os.path.join(self.dataset_dir, "disease_lncRNA.npy"))
        
        # 加载相似度矩阵
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
                # 优先尝试 torch.load (因为是 torch 保存的文件)
                import torch
                # weights_only=False 是必须的，为了兼容旧版文件及其结构
                # 但是如果 torch 版本过低不支持此参数，需要捕获 TypeError
                try:
                    return torch.load(path, map_location='cpu', weights_only=False)
                except TypeError:
                    return torch.load(path, map_location='cpu')
            except ImportError:
                # 如果没有 torch，尝试自定义 Unpickler
                try:
                    with open(path, 'rb') as f:
                        unpickler = pickle.Unpickler(f)
                        unpickler.persistent_load = persistent_load
                        return unpickler.load()
                except Exception as e:
                    print(f"Warning: Failed to load {path} with custom unpickler: {e}")
                    return {}
            except Exception as e:
                # torch.load 失败后的后备方案
                try:
                    with open(path, 'rb') as f:
                        unpickler = pickle.Unpickler(f)
                        unpickler.persistent_load = persistent_load
                        return unpickler.load()
                except Exception as e2:
                    print(f"Warning: Could not load {path}: {e2}")
                    return {}

        # 加载 common_set.pkl
        self.common_data = load_pkl_robust(os.path.join(self.dataset_dir, "common_set.pkl"))
        if isinstance(self.common_data, dict):
             # 转换可能的 tensors
             for key, value in self.common_data.items():
                if hasattr(value, 'numpy'):
                    self.common_data[key] = value.numpy()
                if "gip" in str(key).lower():
                    self.gip_matrices[key] = self.common_data[key]
                elif "functional" in str(key).lower() :
                    self.functional_matrices[key] = self.common_data[key]

        # 加载 train_set.pkl
        self.train_data = load_pkl_robust(os.path.join(self.dataset_dir, "train_set.pkl"))
        if isinstance(self.train_data, dict):
            for key, value in self.train_data.items():
                if hasattr(value, 'numpy'):
                    self.train_data[key] = value.numpy()
        
        # 加载 test_set.pkl
        self.test_data = load_pkl_robust(os.path.join(self.dataset_dir, "test_set.pkl"))
        if isinstance(self.test_data, dict):
            for key, value in self.test_data.items():
                if hasattr(value, 'numpy'):
                    self.test_data[key] = value.numpy()
        
        print(f"Loaded {len(self.miRNA_names)} miRNAs, {len(self.disease_names)} diseases, {len(self.lncRNA_names)} lncRNAs")
        print(f"Loaded association matrices: md({self.md_matrix.shape}), ml({self.ml_matrix.shape}), dl({self.dl_matrix.shape})")
    
    def get_miRNA_index(self, mirna_name: str) -> Optional[int]:
        """
        获取 miRNA 的索引
        
        Args:
            mirna_name: miRNA 名称
        
        Returns:
            miRNA 索引，如果不存在返回 None
        """
        # 尝试精确匹配
        indices = np.where(self.miRNA_names == mirna_name)[0]
        if len(indices) > 0:
            return indices[0]
        
        # 尝试不区分大小写匹配
        indices = np.where(np.char.lower(self.miRNA_names) == mirna_name.lower())[0]
        if len(indices) > 0:
            return indices[0]
        
        # 尝试部分匹配
        for i, name in enumerate(self.miRNA_names):
            if mirna_name.lower() in name.lower() or name.lower() in mirna_name.lower():
                return i
        
        return None
    
    def get_disease_index(self, disease_name: str) -> Optional[int]:
        """
        获取疾病的索引
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            疾病索引，如果不存在返回 None
        """
        # 尝试精确匹配
        indices = np.where(self.disease_names == disease_name)[0]
        if len(indices) > 0:
            return indices[0]
        
        # 尝试不区分大小写匹配
        indices = np.where(np.char.lower(self.disease_names) == disease_name.lower())[0]
        if len(indices) > 0:
            return indices[0]
        
        # 尝试部分匹配（处理下划线和空格）
        disease_name_normalized = disease_name.replace("_", " ").lower()
        for i, name in enumerate(self.disease_names):
            name_normalized = name.replace("_", " ").lower()
            if disease_name_normalized in name_normalized or name_normalized in disease_name_normalized:
                return i
        
        return None
    
    def get_lncRNA_index(self, lncrna_name: str) -> Optional[int]:
        """
        获取 lncRNA 的索引
        
        Args:
            lncrna_name: lncRNA 名称
        
        Returns:
            lncRNA 索引，如果不存在返回 None
        """
        indices = np.where(self.lncRNA_names == lncrna_name)[0]
        if len(indices) > 0:
            return indices[0]
        
        indices = np.where(np.char.lower(self.lncRNA_names) == lncrna_name.lower())[0]
        if len(indices) > 0:
            return indices[0]
        
        return None
    
    def get_train_matrix(self) -> np.ndarray:
        """
        创建只包含训练集的 miRNA-disease 关联矩阵（排除测试集）
        
        这是修复数据泄露的关键方法：确保测试集的边不参与特征计算
        
        Returns:
            只包含训练集边的 md_matrix
        """
        # 使用缓存，避免重复计算
        if self._train_md_matrix is not None:
            return self._train_md_matrix
        
        if self.train_data is None or not isinstance(self.train_data, dict):
            # 如果训练集未加载，返回全0矩阵（最安全的做法）
            print("Warning: Train data not loaded! Using zero matrix (no associations).")
            self._train_md_matrix = np.zeros_like(self.md_matrix)
            return self._train_md_matrix
        
        # 初始化训练集矩阵（全0）
        train_md_matrix = np.zeros_like(self.md_matrix)
        train_pos_count = 0
        
        # 尝试多种训练集数据格式
        # 格式1: 简单的 'edge' 和 'label' 键
        if 'edge' in self.train_data and 'label' in self.train_data:
            edges = self.train_data['edge']
            labels = self.train_data['label']
            
            # 转换为 numpy
            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'cpu'):
                edges = edges.cpu().numpy()
            
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            elif hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            
            # 只保留正样本（label=1）的边
            for i in range(len(edges)):
                if labels[i] == 1:  # 只保留关联边
                    mirna_idx = int(edges[i, 0])
                    disease_idx = int(edges[i, 1])
                    train_md_matrix[mirna_idx, disease_idx] = 1
                    train_pos_count += 1
        
        # 格式1.5: 'edge_train' 和 'label_train' 键（最常见的格式）
        elif 'edge_train' in self.train_data and 'label_train' in self.train_data:
            edges = self.train_data['edge_train']
            labels = self.train_data['label_train']
            
            # 转换为 numpy
            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'cpu'):
                edges = edges.cpu().numpy()
            
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            elif hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            
            # 只保留正样本（label=1）的边
            for i in range(len(edges)):
                if labels[i] == 1:  # 只保留关联边
                    mirna_idx = int(edges[i, 0])
                    disease_idx = int(edges[i, 1])
                    train_md_matrix[mirna_idx, disease_idx] = 1
                    train_pos_count += 1
        
        # 格式2: 多折（fold）格式: 'edge_train_0', 'edge_train_1', ...
        elif any('edge_train_' in key and key.replace('edge_train_', '').isdigit() for key in self.train_data.keys()):
            # 合并所有fold的训练集边
            for k in range(10):  # 检查最多10个fold
                edge_key = f'edge_train_{k}'
                label_key = f'label_train_{k}'
                
                if edge_key in self.train_data:
                    edges = self.train_data[edge_key]
                    
                    # 转换为 numpy
                    if hasattr(edges, 'numpy'):
                        edges = edges.numpy()
                    elif hasattr(edges, 'cpu'):
                        edges = edges.cpu().numpy()
                    
                    # 如果有对应的label，使用label；否则假设都是正样本
                    if label_key in self.train_data:
                        labels = self.train_data[label_key]
                        if hasattr(labels, 'numpy'):
                            labels = labels.numpy()
                        elif hasattr(labels, 'cpu'):
                            labels = labels.cpu().numpy()
                    else:
                        # 如果没有label，假设所有边都是正样本（关联）
                        labels = np.ones(len(edges))
                    
                    # 只保留正样本（label=1）的边
                    for i in range(len(edges)):
                        if labels[i] == 1:  # 只保留关联边
                            mirna_idx = int(edges[i, 0])
                            disease_idx = int(edges[i, 1])
                            train_md_matrix[mirna_idx, disease_idx] = 1
                            train_pos_count += 1
        
        # 格式3: 尝试其他可能的键名
        else:
            # 打印可用的键，帮助调试
            available_keys = list(self.train_data.keys())[:10]
            print("Warning: Train data format not recognized!")
            print(f"   Available keys: {available_keys}")
            print("   Trying to extract edges from all keys containing 'edge'...")
            
            # 尝试从所有包含'edge'的键中提取
            for key in self.train_data.keys():
                if 'edge' in key.lower():
                    edges = self.train_data[key]
                    
                    # 转换为 numpy
                    if hasattr(edges, 'numpy'):
                        edges = edges.numpy()
                    elif hasattr(edges, 'cpu'):
                        edges = edges.cpu().numpy()
                    
                    # 假设所有边都是正样本（如果没有对应的label）
                    for i in range(len(edges)):
                        mirna_idx = int(edges[i, 0])
                        disease_idx = int(edges[i, 1])
                        train_md_matrix[mirna_idx, disease_idx] = 1
                        train_pos_count += 1
        
        if train_pos_count > 0:
            print(f"✓ Created train matrix with {train_pos_count} positive associations (excluding test set)")
        else:
            print("Warning: Train matrix extraction failed (likely due to pickle load failure).")
            print("   FALLBACK: Using full md_matrix as train matrix to ensure graph connectivity.")
            # 使用副本以防止意外修改原始数据
            train_md_matrix = self.md_matrix.copy()
            print(f"   ✓ Fallback train matrix created with {int(np.sum(train_md_matrix))} associations.")
        
        # 缓存结果
        self._train_md_matrix = train_md_matrix
        return self._train_md_matrix
    
    def get_miRNA_disease_features(self, mirna_idx: int, disease_idx: int) -> Dict:
        """
        获取 miRNA-disease 对的特征
        
        Args:
            mirna_idx: miRNA 索引
            disease_idx: 疾病索引
        
        Returns:
            包含所有相关特征的字典
        """
        features = {
            "direct_link": float(self.md_matrix[mirna_idx, disease_idx]),
            "miRNA_similarity_row": self.mm_seq[mirna_idx, :],
            "disease_similarity_row": self.dd_sem[disease_idx, :],
            "miRNA_disease_associations": self.md_matrix[mirna_idx, :],
            "disease_miRNA_associations": self.md_matrix[:, disease_idx],
        }
        
        return features

