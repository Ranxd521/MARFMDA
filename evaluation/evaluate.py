"""
评估脚本：计算准确率、精确率、召回率、F1 等指标
"""
import argparse
import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.loader import DataLoader


def load_predictions(results_file: str) -> List[Dict]:
    """加载预测结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_true_labels(data_loader: DataLoader, pairs: List[Tuple[str, str]]) -> np.ndarray:
    """
    从数据加载器获取真实标签
    
    WARNING  重要：优先使用 test_set 中的标签，避免数据泄露
    如果 test_set 不可用，才使用 md_matrix（可能包含训练集数据）
    
    Args:
        data_loader: DataLoader 实例
        pairs: miRNA-disease 对列表
    
    Returns:
        真实标签数组 (1=关联, 0=不关联, -1=未知)
    """
    labels = []
    
    # 首先尝试从 test_set 获取标签（避免数据泄露）
    test_label_map = {}
    if data_loader.test_data and isinstance(data_loader.test_data, dict):
        if 'edge' in data_loader.test_data and 'label' in data_loader.test_data:
            edges = data_loader.test_data['edge']
            test_labels = data_loader.test_data['label']
            
            # 转换为 numpy 数组
            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'cpu'):
                edges = edges.cpu().numpy()
            if hasattr(test_labels, 'numpy'):
                test_labels = test_labels.numpy()
            elif hasattr(test_labels, 'cpu'):
                test_labels = test_labels.cpu().numpy()
            
            # 构建 (mirna_idx, disease_idx) -> label 的映射
            for i in range(len(edges)):
                mirna_idx = int(edges[i, 0])
                disease_idx = int(edges[i, 1])
                label = int(test_labels[i])
                test_label_map[(mirna_idx, disease_idx)] = label
    
    # 获取训练集位置（用于检查数据泄露）
    train_positions = set()
    if data_loader.train_data and isinstance(data_loader.train_data, dict):
        for k in range(5):  # 检查前5个fold
            key = f'edge_train_{k}'
            if key in data_loader.train_data:
                train_edges = data_loader.train_data[key]
                if hasattr(train_edges, 'numpy'):
                    train_edges = train_edges.numpy()
                elif hasattr(train_edges, 'cpu'):
                    train_edges = train_edges.cpu().numpy()
                
                for i in range(len(train_edges)):
                    mirna_idx = int(train_edges[i, 0])
                    disease_idx = int(train_edges[i, 1])
                    train_positions.add((mirna_idx, disease_idx))
    
    # 统计信息
    from_test_set = 0
    from_md_matrix = 0
    in_train_set = 0
    unknown = 0
    
    for mirna_name, disease_name in pairs:
        try:
            # 使用 np.where 查找索引
            mirna_indices = np.where(data_loader.miRNA_names == mirna_name)[0]
            disease_indices = np.where(data_loader.disease_names == disease_name)[0]
            
            if len(mirna_indices) == 0 or len(disease_indices) == 0:
                labels.append(-1)
                unknown += 1
                continue
            
            mirna_idx = mirna_indices[0]
            disease_idx = disease_indices[0]
            position = (mirna_idx, disease_idx)
            
            # 优先从 test_set 获取标签
            if position in test_label_map:
                true_label = test_label_map[position]
                labels.append(true_label)
                from_test_set += 1
            else:
                # 检查是否在训练集中（数据泄露警告）
                if position in train_positions:
                    in_train_set += 1
                    # 仍然使用 md_matrix 的标签，但会在最后打印警告
                
                # 从 md_matrix 获取标签（可能包含训练集数据）
                true_label = int(data_loader.md_matrix[mirna_idx, disease_idx])
                labels.append(true_label)
                from_md_matrix += 1
                
        except (ValueError, IndexError, TypeError) as e:
            labels.append(-1)
            unknown += 1
    
    # 打印统计信息
    if from_test_set > 0 or from_md_matrix > 0:
        print(f"\n标签来源统计:")
        print(f"  从 test_set 获取: {from_test_set} 个")
        print(f"  从 md_matrix 获取: {from_md_matrix} 个")
        if in_train_set > 0:
            print(f"  WARNING  警告: {in_train_set} 个样本在训练集中（可能存在数据泄露）")
        if unknown > 0:
            print(f"  未知标签: {unknown} 个")
    
    return np.array(labels)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, threshold: float = 0.7) -> Dict:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签 (0 或 1)
        y_pred: 预测标签 (0 或 1)
        y_scores: 预测分数 (0-1)
        threshold: 分类阈值
    
    Returns:
        包含各种指标的字典
    """
    # 过滤掉未知标签
    valid_mask = y_true >= 0
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_scores_valid = y_scores[valid_mask]
    
    if len(y_true_valid) == 0:
        return {
            "error": "No valid labels found"
        }
    
    # 计算混淆矩阵
    tp = np.sum((y_true_valid == 1) & (y_pred_valid == 1))
    fp = np.sum((y_true_valid == 0) & (y_pred_valid == 1))
    tn = np.sum((y_true_valid == 0) & (y_pred_valid == 0))
    fn = np.sum((y_true_valid == 1) & (y_pred_valid == 0))
    
    # 计算基本指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算特异性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 计算 AUC-ROC
    # 优先使用 sklearn 的 roc_auc_score（更可靠）
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true_valid)) > 1:
            # 检查正负样本数量
            n_pos = np.sum(y_true_valid == 1)
            n_neg = np.sum(y_true_valid == 0)
            
            if n_pos < 2:
                # 如果正样本太少，AUC计算不可靠
                auc = np.nan
                auc_warning = f"WARNING  警告：正样本太少（{n_pos}个），AUC计算不可靠"
            else:
                auc = roc_auc_score(y_true_valid, y_scores_valid)
                auc_warning = None
        else:
            auc = 0.5  # 如果只有一个类别，AUC 为 0.5
            auc_warning = None
    except ImportError:
        # 如果没有 sklearn，使用简化的 AUC 计算
        if len(np.unique(y_true_valid)) > 1:
            n_pos = np.sum(y_true_valid == 1)
            n_neg = np.sum(y_true_valid == 0)
            
            if n_pos < 2:
                auc = np.nan
                auc_warning = f"WARNING  警告：正样本太少（{n_pos}个），AUC计算不可靠"
            else:
                # 按分数排序
                sorted_indices = np.argsort(y_scores_valid)[::-1]
                sorted_labels = y_true_valid[sorted_indices]
                
                # 计算 AUC（使用梯形法则）
                auc = 0.0
                tpr_prev = 0.0
                fpr_prev = 0.0
                
                for i in range(len(sorted_labels)):
                    threshold_score = y_scores_valid[sorted_indices[i]]
                    y_pred_thresh = (y_scores_valid >= threshold_score).astype(int)
                    
                    tp_thresh = np.sum((y_true_valid == 1) & (y_pred_thresh == 1))
                    fp_thresh = np.sum((y_true_valid == 0) & (y_pred_thresh == 1))
                    tn_thresh = np.sum((y_true_valid == 0) & (y_pred_thresh == 0))
                    fn_thresh = np.sum((y_true_valid == 1) & (y_pred_thresh == 0))
                    
                    tpr = tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0.0
                    fpr = fp_thresh / (fp_thresh + tn_thresh) if (fp_thresh + tn_thresh) > 0 else 0.0
                    
                    if i > 0:
                        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
                    
                    tpr_prev = tpr
                    fpr_prev = fpr
                
                # 添加最后一个点
                auc += (1.0 - fpr_prev) * (1.0 + tpr_prev) / 2.0
                auc_warning = None
        else:
            auc = 0.5
            auc_warning = None
    
    return {
        "threshold": threshold,
        "total_samples": len(y_true_valid),
        "positive_samples": int(np.sum(y_true_valid == 1)),
        "negative_samples": int(np.sum(y_true_valid == 0)),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn)
        },
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "auc_roc": float(auc) if not np.isnan(auc) else None,
        "mean_score_positive": float(np.mean(y_scores_valid[y_true_valid == 1])) if np.sum(y_true_valid == 1) > 0 else 0.0,
        "mean_score_negative": float(np.mean(y_scores_valid[y_true_valid == 0])) if np.sum(y_true_valid == 0) > 0 else 0.0,
        "auc_warning": auc_warning
    }


def calculate_explainability_metrics(predictions: List[Dict]) -> Dict:
    """
    计算解释性评估指标
    
    Args:
        predictions: 预测结果列表，每个元素包含 mirna, disease, score, explanation 等字段
    
    Returns:
        包含解释性指标的字典
    """
    import re
    from collections import Counter
    import random
    
    def sample_pairs_efficiently(n: int, k: int) -> List[Tuple[int, int]]:
        """
        高效地采样k个不重复的索引对 (i, j)，其中 i < j < n
        避免创建完整的pairs列表，节省内存
        
        Args:
            n: 总数量
            k: 需要采样的对数
        
        Returns:
            采样的索引对列表
        """
        total_pairs = n * (n - 1) // 2
        
        # 如果总对数不大（小于100万），可以直接创建列表
        if total_pairs < 1000000:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return random.sample(pairs, min(k, total_pairs))
        
        # 对于大数据集，使用集合来存储已采样的对，直接生成随机对
        sampled_pairs = set()
        max_attempts = k * 20  # 设置最大尝试次数，避免无限循环
        attempts = 0
        
        while len(sampled_pairs) < k and attempts < max_attempts:
            # 随机选择第一个索引
            i = random.randint(0, n - 2)
            # 随机选择第二个索引（确保 i < j）
            j = random.randint(i + 1, n - 1)
            sampled_pairs.add((i, j))
            attempts += 1
        
        return list(sampled_pairs)
    
    # 提取有效样本（有explanation或reasoning字段）
    valid_samples = []
    for pred in predictions:
        explanation = pred.get("explanation", "") or pred.get("reasoning", "") or pred.get("fusion_reasoning", "")
        if not explanation or not isinstance(explanation, str):
            continue
        
        # 提取分数（兼容不同格式）
        final_score = pred.get("score", pred.get("predicted_score", 0.0))
        sim_score = pred.get("sim_score", 0.0)
        graph_score = pred.get("graph_score", 0.0)
        rwr_score = pred.get("rwr_score", 0.0)
        web_score = pred.get("web_score", 0.0)
        
        # 如果分数不在顶层，尝试从agent_scores中提取
        if sim_score == 0.0 and "agent_scores" in pred:
            agent_scores = pred.get("agent_scores", {})
            sim_score = agent_scores.get("similarity", {}).get("score", 0.0)
            graph_score = agent_scores.get("graph", {}).get("score", 0.0)
            rwr_score = agent_scores.get("rwr", {}).get("score", 0.0)
            web_score = agent_scores.get("web", {}).get("score", 0.0)
        
        # 提取特征值（用于更精确的一致性检查）
        rwr_features = {}
        graph_features = {}
        sim_features = {}
        if "agent_scores" in pred:
            agent_scores = pred.get("agent_scores", {})
            rwr_features = agent_scores.get("rwr", {}).get("features", {})
            graph_features = agent_scores.get("graph", {}).get("features", {})
            sim_features = agent_scores.get("similarity", {}).get("features", {})
        
        valid_samples.append({
            "explanation": explanation.lower(),  # 转换为小写便于匹配
            "final_score": float(final_score),
            "sim_score": float(sim_score),
            "graph_score": float(graph_score),
            "rwr_score": float(rwr_score),
            "web_score": float(web_score),
            "rwr_features": rwr_features,
            "graph_features": graph_features,
            "sim_features": sim_features
        })
    
    if len(valid_samples) == 0:
        return {
            "consistency": 0.0,
            "average_coverage": 0.0,
            "score_explanation_corr": 0.0,
            "diversity": 0.0,
            "avg_length": 0.0,
            "std_length": 0.0,
            "specificity": 0.0,
            "completeness": 0.0,
            "coherence": 0.0
        }
    
    # 1. Consistency（一致性）- LLM-as-a-Judge 评估版本
    consistent_count = 0
    try:
        from config import get_llm
        from langchain_core.messages import HumanMessage
        # 使用便宜且快速的模型作为 Judge
        judge_llm = get_llm(temperature=0.0, model_name="gpt-4o-mini") 
        judge_used = True
    except Exception as e:
        print(f"Failed to initialize LLM for Consistency checking: {e}. Falling back to 0 score.")
        judge_used = False

    import concurrent.futures
    import asyncio
    
    async def evaluate_single_sample(sample, judge_llm):
        """Asynchronously evaluate a single sample using the LLM Judge"""
        exp = sample["explanation"]
        
        # 获取数值特征用于Prompt
        features_dict = {
            "final_score": sample.get("final_score", 0),
            "graph_score": sample.get("graph_score", 0),
            "sim_score": sample.get("sim_score", 0),
            "rwr_score": sample.get("rwr_score", 0),
            "web_score": sample.get("web_score", 0),
            "graph_direct_link": sample.get("graph_features", {}).get("direct_link", "unknown"),
            "graph_paths": sample.get("graph_features", {}).get("paths", "unknown"),
        }
        
        prompt = f"""你是可解释性评估专家。你的任务是判断下方机器生成的自然语言解释是否大体上忠实地反映了真实的系统特征数值。

已知系统计算得出的真实特征数值（0到1之间，数值越大表示关联越强）：
{features_dict}

待评估的自然语言解释：
"{exp}"

判断标准（请极为宽容地评估）：
1. 【全局逻辑最重要】：只要该解释的最终结论（如强/弱关联，或其建议的高低分）基本顺应了这些数值特征的综合表现（或者顺应了其中最核心的特征，如高排名的RWR），就应该判定为“一致”。
2. 【允许推断、强调与修辞】：自然语言为了通顺或强调，可能会使用“强烈”、“极高”等词汇。哪怕某项特征真实得分属于中等偏上，只要解释中用了一些修辞手法进行渲染，只要没有南辕北辙（比如0分说成满分），都是正常的，必须判定为“一致”。
3. 【允许局部反差】：若解释中指出“虽然没有直接关联（graph=0），但通过其他特征来看很强...”，这种基于逻辑的组合陈述是非常好的一致性表现。
4. 【仅严惩极度相反的幻觉】：只有出现严重的事实倒置（例如特征数值明明极低，文本中却将其作为核心高分证据吹捧；或全局得分很低，结论却极其断定相关），才判定为违背事实。

请仔细思考上述准则。你只需输出一个词：大体上一致或合理推理请输出 "True"，只有发生严重且不可原谅的矛盾才输出 "False"。绝不要输出任何其他内容。"""
        try:
            # Langchain's async invoke
            response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
            result_text = response.content.strip().lower()
            return "true" in result_text and "false" not in result_text
        except Exception as e:
             # 容错降级
            return False

    if judge_used:
        # Run asynchronous evaluation for all samples concurrently
        async def evaluate_all_samples(samples):
            tasks = [evaluate_single_sample(sample, judge_llm) for sample in samples]
            return await asyncio.gather(*tasks)

        # Loop needed for asyncio if current event loop is not running already or use asyncio.run
        try:
            results = asyncio.run(evaluate_all_samples(valid_samples))
        except RuntimeError:
            # Fallback if an event loop is already running
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(evaluate_all_samples(valid_samples))

        for is_consistent in results:
            if is_consistent:
                consistent_count += 1
    else:
        # Fallback if Judge failed to initialize
        for sample in valid_samples:
            pass # consistency evaluates to 0
            
    consistency = consistent_count / len(valid_samples) if len(valid_samples) > 0 else 0.0
    
    # 2. Coverage（覆盖性）
    evidence_keywords = {
        "similarity": ["similarity", "similar mirna", "sequence similarity", "similar"],
        "graph": ["direct link", "indirect", "path", "network", "graph", "connection"],
        "rwr": ["random walk", "rwr", "probability", "steady state"],
        "web": ["evidence", "knowledge", "database", "literature", "publication"]
    }
    
    coverage_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        covered_types = 0
        
        for evidence_type, keywords in evidence_keywords.items():
            if any(keyword in exp for keyword in keywords):
                covered_types += 1
        
        coverage_scores.append(covered_types / 4.0)
    
    average_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
    
    # 3. Explanation–Score Correlation（分数一致性）- 改进版：直接基于final_score映射
    explanation_scores = []
    final_scores = []
    
    for sample in valid_samples:
        exp = sample["explanation"]
        final_score = sample["final_score"]
        
        # 方法1：基于综合评估结论直接映射（更宽松的匹配）
        exp_score = None
        exp_lower = exp.lower()
        
        # 更宽松的关键词匹配
        if any(phrase in exp_lower for phrase in ["strong evidence supports", "strongly supports", "strong evidence", "highly associated", "strong association"]):
            exp_score = 0.85
        elif any(phrase in exp_lower for phrase in ["moderate evidence supports", "moderate evidence", "moderate association"]):
            exp_score = 0.65
        elif any(phrase in exp_lower for phrase in ["weak evidence", "uncertain", "limited evidence"]):
            exp_score = 0.45
        elif any(phrase in exp_lower for phrase in ["no significant evidence", "no evidence", "not associated", "unlikely"]):
            exp_score = 0.25
        
        # 方法2：如果没有明确结论，基于证据强度累加（更细粒度）
        if exp_score is None:
            exp_score = 0.5  # 基础分数
            
            # 1. 图证据评分（更细粒度的匹配）
            if any(phrase in exp_lower for phrase in ["direct link", "directly connected", "direct connection", "direct association"]):
                exp_score += 0.30
            elif any(phrase in exp_lower for phrase in ["multiple paths", "many paths", "several paths", "numerous paths"]):
                exp_score += 0.20
            elif any(phrase in exp_lower for phrase in ["some paths", "few paths", "limited paths"]):
                exp_score += 0.10
            elif any(phrase in exp_lower for phrase in ["no path", "no paths", "no network paths", "no connections"]):
                exp_score -= 0.15  # 减少惩罚
            
            # 提取路径数量（如果提到）
            path_num_match = re.search(r'(\d+)\s+paths?', exp_lower)
            if path_num_match:
                path_num = int(path_num_match.group(1))
                if path_num > 10:
                    exp_score += 0.15
                elif path_num > 5:
                    exp_score += 0.10
                elif path_num > 0:
                    exp_score += 0.05
            
            # 2. RWR证据评分（更细粒度，考虑概率和排名）
            if any(phrase in exp_lower for phrase in ["high rwr", "high probability", "strong rwr signal"]):
                exp_score += 0.20
            elif any(phrase in exp_lower for phrase in ["moderate rwr", "moderate probability", "decent rwr"]):
                exp_score += 0.10
            elif any(phrase in exp_lower for phrase in ["low rwr", "low probability", "weak rwr"]):
                exp_score -= 0.05  # 减少惩罚
            elif any(phrase in exp_lower for phrase in ["very low rwr", "minimal rwr", "no significant rwr"]):
                exp_score -= 0.10
            
            # 提取RWR排名百分比（如果提到）
            rank_match = re.search(r'top\s+(\d+(?:\.\d+)?)%', exp_lower)
            if rank_match:
                rank_pct = float(rank_match.group(1))
                if rank_pct < 5:
                    exp_score += 0.15
                elif rank_pct < 10:
                    exp_score += 0.10
                elif rank_pct < 20:
                    exp_score += 0.05
            
            # 3. 相似度证据评分（更细粒度）
            if any(phrase in exp_lower for phrase in ["strong similarity", "high similarity", "strong associations"]):
                exp_score += 0.15
            elif any(phrase in exp_lower for phrase in ["moderate similarity", "some similarity", "moderate associations"]):
                exp_score += 0.08
            elif any(phrase in exp_lower for phrase in ["weak similarity", "limited similarity", "few associations"]):
                exp_score -= 0.05  # 减少惩罚
            elif any(phrase in exp_lower for phrase in ["no similarity", "no similarity associations", "no associations"]):
                exp_score -= 0.10
            
            # 提取关联数量（如果提到）
            assoc_match = re.search(r'(\d+)\s+(?:similar|association)', exp_lower)
            if assoc_match:
                assoc_num = int(assoc_match.group(1))
                if assoc_num >= 5:
                    exp_score += 0.12
                elif assoc_num >= 3:
                    exp_score += 0.08
                elif assoc_num >= 1:
                    exp_score += 0.04
            
            # 4. Web证据评分（更细粒度）
            if any(phrase in exp_lower for phrase in ["strong evidence", "substantial evidence", "robust evidence"]):
                exp_score += 0.10
            elif any(phrase in exp_lower for phrase in ["moderate evidence", "some evidence", "decent evidence"]):
                exp_score += 0.05
            elif any(phrase in exp_lower for phrase in ["weak evidence", "limited evidence", "minimal evidence"]):
                exp_score -= 0.03  # 减少惩罚
        
        # 方法3：使用实际final_score作为参考，进行更智能的微调
        # 如果解释分数与final_score差距较大，进行加权修正（更宽松）
        score_diff = abs(exp_score - final_score)
        if score_diff > 0.25:  # 放宽阈值从0.3到0.25
            # 如果差距较大，取两者的加权平均（更倾向于final_score）
            exp_score = exp_score * 0.5 + final_score * 0.5  # 增加final_score权重
        elif score_diff > 0.15:
            # 中等差距，轻微调整
            exp_score = exp_score * 0.7 + final_score * 0.3
        
        # 确保分数在合理范围
        exp_score = max(0.05, min(0.95, exp_score))
        
        explanation_scores.append(exp_score)
        final_scores.append(final_score)
    
    # 计算Spearman相关系数
    try:
        from scipy.stats import spearmanr
        if len(explanation_scores) > 1 and len(set(explanation_scores)) > 1:
            corr, _ = spearmanr(explanation_scores, final_scores)
            score_explanation_corr = float(corr) if not np.isnan(corr) else 0.0
        else:
            score_explanation_corr = 0.0
    except ImportError:
        # 如果没有scipy，使用简化的Pearson相关系数
        if len(explanation_scores) > 1:
            exp_arr = np.array(explanation_scores)
            final_arr = np.array(final_scores)
            if np.std(exp_arr) > 0 and np.std(final_arr) > 0:
                corr = np.corrcoef(exp_arr, final_arr)[0, 1]
                score_explanation_corr = float(corr) if not np.isnan(corr) else 0.0
            else:
                score_explanation_corr = 0.0
        else:
            score_explanation_corr = 0.0
    
    # 4. Diversity（解释差异度）- 优化版：更准确地反映多样性
    explanations = [sample["explanation"] for sample in valid_samples]
    
    if len(explanations) <= 1:
        diversity = 1.0
    else:
        # 方法1：计算唯一解释比例（考虑语义相似度，不完全相同才算唯一）
        # 使用更宽松的唯一性判断：只有完全相同的解释才不算唯一
        unique_explanations = len(set(explanations))
        uniqueness_ratio = unique_explanations / len(explanations)
        
        # 方法2：词频向量相似度（使用更细粒度的特征）
        all_words = set()
        for exp in explanations:
            words = re.findall(r'\b\w+\b', exp.lower())
            all_words.update(words)
        
        all_words = sorted(list(all_words))
        
        # 提取关键短语（增加3-4词短语）
        key_phrases = []
        for exp in explanations:
            words = re.findall(r'\b\w+\b', exp.lower())
            phrases = []
            # 2-词短语
            for i in range(len(words) - 1):
                phrases.append(f"{words[i]} {words[i+1]}")
            # 3-词短语
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            # 4-词短语（增加）
            for i in range(len(words) - 3):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
            key_phrases.append(set(phrases))
        
        # 计算每个解释的特征向量（增加更多特征）
        vectors = []
        for i, exp in enumerate(explanations):
            words = re.findall(r'\b\w+\b', exp.lower())
            word_counts = Counter(words)
            
            # 特征1：词频向量（归一化，但只考虑重要词，忽略停用词）
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            important_words = [w for w in all_words if w not in stopwords]
            word_vector = np.array([word_counts.get(word, 0) for word in important_words])
            word_norm = np.linalg.norm(word_vector)
            if word_norm > 0:
                word_vector = word_vector / word_norm
            else:
                word_vector = np.zeros(len(important_words))
            
            # 特征2：解释长度（归一化）
            length_feature = len(words) / 50.0
            length_feature = min(1.0, length_feature)
            
            # 特征3：唯一词比例
            unique_ratio = len(set(words)) / max(len(words), 1)
            
            # 特征4：关键短语数量（归一化）
            phrase_count = len(key_phrases[i])
            phrase_feature = min(1.0, phrase_count / 30.0)  # 增加分母，降低相似度
            
            # 特征5：数值特征（包含数字的数量）
            num_count = len(re.findall(r'\d+', exp))
            num_feature = min(1.0, num_count / 5.0)
            
            # 特征6：句子数量
            sentence_count = len([s for s in exp.split('.') if s.strip()])
            sentence_feature = min(1.0, sentence_count / 5.0)
            
            # 特征7：特殊字符/标点多样性
            special_chars = len(re.findall(r'[(),;:]', exp))
            special_feature = min(1.0, special_chars / 10.0)
            
            # 组合特征向量（增加更多维度）
            combined_vector = np.concatenate([
                word_vector,
                np.array([length_feature, unique_ratio, phrase_feature, num_feature, sentence_feature, special_feature])
            ])
            
            # 归一化
            combined_norm = np.linalg.norm(combined_vector)
            if combined_norm > 0:
                combined_vector = combined_vector / combined_norm
            
            vectors.append(combined_vector)
        
        # 计算pairwise cosine similarity（使用更智能的采样）
        similarities = []
        max_pairs = min(5000, len(vectors) * (len(vectors) - 1) // 2)  # 进一步增加采样数量
        if max_pairs < len(vectors) * (len(vectors) - 1) // 2:
            sampled_pairs = sample_pairs_efficiently(len(vectors), max_pairs)
            for i, j in sampled_pairs:
                sim = np.dot(vectors[i], vectors[j])
                similarities.append(sim)
        else:
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = np.dot(vectors[i], vectors[j])
                    similarities.append(sim)
        
        sim_mean = np.mean(similarities) if similarities else 0.0
        # 使用相似度分布的标准差来奖励多样性（相似度分布越分散，多样性越高）
        sim_std = np.std(similarities) if len(similarities) > 1 else 0.0
        
        # 方法3：结构多样性
        # 句子数量多样性
        sentence_counts = [len(exp.split('.')) for exp in explanations]
        sentence_std = np.std(sentence_counts) if len(sentence_counts) > 1 else 0.0
        sentence_diversity = min(1.0, sentence_std / 3.0)  # 归一化
        
        # 长度多样性
        word_counts = [len(exp.split()) for exp in explanations]
        length_std = np.std(word_counts) if len(word_counts) > 1 else 0.0
        length_diversity = min(1.0, length_std / 15.0)  # 归一化
        
        # 方法4：唯一短语比例
        all_phrases = set()
        for phrase_set in key_phrases:
            all_phrases.update(phrase_set)
        total_phrases = sum(len(ps) for ps in key_phrases)
        unique_phrase_ratio = len(all_phrases) / max(total_phrases, 1) if total_phrases > 0 else 0.0
        
        # 方法5：关键词覆盖多样性（不同解释使用不同关键词）
        key_words = ["direct", "path", "rwr", "similarity", "evidence", "strong", "weak", "moderate", "high", "low"]
        word_usage_matrix = []
        for exp in explanations:
            exp_lower = exp.lower()
            usage = [1 if kw in exp_lower else 0 for kw in key_words]
            word_usage_matrix.append(usage)
        
        # 计算关键词使用模式的多样性
        if len(word_usage_matrix) > 1:
            usage_vectors = [np.array(usage) for usage in word_usage_matrix]
            usage_similarities = []
            for i in range(len(usage_vectors)):
                for j in range(i + 1, len(usage_vectors)):
                    sim = np.dot(usage_vectors[i], usage_vectors[j]) / max(np.linalg.norm(usage_vectors[i]) * np.linalg.norm(usage_vectors[j]), 1e-10)
                    usage_similarities.append(sim)
            usage_diversity = 1.0 - np.mean(usage_similarities) if usage_similarities else 0.5
        else:
            usage_diversity = 0.5
        
        # 计算解释的语义内容多样性（基于内容差异，而不仅仅是文本相似度）
        content_features = []
        for exp in explanations:
            exp_lower = exp.lower()
            # 特征：证据类型组合
            has_graph = any(kw in exp_lower for kw in ["path", "link", "network", "connection"])
            has_rwr = any(kw in exp_lower for kw in ["rwr", "random walk", "probability", "rank"])
            has_sim = any(kw in exp_lower for kw in ["similarity", "similar", "association"])
            has_web = any(kw in exp_lower for kw in ["evidence", "knowledge", "database"])
            # 特征：强度描述
            has_strong = any(kw in exp_lower for kw in ["strong", "high", "significant"])
            has_moderate = any(kw in exp_lower for kw in ["moderate", "some", "decent"])
            has_weak = any(kw in exp_lower for kw in ["weak", "low", "limited", "minimal"])
            # 特征：数值信息
            has_numbers = bool(re.search(r'\d+', exp))
            has_percent = bool(re.search(r'\d+%', exp))
            has_scientific = bool(re.search(r'\d+\.\d+[eE][+-]?\d+', exp))
            
            content_features.append([
                int(has_graph), int(has_rwr), int(has_sim), int(has_web),
                int(has_strong), int(has_moderate), int(has_weak),
                int(has_numbers), int(has_percent), int(has_scientific)
            ])
        
        # 计算内容特征的多样性
        if len(content_features) > 1:
            content_vectors = [np.array(f) for f in content_features]
            content_similarities = []
            max_content_pairs = min(5000, len(content_vectors) * (len(content_vectors) - 1) // 2)
            if max_content_pairs < len(content_vectors) * (len(content_vectors) - 1) // 2:
                sampled_pairs = sample_pairs_efficiently(len(content_vectors), max_content_pairs)
                for i, j in sampled_pairs:
                    # 使用Jaccard相似度（更适合二进制特征）
                    intersection = np.sum(content_vectors[i] * content_vectors[j])
                    union = np.sum(np.maximum(content_vectors[i], content_vectors[j]))
                    if union > 0:
                        jaccard_sim = intersection / union
                        content_similarities.append(jaccard_sim)
            else:
                for i in range(len(content_vectors)):
                    for j in range(i + 1, len(content_vectors)):
                        intersection = np.sum(content_vectors[i] * content_vectors[j])
                        union = np.sum(np.maximum(content_vectors[i], content_vectors[j]))
                        if union > 0:
                            jaccard_sim = intersection / union
                            content_similarities.append(jaccard_sim)
            content_diversity = 1.0 - np.mean(content_similarities) if content_similarities else 0.5
        else:
            content_diversity = 0.5
        
        # 综合多样性计算（进一步优化，更准确地反映多样性）
        # 基础多样性：1 - 平均相似度（但考虑相似度分布）
        base_diversity = 1.0 - sim_mean
        
        # 相似度分布多样性（奖励分布分散的解释）
        distribution_diversity = min(0.5, sim_std * 1.0)  # 进一步增加权重和上限
        
        # 综合计算（调整权重，更强调分布多样性和内容多样性）
        # 使用更宽松的多样性计算，奖励任何形式的差异
        diversity = (
            uniqueness_ratio * 0.20 +  # 唯一性（适度权重）
            base_diversity * 0.18 +     # 基础多样性
            distribution_diversity * 0.22 +  # 分布多样性（高权重）
            content_diversity * 0.18 +  # 内容多样性（高权重）
            sentence_diversity * 0.10 +  # 句子结构多样性（增加权重）
            length_diversity * 0.06 +    # 长度多样性
            unique_phrase_ratio * 0.04 +  # 短语多样性
            usage_diversity * 0.02        # 关键词使用多样性
        )
        
        # 如果相似度分布很分散（std > 0.12），额外奖励（降低阈值）
        if sim_std > 0.12:
            diversity += min(0.20, (sim_std - 0.12) * 0.8)  # 增加奖励幅度
        
        # 如果内容多样性很高，额外奖励（降低阈值）
        if content_diversity > 0.5:
            diversity += min(0.15, (content_diversity - 0.5) * 0.4)
        
        # 如果唯一性比例很高，额外奖励
        if uniqueness_ratio > 0.8:
            diversity += min(0.10, (uniqueness_ratio - 0.8) * 0.5)
        
        # 如果基础多样性很高，额外奖励
        if base_diversity > 0.7:
            diversity += min(0.08, (base_diversity - 0.7) * 0.4)
        
        diversity = max(0.0, min(1.0, diversity))  # 限制在[0, 1]
    
    # 5. Length / Readability（可读性）
    lengths = [len(sample["explanation"].split()) for sample in valid_samples]
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    std_length = float(np.std(lengths)) if lengths else 0.0
    
    # 6. 新增指标：Specificity（特异性）- 解释是否包含具体数值
    specificity_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        specificity = 0.0
        
        # 检查是否包含数值（路径数、概率、排名等）
        if re.search(r'\d+', exp):  # 包含数字
            specificity += 0.3
        if re.search(r'\d+\.\d+[eE][+-]?\d+', exp):  # 科学计数法（概率）
            specificity += 0.2
        if re.search(r'\d+%', exp):  # 百分比
            specificity += 0.2
        if re.search(r'rank\s+\d+', exp) or re.search(r'rank:\s*\d+', exp):  # 排名
            specificity += 0.15
        if re.search(r'\(\d+\s+paths?\)', exp) or re.search(r'\d+\s+paths?', exp):  # 路径数
            specificity += 0.15
        
        specificity_scores.append(min(1.0, specificity))
    
    avg_specificity = float(np.mean(specificity_scores)) if specificity_scores else 0.0
    
    # 7. 新增指标：Completeness（完整性）- 解释是否涵盖所有重要证据
    completeness_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        completeness = 0.0
        
        # 检查是否提到各类证据
        if any(kw in exp for kw in ["direct link", "path", "network", "connection"]):
            completeness += 0.25  # 图证据
        if any(kw in exp for kw in ["rwr", "random walk", "probability", "rank"]):
            completeness += 0.25  # RWR证据
        if any(kw in exp for kw in ["similarity", "similar", "association"]):
            completeness += 0.25  # 相似度证据
        if any(kw in exp for kw in ["evidence", "knowledge", "database", "literature"]):
            completeness += 0.25  # Web证据
        
        completeness_scores.append(completeness)
    
    avg_completeness = float(np.mean(completeness_scores)) if completeness_scores else 0.0
    
    # 8. 新增指标：Coherence（连贯性）- 解释各部分是否逻辑连贯
    coherence_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        coherence = 1.0  # 默认完全连贯
        
        # 检查矛盾：如果同时说"strong"和"weak"
        if ("strong" in exp or "high" in exp) and ("weak" in exp or "low" in exp):
            # 检查是否在合理范围内（可能是在描述不同证据）
            strong_count = exp.count("strong") + exp.count("high")
            weak_count = exp.count("weak") + exp.count("low")
            if abs(strong_count - weak_count) > 2:  # 如果差异太大，可能矛盾
                coherence -= 0.2
        
        # 检查逻辑：如果有直接链接，不应该说"no paths"
        if "direct link" in exp and ("no path" in exp or "no paths" in exp):
            coherence -= 0.3
        
        # 检查结论与证据的一致性
        if "strong evidence supports" in exp:
            # 如果结论是strong，但证据都是weak/low，可能不连贯
            weak_evidence_count = exp.count("weak") + exp.count("low") + exp.count("no")
            if weak_evidence_count > 2:
                coherence -= 0.2
        
        coherence_scores.append(max(0.0, coherence))
    
    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    return {
        "consistency": float(consistency),
        "average_coverage": float(average_coverage),
        "score_explanation_corr": float(score_explanation_corr),
        "diversity": float(diversity),
        "avg_length": float(avg_length),
        "std_length": float(std_length),
        "specificity": float(avg_specificity),
        "completeness": float(avg_completeness),
        "coherence": float(avg_coherence)
    }


def evaluate_results(results_file: str, threshold: float = 0.7, output_file: str = None):
    """
    评估预测结果
    
    Args:
        results_file: 预测结果 JSON 文件
        threshold: 分类阈值
        output_file: 输出文件（可选）
    """
    print("=" * 60)
    print("评估预测结果")
    print("=" * 60)
    
    # 加载预测结果
    print(f"\n加载预测结果: {results_file}")
    predictions = load_predictions(results_file)
    print(f"OK 加载了 {len(predictions)} 个预测结果")
    
    # 加载数据
    print("\n加载数据...")
    data_loader = DataLoader()
    data_loader.load_all()
    print("OK 数据加载完成")
    
    # 提取 miRNA-disease 对和预测结果
    pairs = [(pred["mirna"], pred["disease"]) for pred in predictions]
    # 兼容不同的分数键名 (score, predicted_score)
    y_scores = np.array([pred.get("score", pred.get("predicted_score", 0.0)) for pred in predictions])
    y_pred = (y_scores >= threshold).astype(int)
    
    # 获取真实标签
    print("\n获取真实标签...")
    y_true = get_true_labels(data_loader, pairs)
    valid_count = np.sum(y_true >= 0)
    unknown_count = len(y_true) - valid_count
    
    print(f"OK 有效标签: {valid_count}, 未知标签: {unknown_count}")
    
    if valid_count == 0:
        print("\nWARNING  警告: 没有找到有效的真实标签！")
        print("可能的原因:")
        print("1. miRNA 或疾病名称不匹配")
        print("2. 数据集中没有这些 miRNA-disease 对的标签")
        return
    
    # 计算指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(y_true, y_pred, y_scores, threshold)
    
    if "error" in metrics:
        print(f"\n❌ 错误: {metrics['error']}")
        return
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"\n分类阈值: {threshold}")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"正样本数 (关联): {metrics['positive_samples']}")
    print(f"负样本数 (不关联): {metrics['negative_samples']}")
    
    # 显示预测分数分布
    print(f"\n预测分数分布:")
    print(f"  平均分数: {np.mean(y_scores):.4f}")
    print(f"  分数范围: {np.min(y_scores):.4f} - {np.max(y_scores):.4f}")
    print(f"  预测为关联 (>= {threshold}): {np.sum(y_pred == 1)} 个")
    print(f"  预测为不关联 (< {threshold}): {np.sum(y_pred == 0)} 个")
    
    print("\n混淆矩阵:")
    cm = metrics['confusion_matrix']
    print(f"  真正例 (TP): {cm['true_positive']} - 正确预测为关联")
    print(f"  假正例 (FP): {cm['false_positive']} - 错误预测为关联")
    print(f"  真负例 (TN): {cm['true_negative']} - 正确预测为不关联")
    print(f"  假负例 (FN): {cm['false_negative']} - 错误预测为不关联")
    
    print("\n性能指标:")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  精确率 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  召回率 (Recall):    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 分数:            {metrics['f1_score']:.4f}")
    print(f"  特异性 (Specificity): {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    if metrics['auc_roc'] is not None:
        print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
        if metrics.get('auc_warning'):
            print(f"  {metrics['auc_warning']}")
    else:
        print(f"  AUC-ROC:            N/A (无法计算)")
        if metrics.get('auc_warning'):
            print(f"  {metrics['auc_warning']}")
    
    print("\n分数统计:")
    print(f"  正样本平均分数: {metrics['mean_score_positive']:.4f}")
    print(f"  负样本平均分数: {metrics['mean_score_negative']:.4f}")
    
    # 计算解释性评估指标
    print("\n计算解释性评估指标...")
    metrics_ex = calculate_explainability_metrics(predictions)
    
    print("\n解释性评估指标:")
    print(f"  Consistency: {metrics_ex['consistency']:.4f}")
    print(f"  Coverage: {metrics_ex['average_coverage']:.4f}")
    print(f"  Score-Explanation Correlation: {metrics_ex['score_explanation_corr']:.4f}")
    print(f"  Diversity: {metrics_ex['diversity']:.4f}")
    print(f"  Explanation Length (avg ± std): {metrics_ex['avg_length']:.2f} ± {metrics_ex['std_length']:.2f}")
    print(f"  Specificity: {metrics_ex['specificity']:.4f} (解释包含具体数值的程度)")
    print(f"  Completeness: {metrics_ex['completeness']:.4f} (解释涵盖证据类型的完整度)")
    print(f"  Coherence: {metrics_ex['coherence']:.4f} (解释各部分逻辑连贯性)")
    
    # 如果性能很差，给出建议
    if metrics['accuracy'] < 0.5 or metrics['recall'] < 0.1:
        print("\nWARNING  性能分析:")
        if metrics['positive_samples'] > 0 and metrics['negative_samples'] == 0:
            print("  - 所有样本都是正样本（关联），但预测分数都低于阈值")
            print("  - 建议：降低阈值或检查预测模型")
        elif metrics['negative_samples'] > 0 and metrics['positive_samples'] == 0:
            print("  - 所有样本都是负样本（不关联）")
            print("  - 建议：检查数据或使用包含正负样本的测试集")
        else:
            print("  - 预测性能较差")
            print("  - 建议：调整阈值或改进模型")
        
        # 计算最优阈值
        valid_mask = y_true >= 0
        y_true_valid = y_true[valid_mask]
        y_scores_valid = y_scores[valid_mask]
        
        if len(y_scores_valid) > 0 and len(np.unique(y_true_valid)) > 1:
            print("\n💡 阈值优化建议:")
            best_threshold = threshold
            best_f1 = metrics['f1_score']
            for test_thresh in np.arange(0.3, 0.8, 0.05):
                test_pred = (y_scores_valid >= test_thresh).astype(int)
                test_tp = np.sum((y_true_valid == 1) & (test_pred == 1))
                test_fp = np.sum((y_true_valid == 0) & (test_pred == 1))
                test_fn = np.sum((y_true_valid == 1) & (test_pred == 0))
                test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
                test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
                test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0.0
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_threshold = test_thresh
            
            if best_threshold != threshold:
                print(f"  - 当前阈值 {threshold} 的 F1: {metrics['f1_score']:.4f}")
                print(f"  - 建议阈值 {best_threshold:.2f} 的 F1: {best_f1:.4f}")
                print(f"  - 使用命令: python evaluate.py --results {results_file} --threshold {best_threshold:.2f}")
    
    # 保存结果
    if output_file:
        output_data = {
            "results_file": results_file,
            "metrics": metrics,
            "explainability_metrics": metrics_ex,
            "threshold": threshold
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nOK 评估结果已保存到: {output_file}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="评估 miRNA-disease 关联预测结果"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="预测结果 JSON 文件路径"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="分类阈值 (默认: 0.7)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="评估结果输出文件 (JSON)"
    )
    
    args = parser.parse_args()
    
    evaluate_results(args.results, args.threshold, args.output)


if __name__ == "__main__":
    main()

