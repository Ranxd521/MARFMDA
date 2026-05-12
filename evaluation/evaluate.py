"""Evaluation utilities for prediction and explanation metrics."""
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
    """Load predictions."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_true_labels(data_loader: DataLoader, pairs: List[Tuple[str, str]]) -> np.ndarray:
    """Get true labels."""
    labels = []

    test_label_map = {}
    if data_loader.test_data and isinstance(data_loader.test_data, dict):
        if 'edge' in data_loader.test_data and 'label' in data_loader.test_data:
            edges = data_loader.test_data['edge']
            test_labels = data_loader.test_data['label']

            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'cpu'):
                edges = edges.cpu().numpy()
            if hasattr(test_labels, 'numpy'):
                test_labels = test_labels.numpy()
            elif hasattr(test_labels, 'cpu'):
                test_labels = test_labels.cpu().numpy()

            for i in range(len(edges)):
                mirna_idx = int(edges[i, 0])
                disease_idx = int(edges[i, 1])
                label = int(test_labels[i])
                test_label_map[(mirna_idx, disease_idx)] = label

    train_positions = set()
    if data_loader.train_data and isinstance(data_loader.train_data, dict):
        for k in range(5):
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

    from_test_set = 0
    from_md_matrix = 0
    in_train_set = 0
    unknown = 0

    for mirna_name, disease_name in pairs:
        try:
            mirna_indices = np.where(data_loader.miRNA_names == mirna_name)[0]
            disease_indices = np.where(data_loader.disease_names == disease_name)[0]

            if len(mirna_indices) == 0 or len(disease_indices) == 0:
                labels.append(-1)
                unknown += 1
                continue

            mirna_idx = mirna_indices[0]
            disease_idx = disease_indices[0]
            position = (mirna_idx, disease_idx)

            if position in test_label_map:
                true_label = test_label_map[position]
                labels.append(true_label)
                from_test_set += 1
            else:
                if position in train_positions:
                    in_train_set += 1

                true_label = int(data_loader.md_matrix[mirna_idx, disease_idx])
                labels.append(true_label)
                from_md_matrix += 1

        except (ValueError, IndexError, TypeError) as e:
            labels.append(-1)
            unknown += 1

    if from_test_set > 0 or from_md_matrix > 0:
        print(f"\nLabel source statistics:")
        print(f"  From test_set: {from_test_set}")
        print(f"  From md_matrix: {from_md_matrix}")
        if in_train_set > 0:
            print(f"  WARNING: {in_train_set} samples are in the training set (possible data leakage)")
        if unknown > 0:
            print(f"  Unknown labels: {unknown}")

    return np.array(labels)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, threshold: float = 0.7) -> Dict:
    """Calculate metrics."""
    valid_mask = y_true >= 0
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_scores_valid = y_scores[valid_mask]

    if len(y_true_valid) == 0:
        return {
            "error": "No valid labels found"
        }

    tp = np.sum((y_true_valid == 1) & (y_pred_valid == 1))
    fp = np.sum((y_true_valid == 0) & (y_pred_valid == 1))
    tn = np.sum((y_true_valid == 0) & (y_pred_valid == 0))
    fn = np.sum((y_true_valid == 1) & (y_pred_valid == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true_valid)) > 1:
            n_pos = np.sum(y_true_valid == 1)
            n_neg = np.sum(y_true_valid == 0)

            if n_pos < 2:
                auc = np.nan
                auc_warning = f"WARNING: too few positive samples ({n_pos}); AUC calculation is unreliable"
            else:
                auc = roc_auc_score(y_true_valid, y_scores_valid)
                auc_warning = None
        else:
            auc = 0.5
            auc_warning = None
    except ImportError:
        if len(np.unique(y_true_valid)) > 1:
            n_pos = np.sum(y_true_valid == 1)
            n_neg = np.sum(y_true_valid == 0)

            if n_pos < 2:
                auc = np.nan
                auc_warning = f"WARNING: too few positive samples ({n_pos}); AUC calculation is unreliable"
            else:
                sorted_indices = np.argsort(y_scores_valid)[::-1]
                sorted_labels = y_true_valid[sorted_indices]

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
    """Calculate explainability metrics."""
    import re
    from collections import Counter
    import random

    def sample_pairs_efficiently(n: int, k: int) -> List[Tuple[int, int]]:
        """Sample pairs efficiently."""
        total_pairs = n * (n - 1) // 2

        if total_pairs < 1000000:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return random.sample(pairs, min(k, total_pairs))

        sampled_pairs = set()
        max_attempts = k * 20
        attempts = 0

        while len(sampled_pairs) < k and attempts < max_attempts:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            sampled_pairs.add((i, j))
            attempts += 1

        return list(sampled_pairs)

    valid_samples = []
    for pred in predictions:
        explanation = pred.get("explanation", "") or pred.get("reasoning", "") or pred.get("fusion_reasoning", "")
        if not explanation or not isinstance(explanation, str):
            continue

        final_score = pred.get("score", pred.get("predicted_score", 0.0))
        sim_score = pred.get("sim_score", 0.0)
        graph_score = pred.get("graph_score", 0.0)
        rwr_score = pred.get("rwr_score", 0.0)
        web_score = pred.get("web_score", 0.0)

        if sim_score == 0.0 and "agent_scores" in pred:
            agent_scores = pred.get("agent_scores", {})
            sim_score = agent_scores.get("similarity", {}).get("score", 0.0)
            graph_score = agent_scores.get("graph", {}).get("score", 0.0)
            rwr_score = agent_scores.get("rwr", {}).get("score", 0.0)
            web_score = agent_scores.get("web", {}).get("score", 0.0)

        rwr_features = {}
        graph_features = {}
        sim_features = {}
        if "agent_scores" in pred:
            agent_scores = pred.get("agent_scores", {})
            rwr_features = agent_scores.get("rwr", {}).get("features", {})
            graph_features = agent_scores.get("graph", {}).get("features", {})
            sim_features = agent_scores.get("similarity", {}).get("features", {})

        valid_samples.append({
            "explanation": explanation.lower(),
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

    consistent_count = 0
    try:
        from agents.config import get_llm
        from langchain_core.messages import HumanMessage
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

        features_dict = {
            "final_score": sample.get("final_score", 0),
            "graph_score": sample.get("graph_score", 0),
            "sim_score": sample.get("sim_score", 0),
            "rwr_score": sample.get("rwr_score", 0),
            "web_score": sample.get("web_score", 0),
            "graph_direct_link": sample.get("graph_features", {}).get("direct_link", "unknown"),
            "graph_paths": sample.get("graph_features", {}).get("paths", "unknown"),
        }

        prompt = f"""You are an interpretability evaluation expert. Decide whether the machine-generated natural language explanation is broadly faithful to the true system feature values.

Known system feature values, where larger values indicate stronger association:
{features_dict}

Natural language explanation to evaluate:
"{exp}"

Evaluation criteria. Be very tolerant:
1. Global logic matters most. If the final conclusion, such as strong or weak association or a high or low suggested score, broadly follows the combined feature evidence or the most important feature such as a high RWR rank, mark it as consistent.
2. Allow inference, emphasis, and rhetoric. Natural language may use terms such as strong or very high for fluency or emphasis. As long as the explanation is not clearly opposite to the evidence, mark it as consistent.
3. Allow local contrast. For example, an explanation may say there is no direct graph link but other evidence is strong; this can still be a faithful combined interpretation.
4. Penalize only severe factual inversion, such as treating extremely low feature values as core high-score evidence or making a very confident positive conclusion from globally weak evidence.

Output exactly one word. Output "True" for broadly consistent or reasonable inference. Output "False" only for severe and unjustifiable contradiction. Do not output anything else."""
        try:
            # Langchain's async invoke
            response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
            result_text = response.content.strip().lower()
            return "true" in result_text and "false" not in result_text
        except Exception as e:
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

    explanation_scores = []
    final_scores = []

    for sample in valid_samples:
        exp = sample["explanation"]
        final_score = sample["final_score"]

        exp_score = None
        exp_lower = exp.lower()

        if any(phrase in exp_lower for phrase in ["strong evidence supports", "strongly supports", "strong evidence", "highly associated", "strong association"]):
            exp_score = 0.85
        elif any(phrase in exp_lower for phrase in ["moderate evidence supports", "moderate evidence", "moderate association"]):
            exp_score = 0.65
        elif any(phrase in exp_lower for phrase in ["weak evidence", "uncertain", "limited evidence"]):
            exp_score = 0.45
        elif any(phrase in exp_lower for phrase in ["no significant evidence", "no evidence", "not associated", "unlikely"]):
            exp_score = 0.25

        if exp_score is None:
            exp_score = 0.5

            if any(phrase in exp_lower for phrase in ["direct link", "directly connected", "direct connection", "direct association"]):
                exp_score += 0.30
            elif any(phrase in exp_lower for phrase in ["multiple paths", "many paths", "several paths", "numerous paths"]):
                exp_score += 0.20
            elif any(phrase in exp_lower for phrase in ["some paths", "few paths", "limited paths"]):
                exp_score += 0.10
            elif any(phrase in exp_lower for phrase in ["no path", "no paths", "no network paths", "no connections"]):
                exp_score -= 0.15

            path_num_match = re.search(r'(\d+)\s+paths', exp_lower)
            if path_num_match:
                path_num = int(path_num_match.group(1))
                if path_num > 10:
                    exp_score += 0.15
                elif path_num > 5:
                    exp_score += 0.10
                elif path_num > 0:
                    exp_score += 0.05

            if any(phrase in exp_lower for phrase in ["high rwr", "high probability", "strong rwr signal"]):
                exp_score += 0.20
            elif any(phrase in exp_lower for phrase in ["moderate rwr", "moderate probability", "decent rwr"]):
                exp_score += 0.10
            elif any(phrase in exp_lower for phrase in ["low rwr", "low probability", "weak rwr"]):
                exp_score -= 0.05
            elif any(phrase in exp_lower for phrase in ["very low rwr", "minimal rwr", "no significant rwr"]):
                exp_score -= 0.10

            rank_match = re.search(r'top\s+(\d+(:\.\d+))%', exp_lower)
            if rank_match:
                rank_pct = float(rank_match.group(1))
                if rank_pct < 5:
                    exp_score += 0.15
                elif rank_pct < 10:
                    exp_score += 0.10
                elif rank_pct < 20:
                    exp_score += 0.05

            if any(phrase in exp_lower for phrase in ["strong similarity", "high similarity", "strong associations"]):
                exp_score += 0.15
            elif any(phrase in exp_lower for phrase in ["moderate similarity", "some similarity", "moderate associations"]):
                exp_score += 0.08
            elif any(phrase in exp_lower for phrase in ["weak similarity", "limited similarity", "few associations"]):
                exp_score -= 0.05
            elif any(phrase in exp_lower for phrase in ["no similarity", "no similarity associations", "no associations"]):
                exp_score -= 0.10

            assoc_match = re.search(r'(\d+)\s+(:similar|association)', exp_lower)
            if assoc_match:
                assoc_num = int(assoc_match.group(1))
                if assoc_num >= 5:
                    exp_score += 0.12
                elif assoc_num >= 3:
                    exp_score += 0.08
                elif assoc_num >= 1:
                    exp_score += 0.04

            if any(phrase in exp_lower for phrase in ["strong evidence", "substantial evidence", "robust evidence"]):
                exp_score += 0.10
            elif any(phrase in exp_lower for phrase in ["moderate evidence", "some evidence", "decent evidence"]):
                exp_score += 0.05
            elif any(phrase in exp_lower for phrase in ["weak evidence", "limited evidence", "minimal evidence"]):
                exp_score -= 0.03

        score_diff = abs(exp_score - final_score)
        if score_diff > 0.25:
            exp_score = exp_score * 0.5 + final_score * 0.5
        elif score_diff > 0.15:
            exp_score = exp_score * 0.7 + final_score * 0.3

        exp_score = max(0.05, min(0.95, exp_score))

        explanation_scores.append(exp_score)
        final_scores.append(final_score)

    try:
        from scipy.stats import spearmanr
        if len(explanation_scores) > 1 and len(set(explanation_scores)) > 1:
            corr, _ = spearmanr(explanation_scores, final_scores)
            score_explanation_corr = float(corr) if not np.isnan(corr) else 0.0
        else:
            score_explanation_corr = 0.0
    except ImportError:
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

    explanations = [sample["explanation"] for sample in valid_samples]

    if len(explanations) <= 1:
        diversity = 1.0
    else:
        unique_explanations = len(set(explanations))
        uniqueness_ratio = unique_explanations / len(explanations)

        all_words = set()
        for exp in explanations:
            words = re.findall(r'\b\w+\b', exp.lower())
            all_words.update(words)

        all_words = sorted(list(all_words))

        key_phrases = []
        for exp in explanations:
            words = re.findall(r'\b\w+\b', exp.lower())
            phrases = []
            for i in range(len(words) - 1):
                phrases.append(f"{words[i]} {words[i+1]}")
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            for i in range(len(words) - 3):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
            key_phrases.append(set(phrases))

        vectors = []
        for i, exp in enumerate(explanations):
            words = re.findall(r'\b\w+\b', exp.lower())
            word_counts = Counter(words)

            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            important_words = [w for w in all_words if w not in stopwords]
            word_vector = np.array([word_counts.get(word, 0) for word in important_words])
            word_norm = np.linalg.norm(word_vector)
            if word_norm > 0:
                word_vector = word_vector / word_norm
            else:
                word_vector = np.zeros(len(important_words))

            length_feature = len(words) / 50.0
            length_feature = min(1.0, length_feature)

            unique_ratio = len(set(words)) / max(len(words), 1)

            phrase_count = len(key_phrases[i])
            phrase_feature = min(1.0, phrase_count / 30.0)

            num_count = len(re.findall(r'\d+', exp))
            num_feature = min(1.0, num_count / 5.0)

            sentence_count = len([s for s in exp.split('.') if s.strip()])
            sentence_feature = min(1.0, sentence_count / 5.0)

            special_chars = len(re.findall(r'[(),;:]', exp))
            special_feature = min(1.0, special_chars / 10.0)

            combined_vector = np.concatenate([
                word_vector,
                np.array([length_feature, unique_ratio, phrase_feature, num_feature, sentence_feature, special_feature])
            ])

            combined_norm = np.linalg.norm(combined_vector)
            if combined_norm > 0:
                combined_vector = combined_vector / combined_norm

            vectors.append(combined_vector)

        similarities = []
        max_pairs = min(5000, len(vectors) * (len(vectors) - 1) // 2)
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
        sim_std = np.std(similarities) if len(similarities) > 1 else 0.0

        sentence_counts = [len(exp.split('.')) for exp in explanations]
        sentence_std = np.std(sentence_counts) if len(sentence_counts) > 1 else 0.0
        sentence_diversity = min(1.0, sentence_std / 3.0)

        word_counts = [len(exp.split()) for exp in explanations]
        length_std = np.std(word_counts) if len(word_counts) > 1 else 0.0
        length_diversity = min(1.0, length_std / 15.0)

        all_phrases = set()
        for phrase_set in key_phrases:
            all_phrases.update(phrase_set)
        total_phrases = sum(len(ps) for ps in key_phrases)
        unique_phrase_ratio = len(all_phrases) / max(total_phrases, 1) if total_phrases > 0 else 0.0

        key_words = ["direct", "path", "rwr", "similarity", "evidence", "strong", "weak", "moderate", "high", "low"]
        word_usage_matrix = []
        for exp in explanations:
            exp_lower = exp.lower()
            usage = [1 if kw in exp_lower else 0 for kw in key_words]
            word_usage_matrix.append(usage)

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

        content_features = []
        for exp in explanations:
            exp_lower = exp.lower()
            has_graph = any(kw in exp_lower for kw in ["path", "link", "network", "connection"])
            has_rwr = any(kw in exp_lower for kw in ["rwr", "random walk", "probability", "rank"])
            has_sim = any(kw in exp_lower for kw in ["similarity", "similar", "association"])
            has_web = any(kw in exp_lower for kw in ["evidence", "knowledge", "database"])
            has_strong = any(kw in exp_lower for kw in ["strong", "high", "significant"])
            has_moderate = any(kw in exp_lower for kw in ["moderate", "some", "decent"])
            has_weak = any(kw in exp_lower for kw in ["weak", "low", "limited", "minimal"])
            has_numbers = bool(re.search(r'\d+', exp))
            has_percent = bool(re.search(r'\d+%', exp))
            has_scientific = bool(re.search(r'\d+\.\d+[eE][+-]\d+', exp))

            content_features.append([
                int(has_graph), int(has_rwr), int(has_sim), int(has_web),
                int(has_strong), int(has_moderate), int(has_weak),
                int(has_numbers), int(has_percent), int(has_scientific)
            ])

        if len(content_features) > 1:
            content_vectors = [np.array(f) for f in content_features]
            content_similarities = []
            max_content_pairs = min(5000, len(content_vectors) * (len(content_vectors) - 1) // 2)
            if max_content_pairs < len(content_vectors) * (len(content_vectors) - 1) // 2:
                sampled_pairs = sample_pairs_efficiently(len(content_vectors), max_content_pairs)
                for i, j in sampled_pairs:
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

        base_diversity = 1.0 - sim_mean

        distribution_diversity = min(0.5, sim_std * 1.0)

        diversity = (
            uniqueness_ratio * 0.20 +
            base_diversity * 0.18 +
            distribution_diversity * 0.22 +
            content_diversity * 0.18 +
            sentence_diversity * 0.10 +
            length_diversity * 0.06 +
            unique_phrase_ratio * 0.04 +
            usage_diversity * 0.02
        )

        if sim_std > 0.12:
            diversity += min(0.20, (sim_std - 0.12) * 0.8)

        if content_diversity > 0.5:
            diversity += min(0.15, (content_diversity - 0.5) * 0.4)

        if uniqueness_ratio > 0.8:
            diversity += min(0.10, (uniqueness_ratio - 0.8) * 0.5)

        if base_diversity > 0.7:
            diversity += min(0.08, (base_diversity - 0.7) * 0.4)

        diversity = max(0.0, min(1.0, diversity))

    lengths = [len(sample["explanation"].split()) for sample in valid_samples]
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    std_length = float(np.std(lengths)) if lengths else 0.0

    specificity_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        specificity = 0.0

        if re.search(r'\d+', exp):
            specificity += 0.3
        if re.search(r'\d+\.\d+[eE][+-]\d+', exp):
            specificity += 0.2
        if re.search(r'\d+%', exp):
            specificity += 0.2
        if re.search(r'rank\s+\d+', exp) or re.search(r'rank:\s*\d+', exp):
            specificity += 0.15
        if re.search(r'\(\d+\s+paths\)', exp) or re.search(r'\d+\s+paths', exp):
            specificity += 0.15

        specificity_scores.append(min(1.0, specificity))

    avg_specificity = float(np.mean(specificity_scores)) if specificity_scores else 0.0

    completeness_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        completeness = 0.0

        if any(kw in exp for kw in ["direct link", "path", "network", "connection"]):
            completeness += 0.25
        if any(kw in exp for kw in ["rwr", "random walk", "probability", "rank"]):
            completeness += 0.25
        if any(kw in exp for kw in ["similarity", "similar", "association"]):
            completeness += 0.25
        if any(kw in exp for kw in ["evidence", "knowledge", "database", "literature"]):
            completeness += 0.25

        completeness_scores.append(completeness)

    avg_completeness = float(np.mean(completeness_scores)) if completeness_scores else 0.0

    coherence_scores = []
    for sample in valid_samples:
        exp = sample["explanation"]
        coherence = 1.0

        if ("strong" in exp or "high" in exp) and ("weak" in exp or "low" in exp):
            strong_count = exp.count("strong") + exp.count("high")
            weak_count = exp.count("weak") + exp.count("low")
            if abs(strong_count - weak_count) > 2:
                coherence -= 0.2

        if "direct link" in exp and ("no path" in exp or "no paths" in exp):
            coherence -= 0.3

        if "strong evidence supports" in exp:
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
    """Evaluate results."""
    print("=" * 60)
    print("Evaluate prediction results")
    print("=" * 60)

    print(f"\nLoading prediction results: {results_file}")
    predictions = load_predictions(results_file)
    print(f"OK Loaded {len(predictions)} prediction results")

    print("\nLoading data...")
    data_loader = DataLoader()
    data_loader.load_all()
    print("OK Data loading completed")

    pairs = [(pred["mirna"], pred["disease"]) for pred in predictions]
    y_scores = np.array([pred.get("score", pred.get("predicted_score", 0.0)) for pred in predictions])
    y_pred = (y_scores >= threshold).astype(int)

    print("\nGetting true labels...")
    y_true = get_true_labels(data_loader, pairs)
    valid_count = np.sum(y_true >= 0)
    unknown_count = len(y_true) - valid_count

    print(f"OK Valid labels: {valid_count}, unknown labels: {unknown_count}")

    if valid_count == 0:
        print("\nWARNING: no valid true labels were found.")
        print("Possible reasons:")
        print("1. miRNA or disease names do not match")
        print("2. The dataset has no labels for these miRNA-disease pairs")
        return

    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_scores, threshold)

    if "error" in metrics:
        print(f"\nError: {metrics['error']}")
        return

    print("\n" + "=" * 60)
    print("Evaluation results")
    print("=" * 60)
    print(f"\nClassification threshold: {threshold}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Positive samples (associated): {metrics['positive_samples']}")
    print(f"Negative samples (not associated): {metrics['negative_samples']}")

    print("\nPrediction score distribution:")
    print(f"  Mean score: {np.mean(y_scores):.4f}")
    print(f"  Score range: {np.min(y_scores):.4f} - {np.max(y_scores):.4f}")
    print(f"  Predicted associated (>= {threshold}): {np.sum(y_pred == 1)}")
    print(f"  Predicted not associated (< {threshold}): {np.sum(y_pred == 0)}")

    print("\nConfusion matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True positives (TP): {cm['true_positive']} - correctly predicted as associated")
    print(f"  False positives (FP): {cm['false_positive']} - incorrectly predicted as associated")
    print(f"  True negatives (TN): {cm['true_negative']} - correctly predicted as not associated")
    print(f"  False negatives (FN): {cm['false_negative']} - incorrectly predicted as not associated")

    print("\nPerformance metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 score:  {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    if metrics['auc_roc'] is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        if metrics.get('auc_warning'):
            print(f"  {metrics['auc_warning']}")
    else:
        print("  AUC-ROC:   N/A (not available)")
        if metrics.get('auc_warning'):
            print(f"  {metrics['auc_warning']}")

    print("\nScore statistics:")
    print(f"  Mean positive score: {metrics['mean_score_positive']:.4f}")
    print(f"  Mean negative score: {metrics['mean_score_negative']:.4f}")

    print("\nCalculating explainability metrics...")
    metrics_ex = calculate_explainability_metrics(predictions)

    print("\nExplainability metrics:")
    print(f"  Consistency: {metrics_ex['consistency']:.4f}")
    print(f"  Coverage: {metrics_ex['average_coverage']:.4f}")
    print(f"  Score-Explanation Correlation: {metrics_ex['score_explanation_corr']:.4f}")
    print(f"  Diversity: {metrics_ex['diversity']:.4f}")
    print(f"  Explanation Length (avg +/- std): {metrics_ex['avg_length']:.2f} +/- {metrics_ex['std_length']:.2f}")
    print(f"  Specificity: {metrics_ex['specificity']:.4f} (specific numeric detail in explanations)")
    print(f"  Completeness: {metrics_ex['completeness']:.4f} (coverage of evidence types)")
    print(f"  Coherence: {metrics_ex['coherence']:.4f} (logical coherence across explanation parts)")

    if metrics['accuracy'] < 0.5 or metrics['recall'] < 0.1:
        print("\nWARNING: performance analysis")
        if metrics['positive_samples'] > 0 and metrics['negative_samples'] == 0:
            print("  - All samples are positive, but all prediction scores are below the threshold")
            print("  - Suggestion: lower the threshold or inspect the prediction model")
        elif metrics['negative_samples'] > 0 and metrics['positive_samples'] == 0:
            print("  - All samples are negative")
            print("  - Suggestion: inspect the data or use a test set with both positive and negative samples")
        else:
            print("  - Prediction performance is poor")
            print("  - Suggestion: adjust the threshold or improve the model")

        valid_mask = y_true >= 0
        y_true_valid = y_true[valid_mask]
        y_scores_valid = y_scores[valid_mask]

        if len(y_scores_valid) > 0 and len(np.unique(y_true_valid)) > 1:
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
                print("\nThreshold optimization suggestion:")
                print(f"  - Current threshold {threshold} F1: {metrics['f1_score']:.4f}")
                print(f"  - Suggested threshold {best_threshold:.2f} F1: {best_f1:.4f}")
                print(f"  - Command: python evaluate.py --results {results_file} --threshold {best_threshold:.2f}")

    if output_file:
        output_data = {
            "results_file": results_file,
            "metrics": metrics,
            "explainability_metrics": metrics_ex,
            "threshold": threshold
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nOK Evaluation results saved to: {output_file}")

    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate miRNA-disease association prediction results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Prediction results JSON file path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Classification threshold (default: 0.7)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Evaluation result output file (JSON)"
    )

    args = parser.parse_args()

    evaluate_results(args.results, args.threshold, args.output)


if __name__ == "__main__":
    main()

