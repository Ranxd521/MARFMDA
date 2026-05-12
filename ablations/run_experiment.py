import sys
import os
import json
import time
import math
import re
import argparse
import asyncio
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from tqdm.asyncio import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

try:
    import torch
except Exception:
    torch = None

# Setup paths (so it runs correctly from workspace root)
sys.path.append(os.getcwd())
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from data_loader.loader import DataLoader
from agents.rwr_agent import RWRAgent
from agents.similarity_agent import SimilarityAgent
from agents.graph_agent import GraphAgent

try:
    from config import get_llm
except ImportError:
    get_llm = None

try:
    from agents.workflow import create_workflow
except ImportError:
    create_workflow = None

# --- Anchors ---
ANCHOR_POS = {
    "text": "ID: ANCHOR_POS | hsa-mir-21 - Lung Neoplasms | Rank: 5 | SimM: 0.95 | SimD: 0.88 | Link: 1 | Paths: 4",
    "output": "ANCHOR_POS | 0.99 | Textbook positive case: Top-tier RWR rank, strong functional clustering, and direct database evidence."
}
ANCHOR_NEG = {
    "text": "ID: ANCHOR_NEG | hsa-mir-dummy - Random Disease | Rank: 2500 | SimM: 0.10 | SimD: 0.05 | Link: 0 | Paths: 0",
    "output": "ANCHOR_NEG | 0.01 | Textbook negative case: No topological or functional evidence found."
}

DEFAULT_INPUT_FILE = r"E:\Multi_Agent_api\dataset\test_set_1000.json"

class OptimizedFeatureExtractor:
    def __init__(self, data_loader, test_pairs=None):
        self.data_loader = data_loader
        self.test_pairs = test_pairs or []
        self.transition_matrix = None
        self.use_gpu = torch is not None and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu") if torch is not None else None
        self.total_nodes = 0
        self.n_mirna = 0
        self.n_disease = 0
        self.n_lncrna = 0
        self._build_rwr_matrix()
        
    def _build_rwr_matrix(self):
        train_md_matrix = self.data_loader.get_train_matrix()
        if self.test_pairs:
            if isinstance(train_md_matrix, np.ndarray):
                train_md_matrix = train_md_matrix.copy()
                for item in self.test_pairs:
                    if item.get('label') == 1:
                        m_idx = self.data_loader.get_miRNA_index(item['mirna'])
                        d_idx = self.data_loader.get_disease_index(item['disease'])
                        if m_idx is not None and d_idx is not None:
                            if train_md_matrix[m_idx, d_idx] != 0:
                                train_md_matrix[m_idx, d_idx] = 0
        
        self.n_mirna = train_md_matrix.shape[0]
        self.n_disease = train_md_matrix.shape[1]
        self.n_lncrna = self.data_loader.ml_matrix.shape[1]
        self.total_nodes = self.n_mirna + self.n_disease + self.n_lncrna
        
        adj_matrix = sp.lil_matrix((self.total_nodes, self.total_nodes))
        def to_n(x): return x if isinstance(x, np.ndarray) else x.toarray()
        
        md = to_n(train_md_matrix)
        ml = to_n(self.data_loader.ml_matrix)
        dl = to_n(self.data_loader.dl_matrix)
        
        adj_matrix[:self.n_mirna, self.n_mirna:self.n_mirna+self.n_disease] = md
        adj_matrix[self.n_mirna:self.n_mirna+self.n_disease, :self.n_mirna] = md.T
        adj_matrix[:self.n_mirna, self.n_mirna+self.n_disease:] = ml
        adj_matrix[self.n_mirna+self.n_disease:, :self.n_mirna] = ml.T
        adj_matrix[self.n_mirna:self.n_mirna+self.n_disease, self.n_mirna+self.n_disease:] = dl
        adj_matrix[self.n_mirna+self.n_disease:, self.n_mirna:self.n_mirna+self.n_disease] = dl.T
        
        adj_matrix = adj_matrix.tocsr()
        row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        d_inv = sp.diags(1.0 / row_sums)
        norm_matrix = d_inv.dot(adj_matrix)
        self.transition_matrix = norm_matrix.T
        
        if self.use_gpu:
            try:
                coo = self.transition_matrix.tocoo()
                i = torch.LongTensor(np.vstack((coo.row, coo.col)))
                v = torch.FloatTensor(coo.data)
                self.transition_matrix_gpu = torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).to(self.device).coalesce()
            except:
                self.use_gpu = False

    def run_rwr_for_mirna(self, mirna_idx, restart_prob=0.3, max_iter=100):
        if self.use_gpu:
            p = torch.zeros((self.total_nodes, 1), device=self.device)
            p[mirna_idx, 0] = 1.0
            p0 = p.clone()
            for _ in range(max_iter):
                 mv = torch.sparse.mm(self.transition_matrix_gpu, p)
                 p_new = (1 - restart_prob) * mv + restart_prob * p0
                 if _ % 10 == 0 and torch.norm(p_new - p) < 1e-6:
                     p = p_new; break
                 p = p_new
            return p[self.n_mirna:self.n_mirna + self.n_disease].flatten().cpu().numpy()
        else:
            p = np.zeros(self.total_nodes)
            p[mirna_idx] = 1.0
            p0 = p.copy()
            for _ in range(max_iter):
                p_new = (1 - restart_prob) * (self.transition_matrix.dot(p)) + restart_prob * p0
                if _ % 10 == 0 and np.linalg.norm(p_new - p) < 1e-6:
                    p = p_new; break
                p = p_new
            return p[self.n_mirna:self.n_mirna + self.n_disease]

def _sample_line(i, sample):
    return (
        f"ID: {i} | {sample['mirna']} - {sample['disease']} | "
        f"Rank: {sample['rwr_rank']} | RWR: {sample['rwr_prob']:.6g} | "
        f"SimM: {sample['sim_mirna_max']:.2f} | SimD: {sample['sim_disease_max']:.2f} | "
        f"SimMCnt: {sample.get('sim_mirna_assoc_count', 0)}/10 | "
        f"SimDCnt: {sample.get('sim_disease_assoc_count', 0)}/10 | "
        f"Link: {sample['direct_link']} | Paths: {sample['paths']} | "
        f"PathMax: {sample.get('path_max_strength', 0.0):.2f}"
    )


def _main_sample_line(i, sample):
    return (
        f"ID: {i} | {sample['mirna']} - {sample['disease']} | "
        f"Rank: {sample['rwr_rank']} | SimM: {sample['sim_mirna_max']:.2f} | "
        f"SimD: {sample['sim_disease_max']:.2f} | Link: {sample['direct_link']} | "
        f"Paths: {sample['paths']}"
    )


def _coerce_score(value):
    score = float(value)
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def construct_batch_prompt(batch_samples, strategy):
    if strategy == "zeroshot_direct":
        prompt = "Rate the probability of association (0.0 to 1.0) for the following miRNA-Disease pairs.\n"
        prompt += "OUTPUT FORMAT (STRICT): Produce ONE LINE per sample. Format: ID | Score\n\n"
        for i, sample in enumerate(batch_samples):
            prompt += _sample_line(i, sample) + "\n"
    elif strategy == "zeroshot_cot":
        prompt = "Rate the probability of association (0.0 to 1.0) for the following miRNA-Disease pairs.\n"
        prompt += "OUTPUT FORMAT (STRICT): ONE LINE per sample. Format: ID | Score | Brief_Reason (max 15 words)\n\n"
        for i, sample in enumerate(batch_samples):
            prompt += _sample_line(i, sample) + "\n"
    elif strategy == "evidence_first":
        prompt = "Rate the probability of association (0.0 to 1.0).\n"
        prompt += "OUTPUT FORMAT: ONE LINE per sample. Format: ID | Brief_Reason (max 15 words) | Score\n\n"
        for i, sample in enumerate(batch_samples):
            prompt += _sample_line(i, sample) + "\n"
    elif strategy == "json_strict":
        prompt = "Rate the probability of association (0.0 to 1.0).\n"
        prompt += "OUTPUT FORMAT: YOU MUST OUTPUT STRICT JSON ARRAY of objects. Fields: id, score, reason.\n\n[\n"
        for i, sample in enumerate(batch_samples):
            prompt += f'  {{"id": {i}, "features": "{_sample_line(i, sample)}"}}\n'
        prompt += "]"
    elif strategy == "calibrated_cot":
        prompt = """Rate miRNA-disease association probability from structured evidence.
Calibration: rank<=20 -> 0.90-0.99; 21-100 -> 0.75-0.90; 101-300 -> 0.60-0.80 when similarity/path support exists; 301-800 -> 0.45-0.65; >800 with no support -> <0.30. Trust rank over tiny RWR probabilities. Similarity counts and paths raise confidence.
OUTPUT FORMAT (STRICT): ONE LINE per sample. Format: ID | Score | Brief_Reason (max 12 words)

"""
        for i, sample in enumerate(batch_samples):
            prompt += _sample_line(i, sample) + "\n"
    else: # baseline
        prompt = """You are an expert Biologist. Rate the probability of association for the following miRNA-Disease pairs.

=== OUTPUT FORMAT (STRICT) ===
Return a simple list, ONE LINE per sample. Do NOT use Markdown formatting (no bold **, no tables).
Format: ID | Score | Brief_Reason
- Score: float between 0.0 and 1.0
- Brief_Reason: COMPLETE SENTENCE, MAX 15 WORDS. Be extremely concise.

=== REFERENCE ANCHORS ===
Input: {anchor_pos}
Output: {anchor_pos_out}

Input: {anchor_neg}
Output: {anchor_neg_out}

=== SCORING RULES ===
1. **High (>0.75)**: Rank < 1000 (Top 50%) OR Direct Link=1 OR Sim > 0.6.
2. **Medium (0.5 - 0.7)**: Rank < 1500 but no other strong evidence.
3. **Low (<0.3)**: Rank > 2000 AND No Paths AND Low Sim.

=== SAMPLES TO RATE ===
""".format(anchor_pos=ANCHOR_POS["text"], anchor_pos_out=ANCHOR_POS["output"], anchor_neg=ANCHOR_NEG["text"], anchor_neg_out=ANCHOR_NEG["output"])
        for i, sample in enumerate(batch_samples):
            prompt += _main_sample_line(i, sample) + "\n"
        prompt += "\n\nYour Output:\n"
    return prompt

def parse_llm_response(response_text, strategy):
    results = {}
    if strategy == "json_strict":
        try:
            # find JSON array string
            idx_start = response_text.find('[')
            idx_end = response_text.rfind(']')
            if idx_start != -1 and idx_end != -1:
                arr = json.loads(response_text[idx_start:idx_end+1])
                for item in arr:
                    results[int(item['id'])] = {'score': _coerce_score(item['score']), 'reason': str(item.get('reason', ''))}
            return results
        except: pass  # Fallback to loose logic

    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or "ANCHOR" in line: continue
        line = line.strip('|').strip()
        line = re.sub(r'^\s*[-*]\s*', '', line)
        line = re.sub(r'^\s*ID\s*[:#]?\s*', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^\s*Sample\s+', '', line, flags=re.IGNORECASE)
        if re.match(r'^\s*id\s*[|:]', line, flags=re.IGNORECASE):
            continue
        
        if strategy == "zeroshot_direct":
            match = re.search(r'^(\d+)\s*[|:]\s*(\d*\.?\d+)', line)
            if match: results[int(match.group(1))] = {'score': _coerce_score(match.group(2)), 'reason': ''}
        elif strategy == "evidence_first":
            match = re.search(r'^(\d+)\s*[|:]\s*(.*?)\s*[|:]\s*(\d*\.?\d+)\s*$', line)
            if match: results[int(match.group(1))] = {'score': _coerce_score(match.group(3)), 'reason': match.group(2).strip()}
        else: # baseline and zeroshot_cot
            match = re.search(r'^(\d+)\s*[|:]\s*(\d*\.?\d+)\s*[|:]\s*(.*)', line)
            if match: results[int(match.group(1))] = {'score': _coerce_score(match.group(2)), 'reason': match.group(3).strip()}
    return results

def check_needs_review(score, reason, args):
    if args.uncertainty_low <= score <= args.uncertainty_high: 
        return True, f"Score in uncertain range"
    uncertain_keywords = ["conflicting evidence", "contradictory", "highly uncertain"]
    reason_lower = reason.lower()
    for kw in uncertain_keywords:
        if kw in reason_lower:
            return True, f"Uncertainty keyword found: '{kw}'"
    return False, ""


def structured_evidence_score(sample, args):
    scores = []
    weights = []

    if not args.disable_rwr and not args.llm_only:
        rank = sample.get('rwr_rank', 9999)
        if rank <= 20:
            rwr_score = 0.98
        elif rank <= 100:
            rwr_score = 0.88
        elif rank <= 300:
            rwr_score = 0.72
        elif rank <= 800:
            rwr_score = 0.55
        elif rank <= args.rwr_hard_filter_rank:
            rwr_score = 0.28
        else:
            rwr_score = 0.08
        scores.append(rwr_score)
        weights.append(0.65)

    if not args.disable_sim and not args.llm_only:
        support_count = sample.get('sim_mirna_assoc_count', 0) + sample.get('sim_disease_assoc_count', 0)
        sim_strength = max(sample.get('sim_mirna_max', 0.0), sample.get('sim_disease_max', 0.0))
        sim_score = 0.10 + min(0.70, support_count * 0.055)
        if sim_strength >= 0.70:
            sim_score += 0.10
        scores.append(max(0.0, min(0.95, sim_score)))
        weights.append(0.20)

    if not args.disable_graph and not args.llm_only:
        if sample.get('direct_link', 0) > 0:
            graph_score = 0.90
        elif sample.get('paths', 0) > 0:
            graph_score = min(0.85, 0.55 + 0.05 * min(sample.get('paths', 0), 5))
        else:
            graph_score = 0.15
        scores.append(graph_score)
        weights.append(0.15)

    if not scores:
        return 0.5

    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def compute_metrics(y_true, y_pred, y_scores):
    try:
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
        return (
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            roc_auc_score(y_true, y_scores),
            average_precision_score(y_true, y_scores),
        )
    except ImportError:
        n = len(y_true)
        acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n if n else 0.0
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        pos_scores = [s for t, s in zip(y_true, y_scores) if t == 1]
        neg_scores = [s for t, s in zip(y_true, y_scores) if t == 0]
        if not pos_scores or not neg_scores:
            auc = 0.0
            aupr = 0.0
        else:
            wins = 0.0
            for ps in pos_scores:
                for ns in neg_scores:
                    if ps > ns:
                        wins += 1.0
                    elif ps == ns:
                        wins += 0.5
            auc = wins / (len(pos_scores) * len(neg_scores))

            sorted_pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
            tp = 0
            precisions = []
            for rank, (_, label) in enumerate(sorted_pairs, start=1):
                if label == 1:
                    tp += 1
                    precisions.append(tp / rank)
            aupr = sum(precisions) / len(pos_scores) if pos_scores else 0.0
        return acc, f1, auc, aupr


def install_json_split_train_matrix(data_loader, current_input_file):
    if isinstance(data_loader.train_data, dict) and data_loader.train_data:
        return

    mask_file = current_input_file

    with open(mask_file, 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    train_md_matrix = data_loader.md_matrix.copy()
    masked = 0
    for item in split_data:
        if item.get('label') != 1:
            continue
        m_idx = data_loader.get_miRNA_index(item['mirna'])
        d_idx = data_loader.get_disease_index(item['disease'])
        if m_idx is not None and d_idx is not None and train_md_matrix[m_idx, d_idx] != 0:
            train_md_matrix[m_idx, d_idx] = 0
            masked += 1

    data_loader._train_md_matrix = train_md_matrix
    print(f"Using JSON split fallback train matrix from {mask_file}; masked {masked} positive test edges.")

async def process_single_pair_async(item, app, sim_agent, graph_agent, semaphore, args):
    async with semaphore:
        try:
            m_name = item['mirna']
            d_name = item['disease']
            
            loop = asyncio.get_running_loop()
            def get_features():
                m_idx = sim_agent.data_loader.get_miRNA_index(m_name)
                d_idx = sim_agent.data_loader.get_disease_index(d_name)
                return sim_agent.compute_evidence(m_idx, d_idx), graph_agent.compute_evidence(m_idx, d_idx)
            
            sim_res, graph_res = await loop.run_in_executor(None, get_features)
            
            # Apply feature ablations for review stage too
            sf = sim_res.get("features", {}) if not args.disable_sim else {}
            gf = graph_res.get("features", {}) if not args.disable_graph else {}
            item_rank = item.get('rwr_rank', 9999) if not args.disable_rwr else 9999
            item_prob = item.get('rwr_prob', 0.0) if not args.disable_rwr else 0.0

            rwr_feat = {"rank": item_rank, "score": item_prob, "target_disease": d_name}
            
            feature_data = {"rwr": rwr_feat, "similarity": sf, "graph": gf}
            
            if args.llm_only:
                feature_data = {"rwr": {}, "similarity": {}, "graph": {}} # Wipe all
            
            initial_state = {
                "mirna_id": m_name, "disease_id": d_name, "feature_data": feature_data,
                "draft_reasoning": None, "critique_feedback": None, "critique_passed": False, "revision_count": 0
            }
            
            final_state = await app.ainvoke(initial_state, config={"configurable": {"thread_id": f"{m_name}_{d_name}"}})
            reasoning = final_state.get("draft_reasoning", "")
            
            score = 0.5
            try:
                match = re.search(r'\{\s*"score"\s*:\s*(\d+\.?\d*)\s*\}', reasoning)
                if match: score = float(match.group(1))
                else:
                    match_loose = re.search(r'score.*?(\d+\.?\d*)', reasoning, re.IGNORECASE)
                    if match_loose: score = float(match_loose.group(1))
            except: pass
            
            item['predicted_score'] = score
            item['reasoning'] = f"[Review] {reasoning}"
            item['method'] = "full_agent_review_async"
            return item
        except Exception as e:
            item['predicted_score'] = 0.5
            item['method'] = "failed"
            return item

async def run_review_phase(review_queue, app, sim_agent, graph_agent, args):
    semaphore = asyncio.Semaphore(15) # Concurrency
    tasks = [process_single_pair_async(item, app, sim_agent, graph_agent, semaphore, args) for item in review_queue]
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async Reviewing"):
        results.append(await f)
    return results

async def main(args):
    start_time = time.time()
    INPUT_FILE = os.path.abspath(args.input_file)
    OUTPUT_FILE = f"ablations/results_gpt/res_{args.exp_name}.json"
    
    print(f"=== Starting Ablation: {args.exp_name} ===")
    print(f"Using test set: {INPUT_FILE}")
    
    data_loader = DataLoader(dataset_dir="./dataset")
    data_loader.load_all()
    with open(INPUT_FILE, 'r', encoding='utf-8') as f: raw_data = json.load(f)
    if args.limit_samples > 0:
        raw_data = raw_data[:args.limit_samples]
        print(f"Debug limit enabled: running first {len(raw_data)} samples only.")
    install_json_split_train_matrix(data_loader, INPUT_FILE)
    
    extractor = OptimizedFeatureExtractor(data_loader, test_pairs=raw_data)
    sim_agent = SimilarityAgent(data_loader)
    graph_agent = GraphAgent(data_loader)
    
    grouped_data = defaultdict(list)
    for item in raw_data: grouped_data[item['mirna']].append(item)
        
    extracted_features = []
    for m_name, items in tqdm(grouped_data.items(), desc="Phase 1: Feature Extraction"):
        m_idx = data_loader.get_miRNA_index(m_name)
        if m_idx is None:
            for item in items: extracted_features.append(item)
            continue
            
        disease_probs = extractor.run_rwr_for_mirna(m_idx, restart_prob=args.rwr_restart_prob)
        
        for item in items:
            d_name = item['disease']
            d_idx = data_loader.get_disease_index(d_name)
            if d_idx is None:
                extracted_features.append(item)
                continue
                
            sim_res = sim_agent.compute_evidence(m_idx, d_idx)
            graph_res = graph_agent.compute_evidence(m_idx, d_idx)
            
            sim_feats = sim_res.get('features', {})
            graph_feats = graph_res.get('features', {})
            
            # --- Apply Feature Ablations ---
            r_rank = int(np.sum(disease_probs > disease_probs[d_idx])) + 1 if not args.disable_rwr and not args.llm_only else 9999
            r_prob = float(disease_probs[d_idx]) if not args.disable_rwr and not args.llm_only else 0.0
            
            s_m_max = sim_feats.get('max_mirna_sim', sim_feats.get('top_mirna_sim', 0.0)) if not args.disable_sim and not args.llm_only else 0.0
            s_d_max = sim_feats.get('max_disease_sim', sim_feats.get('top_disease_sim', 0.0)) if not args.disable_sim and not args.llm_only else 0.0
            s_m_count = sim_feats.get('mirna_assoc_count', 0) if not args.disable_sim and not args.llm_only else 0
            s_d_count = sim_feats.get('disease_assoc_count', 0) if not args.disable_sim and not args.llm_only else 0
            
            g_link = graph_feats.get('direct_link', 0) if not args.disable_graph and not args.llm_only else 0
            g_paths = graph_feats.get('num_paths', len(graph_feats.get('paths', []))) if not args.disable_graph and not args.llm_only else 0
            g_max_strength = graph_feats.get('max_strength', 0.0) if not args.disable_graph and not args.llm_only else 0.0
            g_mean_strength = graph_feats.get('mean_strength', 0.0) if not args.disable_graph and not args.llm_only else 0.0
            g_mirna_degree = graph_feats.get('mirna_degree', 0) if not args.disable_graph and not args.llm_only else 0
            g_disease_degree = graph_feats.get('disease_degree', 0) if not args.disable_graph and not args.llm_only else 0
            
            item.update({
                "rwr_rank": r_rank, "rwr_prob": r_prob,
                "sim_mirna_max": s_m_max, "sim_disease_max": s_d_max,
                "sim_mirna_assoc_count": int(s_m_count),
                "sim_disease_assoc_count": int(s_d_count),
                "direct_link": int(g_link), "paths": int(g_paths),
                "path_max_strength": float(g_max_strength),
                "path_mean_strength": float(g_mean_strength),
                "mirna_degree": int(g_mirna_degree),
                "disease_degree": int(g_disease_degree)
            })
            extracted_features.append(item)

    if not args.disable_llm and get_llm is None:
        raise ImportError("LLM dependencies are not installed. Install langchain_openai/langgraph or run with --disable_llm.")

    if not args.disable_llm and not args.disable_review and create_workflow is None:
        raise ImportError("Review workflow dependencies are not installed. Install langgraph or disable review-only modes.")

    llm = None if args.disable_llm else get_llm(temperature=0.1)
    final_processed_results = []
    hard_filtered, auto_accepted, to_predict = [], [], []
    review_queue = []
    
    for feat in extracted_features:
        if args.force_all_review:
            review_queue.append(feat)
            continue
            
        if not args.disable_rules:
            rwr_available = not args.disable_rwr and not args.llm_only
            sim_available = not args.disable_sim and not args.llm_only
            graph_available = not args.disable_graph and not args.llm_only

            weak_checks = []
            if rwr_available:
                weak_checks.append(feat['rwr_rank'] > args.rwr_hard_filter_rank)
            if graph_available:
                weak_checks.append(feat['paths'] == 0)
            if sim_available:
                weak_checks.append(feat['sim_mirna_max'] < 0.4 and feat['sim_disease_max'] < 0.4)

            strong_accept = False
            if graph_available and feat['direct_link'] == 1:
                strong_accept = True
            if rwr_available and feat['rwr_rank'] <= 20:
                strong_accept = True
            if sim_available and graph_available and feat['sim_mirna_max'] > 0.95 and feat['paths'] > 0:
                strong_accept = True

            if weak_checks and all(weak_checks):
                feat['predicted_score'] = 0.01; feat['method'] = "hard_filter"
                hard_filtered.append(feat)
            elif strong_accept:
                 feat['predicted_score'] = 0.99; feat['method'] = "auto_accept"
                 auto_accepted.append(feat)
            else: to_predict.append(feat)
        else:
            to_predict.append(feat)
            
    final_processed_results.extend(hard_filtered)
    final_processed_results.extend(auto_accepted)
    
    async def process_batch_task(batch_items, batch_sem):
        async with batch_sem:
            try:
                loop_in = asyncio.get_running_loop()
                p = construct_batch_prompt(batch_items, args.prompt_strategy)
                resp_content = await loop_in.run_in_executor(None, lambda: llm.invoke(p).content)
                results_map = parse_llm_response(resp_content, args.prompt_strategy)
                if not results_map:
                    print(f"Warning: parsed 0 items from batch response preview: {resp_content[:300]!r}")
                
                l_res, l_rev = [], []
                for lid, sample in enumerate(batch_items):
                    if lid in results_map:
                        score = results_map[lid]['score']
                        reason = results_map[lid]['reason']
                        needs_review, _ = check_needs_review(score, reason, args)
                        
                        if args.disable_review: needs_review = False  # OVERRIDE
                        
                        if needs_review:
                            sample['batch_score'] = score
                            l_rev.append(sample)
                        else:
                            sample['predicted_score'] = score
                            sample['method'] = "fast_cot_batch"
                            l_res.append(sample)
                    else:
                        if args.disable_review:
                            sample['predicted_score'] = 0.5  # Forced guess
                            sample['method'] = "fast_cot_batch_failed"
                            l_res.append(sample)
                        else:
                            l_rev.append(sample)
                return l_res, l_rev
            except Exception as e:
                print(f"Warning: batch inference failed: {e}")
                if args.disable_review:
                    for s in batch_items: s['predicted_score'] = 0.5; s['method'] = "fast_cot_batch_failed"
                    return batch_items, []
                return [], batch_items

    app = create_workflow() if (not args.disable_review or args.force_all_review) and not args.disable_llm else None

    if to_predict and args.disable_llm:
        for sample in to_predict:
            sample['predicted_score'] = structured_evidence_score(sample, args)
            sample['method'] = "structured_score_no_llm"
        final_processed_results.extend(to_predict)
    elif to_predict and args.disable_fast_cot:
        review_queue.extend(to_predict)
    elif to_predict and not args.force_all_review:
        BATCH_SIZE = args.batch_size
        num_batches = math.ceil(len(to_predict) / BATCH_SIZE)
        batches = [to_predict[i*BATCH_SIZE : (i+1)*BATCH_SIZE] for i in range(num_batches)]
        batch_sem = asyncio.Semaphore(args.parallel_requests) 
        tasks_batch = [process_batch_task(b, batch_sem) for b in batches]
        
        batch_results = []
        for f in tqdm(asyncio.as_completed(tasks_batch), total=len(batches), desc="Batches"):
            res, rev = await f
            batch_results.extend(res)
            review_queue.extend(rev)
        final_processed_results.extend(batch_results)
    
    if review_queue and not args.disable_review and not args.disable_llm:
        reviewed_results = await run_review_phase(review_queue, app, sim_agent, graph_agent, args)
        final_processed_results.extend(reviewed_results)
        
    # Evaluate
    y_true, y_pred, y_scores = [], [], []
    for s in final_processed_results:
        y_true.append(s['label'])
        s_val = s.get('predicted_score', 0.5)
        y_scores.append(s_val)
        y_pred.append(1 if s_val >= 0.5 else 0)
        
    try:
        acc, f1, auc, aupr = compute_metrics(y_true, y_pred, y_scores)
    except: acc, f1, auc, aupr = 0, 0, 0, 0
    
    total_time = time.time() - start_time
    
    metrics = {
        "exp_name": args.exp_name,
        "input_file": INPUT_FILE,
        "threshold": 0.5,
        "accuracy": acc, "f1": f1, "auc": auc, "aupr": aupr,
        "time_seconds": total_time,
        "review_rate": len(review_queue) / max(1, len(raw_data)),
        "samples_hard_filtered": len(hard_filtered),
        "samples_auto_accepted": len(auto_accepted),
        "samples_reviewed": len(review_queue)
    }
    print(f"\nResults for {args.exp_name}: ACC={acc:.4f}, AUC={auc:.4f}, AUPR={aupr:.4f}, Time={total_time:.1f}s")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "results": final_processed_results}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--disable_rwr", action="store_true")
    parser.add_argument("--disable_sim", action="store_true")
    parser.add_argument("--disable_graph", action="store_true")
    parser.add_argument("--llm_only", action="store_true")
    parser.add_argument("--disable_llm", action="store_true")
    parser.add_argument("--disable_rules", action="store_true")
    parser.add_argument("--disable_fast_cot", action="store_true")
    parser.add_argument("--disable_review", action="store_true")
    parser.add_argument("--force_all_review", action="store_true")
    parser.add_argument("--rwr_restart_prob", type=float, default=0.3)
    parser.add_argument("--rwr_hard_filter_rank", type=int, default=1800)
    parser.add_argument("--uncertainty_low", type=float, default=0.5)
    parser.add_argument("--uncertainty_high", type=float, default=0.65)
    parser.add_argument("--prompt_strategy", type=str, default="baseline", choices=["baseline", "zeroshot_direct", "zeroshot_cot", "evidence_first", "json_strict", "calibrated_cot"])
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--parallel_requests", type=int, default=5)
    parser.add_argument("--limit_samples", type=int, default=0)
    
    args = parser.parse_args()
    asyncio.run(main(args))
