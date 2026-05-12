import sys
import os
import json
import time
import math
import re
import asyncio
import numpy as np
import scipy.sparse as sp
import torch
from tqdm.asyncio import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Setup paths
sys.path.append(os.getcwd())
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from data_loader.loader import DataLoader
from agents.rwr_agent import RWRAgent
from agents.similarity_agent import SimilarityAgent
from agents.graph_agent import GraphAgent
from agents.config import get_llm
from agents.workflow import create_workflow
# from batch_predict import process_single_pair # Implemented async version locally

# --- Configuration ---
BATCH_SIZE = 25  # Reduced batch size for stability with reasoning
CONCURRENCY_LIMIT = 20 # Increased for speed
DATASET_DIR = r"dataset"
INPUT_FILE = r"dataset/test_set_boosted.json"
OUTPUT_FILE = r"results/hybrid_assessment_results.json"
TOP_K = 50

# --- Anchors ---
ANCHOR_POS = {
    "text": "ID: ANCHOR_POS | hsa-mir-21 - Lung Neoplasms | Rank: 5 | SimM: 0.95 | SimD: 0.88 | Link: 1 | Paths: 4",
    "output": "ANCHOR_POS | 0.99 | Textbook positive case: Top-tier RWR rank, strong functional clustering, and direct database evidence."
}
ANCHOR_NEG = {
    "text": "ID: ANCHOR_NEG | hsa-mir-dummy - Random Disease | Rank: 2500 | SimM: 0.10 | SimD: 0.05 | Link: 0 | Paths: 0",
    "output": "ANCHOR_NEG | 0.01 | Textbook negative case: No topological or functional evidence found."
}

class OptimizedFeatureExtractor:
    def __init__(self, data_loader, test_pairs=None):
        self.data_loader = data_loader
        self.test_pairs = test_pairs or []
        self.transition_matrix = None
        
        # GPU Setup
        self.transition_matrix_gpu = None
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        self.total_nodes = 0
        self.n_mirna = 0
        self.n_disease = 0
        self.n_lncrna = 0
        self._build_rwr_matrix()
        
    def _build_rwr_matrix(self):
        print("Building RWR Transition Matrix (Sparse)...")
        if self.use_gpu:
            print(f"CUDA/GPU Detected: {torch.cuda.get_device_name(0)}. RWR will proceed on GPU for extreme speed.")
        else:
            print("CUDA Not Found. Using CPU for RWR (slower).")
            
        train_md_matrix = self.data_loader.get_train_matrix()
        
        # --- ANTI-LEAKAGE SAFETY ---
        # Explicitly mask out test edges from the matrix to ensure no leakage,
        # even if loader fallback was triggered.
        if self.test_pairs:
            print(f"Applying Anti-Leakage Masking for {len(self.test_pairs)} test pairs...")
            masked_count = 0
            # Ensure it is mutable (lil or dense) - get_train_matrix returns numpy or dense usually
            # But let's work on the sparse construction phase below.
            
            # We will perform masking AFTER constructing the full adjacency matrix chunks
            # but BEFORE creating the final sparse matrix.
            # Actually, train_md_matrix is numpy array usually.
            if isinstance(train_md_matrix, np.ndarray):
                train_md_matrix = train_md_matrix.copy() # Safe copy
                for item in self.test_pairs:
                    # Only mask Positive samples (label=1) in the test set.
                    # Negatives (label=0) are not edges in the graph anyway.
                    if item.get('label') == 1:
                        m_idx = self.data_loader.get_miRNA_index(item['mirna'])
                        d_idx = self.data_loader.get_disease_index(item['disease'])
                        if m_idx is not None and d_idx is not None:
                            if train_md_matrix[m_idx, d_idx] != 0:
                                train_md_matrix[m_idx, d_idx] = 0
                                masked_count += 1
                if masked_count > 0:
                    print(f"Safe-guard: Masked {masked_count} potential leakage edges from RWR graph.")
            else:
                 print("Warning: train_md_matrix is not numpy array, skipping safety mask (assuming loader did it right).")
        # ---------------------------

        self.n_mirna = train_md_matrix.shape[0]
        self.n_disease = train_md_matrix.shape[1]
        self.n_lncrna = self.data_loader.ml_matrix.shape[1]
        self.total_nodes = self.n_mirna + self.n_disease + self.n_lncrna
        
        # Use LIL for efficient construction
        adj_matrix = sp.lil_matrix((self.total_nodes, self.total_nodes))
        
        # Helper to ensure numpy
        def to_n(x): return x if isinstance(x, np.ndarray) else x.toarray()
        
        md = to_n(train_md_matrix)
        ml = to_n(self.data_loader.ml_matrix)
        dl = to_n(self.data_loader.dl_matrix)
        
        # miRNA-disease (Train only)
        adj_matrix[:self.n_mirna, self.n_mirna:self.n_mirna+self.n_disease] = md
        adj_matrix[self.n_mirna:self.n_mirna+self.n_disease, :self.n_mirna] = md.T
        
        # miRNA-lncRNA
        adj_matrix[:self.n_mirna, self.n_mirna+self.n_disease:] = ml
        adj_matrix[self.n_mirna+self.n_disease:, :self.n_mirna] = ml.T
        
        # disease-lncRNA
        adj_matrix[self.n_mirna:self.n_mirna+self.n_disease, self.n_mirna+self.n_disease:] = dl
        adj_matrix[self.n_mirna+self.n_disease:, self.n_mirna:self.n_mirna+self.n_disease] = dl.T
        
        # Convert to CSR for calculation and Normalize
        adj_matrix = adj_matrix.tocsr()
        row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        d_inv = sp.diags(1.0 / row_sums)
        norm_matrix = d_inv.dot(adj_matrix)
        
        # Store Transpose for RWR (W^T)
        self.transition_matrix = norm_matrix.T
        
        # --- GPU CONVERSION ---
        if self.use_gpu:
            print("Moving Transition Matrix to GPU...")
            try:
                # Convert Scipy CSR -> Scipy COO -> Torch Sparse
                coo = self.transition_matrix.tocoo()
                values = coo.data
                indices = np.vstack((coo.row, coo.col))
                
                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = coo.shape
                
                # Create sparse tensor on CPU then move to GPU to save GPU Memory during creation
                self.transition_matrix_gpu = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(self.device).coalesce()
                
                print("RWR Matrix moved to GPU successfully.")
            except Exception as e:
                print(f"GPU Conversion Failed ({e}). Falling back to CPU.")
                self.use_gpu = False
        else:
            print("RWR Matrix Built (CPU-Only).")

    def run_rwr_for_mirna(self, mirna_idx, restart_prob=0.3, max_iter=100):
        if self.use_gpu:
            return self._run_rwr_gpu(mirna_idx, restart_prob, max_iter)
        else:
            return self._run_rwr_cpu(mirna_idx, restart_prob, max_iter)

    def _run_rwr_cpu(self, mirna_idx, restart_prob, max_iter):
        # RWR from seed
        p = np.zeros(self.total_nodes)
        p[mirna_idx] = 1.0
        p0 = p.copy()
        
        for _ in range(max_iter):
            # p_new = (1 - restart_prob) * W^T * p + restart_prob * p0
            p_new = (1 - restart_prob) * (self.transition_matrix.dot(p)) + restart_prob * p0
            
            # Check convergence
            if _ % 10 == 0 and np.linalg.norm(p_new - p) < 1e-6:
                p = p_new
                break
            p = p_new
            
        disease_start = self.n_mirna
        disease_end = self.n_mirna + self.n_disease
        return p[disease_start:disease_end]

    def _run_rwr_gpu(self, mirna_idx, restart_prob, max_iter):
        # 1. Init p vector on GPU
        p = torch.zeros((self.total_nodes, 1), device=self.device)
        p[mirna_idx, 0] = 1.0
        p0 = p.clone()
        
        # 2. Iterate
        for _ in range(max_iter):
             # sparse.mm is Sparse x Dense -> Dense
             mv = torch.sparse.mm(self.transition_matrix_gpu, p)
             p_new = (1 - restart_prob) * mv + restart_prob * p0
             
             # Check convergence every 10 steps
             if _ % 10 == 0:
                 dist = torch.norm(p_new - p)
                 if dist < 1e-6:
                     p = p_new
                     break
             p = p_new
        
        # 3. Extract and Move to CPU
        disease_start = self.n_mirna
        disease_end = self.n_mirna + self.n_disease
        
        return p[disease_start:disease_end].flatten().cpu().numpy()

def construct_batch_prompt(batch_samples):
    prompt = """You are an expert Biologist. Rate the probability of association for the following miRNA-Disease pairs.

=== OUTPUT FORMAT (STRICT) ===
Return a simple list, ONE LINE per sample. Do NOT use Markdown formatting (no bold **, no tables).
Format: ID | Score | Brief_Reason
- Score: float between 0.0 and 1.0
- Brief_Reason: COMPLETE SENTENCE, MAX 15 WORDS. Be extremely concise.

=== REFERENCE ANCHORS ===
Input: {anchor_pos_text}
Output: {anchor_pos_out}

Input: {anchor_neg_text}
Output: {anchor_neg_out}

=== SCORING RULES ===
1. **High (>0.75)**: Rank < 1000 (Top 50%) OR Direct Link=1 OR Sim > 0.6.
2. **Medium (0.5 - 0.7)**: Rank < 1500 but no other strong evidence.
3. **Low (<0.3)**: Rank > 2000 AND No Paths AND Low Sim.

=== SAMPLES TO RATE ===
""".format(anchor_pos_text=ANCHOR_POS["text"], anchor_pos_out=ANCHOR_POS["output"],
           anchor_neg_text=ANCHOR_NEG["text"], anchor_neg_out=ANCHOR_NEG["output"])

    for i, sample in enumerate(batch_samples):
        # Use simple ID for LLM prompt (index within batch)
        line = (f"ID: {i} | {sample['mirna']} - {sample['disease']} | "
                f"Rank: {sample['rwr_rank']} | SimM: {sample['sim_mirna_max']:.2f} | "
                f"SimD: {sample['sim_disease_max']:.2f} | Link: {sample['direct_link']} | "
                f"Paths: {sample['paths']}\n")
        prompt += line
        
    prompt += "\n\nYour Output:\n"
    return prompt

def parse_llm_response_lines(response_text, batch_size):
    """
    Parses line-based output: ID | Score | Reason
    Returns dict: {id: {'score': float, 'reason': str}}
    """
    results = {}
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Skip anchor repetitions if LLM hallucinates them
        if "ANCHOR" in line: continue
        
        # Regex to capture "ID | Score | Reason"
        # Allow loose spacing and separators like | or :
        # Structure: Int | Float | Text
        match = re.search(r'^(\d+)\s*[|:]\s*(\d+\.?\d*)\s*[|:]\s*(.*)', line)
        if match:
            local_id = int(match.group(1))
            score = float(match.group(2))
            reason = match.group(3).strip()
            results[local_id] = {'score': score, 'reason': reason}
            
    return results

def check_needs_review(score, reason):
    """
    Determines if a sample needs Full Agent Review.
    Trigger 1: Fuzzy Score (0.4 - 0.6)
    Trigger 2: Uncertainty keywords in reason (Strict Mode OFF)
    """
    # OPTIMIZATION: Narrow the uncertainty window.
    # If LLM gives 0.7 or 0.3, trust it.
    if 0.5 <= score <= 0.65: 
        return True, "Score in uncertain range (0.5-0.65)"
    
    # OPTIMIZATION: Reduce keywords that cause false positives
    uncertain_keywords = ["conflicting evidence", "contradictory", "highly uncertain"]
    reason_lower = reason.lower()
    for kw in uncertain_keywords:
        if kw in reason_lower:
            return True, f"Uncertainty keyword found: '{kw}'"
            
    return False, ""

async def process_single_pair_async(item, app, sim_agent, graph_agent, semaphore):
    """
    Async version of process_single_pair with optimizations:
    1. Reuses RWR rank/score from Phase 1 (Avoids re-running RWR).
    2. Runs Sim/Graph agents (fast) to get details.
    3. Uses app.ainvoke for parallel LLM calls.
    """
    async with semaphore:
        try:
            m_name = item['mirna']
            d_name = item['disease']
            
            # 1. Feature Construction (Hybrid)
            # Sim & Graph: these are fast lookups, so we run them to get full evidence details
            # (Reason: We need neighbor names for the LLM prompt)
            
            # Helper to run blocking code in thread
            loop = asyncio.get_running_loop()
            
            def get_features():
                m_idx = sim_agent.data_loader.get_miRNA_index(m_name)
                d_idx = sim_agent.data_loader.get_disease_index(d_name)
                s_res = sim_agent.compute_evidence(m_idx, d_idx)
                g_res = graph_agent.compute_evidence(m_idx, d_idx)
                return s_res, g_res

            sim_res, graph_res = await loop.run_in_executor(None, get_features)
            
            # RWR: Reuse Phase 1 results to save massive compute time
            rwr_feat = {
                "rank": item.get('rwr_rank', 9999),
                "score": item.get('rwr_prob', 0.0),
                "target_disease": d_name,
                "note": "Pre-computed using Optimized Feature Extractor"
            }
            
            feature_data = {
                "rwr": rwr_feat,
                "similarity": sim_res.get("features", {}),
                "graph": graph_res.get("features", {})
            }
            
            # 2. LLM Workflow
            initial_state = {
                "mirna_id": m_name,
                "disease_id": d_name,
                "feature_data": feature_data,
                "draft_reasoning": None,
                "critique_feedback": None,
                "critique_passed": False,
                "revision_count": 0
            }
            
            # COST SAVING: Enforce strict retry limit in config if supported, 
            # though workflow.py handles logic.
            # We can also add a 'cost_saving_mode' flag to state if we modify workflow, 
            # but for now rely on standard flow.
            
            config = {"configurable": {"thread_id": f"{m_name}_{d_name}"}}
            
            final_state = await app.ainvoke(initial_state, config=config)
            
            reasoning = final_state.get("draft_reasoning", "")
            
            # Extract score
            score = 0.5
            try:
                match = re.search(r'\{\s*"score"\s*:\s*(\d+\.?\d*)\s*\}', reasoning)
                if match:
                    score = float(match.group(1))
                else:
                    match_loose = re.search(r'score.*?(\d+\.?\d*)', reasoning, re.IGNORECASE)
                    if match_loose:
                        score = float(match_loose.group(1))
            except: pass
            
            item['predicted_score'] = score
            item['reasoning'] = f"[Full-Agent Review] {reasoning} (Trigger: {item.get('trigger_reason', '')})"
            item['method'] = "full_agent_review_async"
            item['iterations'] = final_state.get("revision_count", 0)
            return item
            
        except Exception as e:
            item['predicted_score'] = 0.5
            item['reasoning'] = f"Async Review Failed: {e}"
            item['method'] = "failed"
            return item

async def run_review_phase(review_queue, app, sim_agent, graph_agent):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    
    for item in review_queue:
        task = process_single_pair_async(item, app, sim_agent, graph_agent, semaphore)
        tasks.append(task)
        
    results = []
    # Use tqdm.as_completed for progress bar
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async Reviewing"):
        res = await f
        results.append(res)
        
    return results

async def main():
    start_time = time.time()
    print("="*60)
    print("Hybrid Batch Prediction Pipeline (Fast-CoT + Review)")
    print("="*60)
    
    # 1. Load Data
    data_loader = DataLoader(dataset_dir=DATASET_DIR)
    data_loader.load_all()
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} samples.")
    
    # 2. Extract Features (Grouped)
    extractor = OptimizedFeatureExtractor(data_loader, test_pairs=raw_data)
    sim_agent = SimilarityAgent(data_loader)
    graph_agent = GraphAgent(data_loader)
    
    grouped_data = defaultdict(list)
    for item in raw_data:
        grouped_data[item['mirna']].append(item)
        
    extracted_features = []
    print("Phase 1: Feature Extraction...")
    
    for m_name, items in tqdm(grouped_data.items()):
        m_idx = data_loader.get_miRNA_index(m_name)
        if m_idx is None:
            for item in items:
                item.update({"rwr_rank": 9999, "sim_mirna_max":0, "sim_disease_max":0, "direct_link":0, "paths":0})
                extracted_features.append(item)
            continue
            
        disease_probs = extractor.run_rwr_for_mirna(m_idx)
        
        for item in items:
            d_name = item['disease']
            d_idx = data_loader.get_disease_index(d_name)
            if d_idx is None:
                item.update({"rwr_rank": 9999, "sim_mirna_max":0, "sim_disease_max":0, "direct_link":0, "paths":0})
                extracted_features.append(item)
                continue
                
            prob = disease_probs[d_idx]
            rank = int(np.sum(disease_probs > prob)) + 1
            
            sim_res = sim_agent.compute_evidence(m_idx, d_idx)
            sim_feats = sim_res.get('features', {})
            graph_res = graph_agent.compute_evidence(m_idx, d_idx)
            graph_feats = graph_res.get('features', {})
            
            item.update({
                "rwr_rank": rank,
                "rwr_prob": float(prob),
                "sim_mirna_max": sim_feats.get('top_mirna_sim', 0.0),
                "sim_disease_max": sim_feats.get('top_disease_sim', 0.0),
                "direct_link": graph_feats.get('direct_link', 0),
                "paths": graph_feats.get('num_paths', 0)
            })
            extracted_features.append(item)
            
    # 3. Hybrid Batch Scoring
    print("\nPhase 2: Hybrid Batch Scoring with Review...")
    llm = get_llm(temperature=0.1)
    final_processed_results = []
    
    # Buckets
    hard_filtered = []
    auto_accepted = []
    to_predict = []
    
    for feat in extracted_features:
        # 1. Hard Filter (Negative) - STRICTER
        if (feat['rwr_rank'] > 1800 and feat['paths'] == 0 and 
            feat['sim_mirna_max'] < 0.4 and feat['sim_disease_max'] < 0.4):
            feat['predicted_score'] = 0.01
            feat['reasoning'] = "Hard Filtered (Rank > 1800)"
            feat['method'] = "hard_filter"
            hard_filtered.append(feat)
        # 2. Auto-Accept (Positive) - NEW OPTIMIZATION
        # If very high rank or known link, skip LLM to save cost
        elif (feat['direct_link'] == 1 or feat['rwr_rank'] <= 20 or 
              (feat['sim_mirna_max'] > 0.95 and feat['paths'] > 0)):
             feat['predicted_score'] = 0.99
             feat['reasoning'] = "Auto-Accepted (High Confidence: Link/Rank/Sim)"
             feat['method'] = "auto_accept"
             auto_accepted.append(feat)
        else:
            to_predict.append(feat)
            
    final_processed_results.extend(hard_filtered)
    final_processed_results.extend(auto_accepted)
    print(f"Stats: {len(hard_filtered)} Hard-Filtered, {len(auto_accepted)} Auto-Accepted.")
    print(f"Sending {len(to_predict)} samples to LLM Batching.")
    
    # Initialize Full Workflow for reviews
    app = create_workflow()
    
    # Batch Process (Async Wrapper)
    async def process_batch_task(batch_items, batch_sem):
        async with batch_sem:
            try:
                # Run sync LLM in thread to not block loop
                loop_in = asyncio.get_running_loop()
                p = construct_batch_prompt(batch_items)
                
                # Using invoke is blocking, so run in executor
                resp_content = await loop_in.run_in_executor(None, lambda: llm.invoke(p).content)
                
                results_map = parse_llm_response_lines(resp_content, len(batch_items))
                
                local_results = []
                local_review = []
                
                for lid, sample in enumerate(batch_items):
                    if lid in results_map:
                        score = results_map[lid]['score']
                        reason = results_map[lid]['reason']
                        needs_review, trigger_reason = check_needs_review(score, reason)
                        
                        if needs_review:
                            sample['trigger_reason'] = trigger_reason
                            sample['batch_score'] = score
                            sample['batch_reason'] = reason
                            local_review.append(sample)
                        else:
                            sample['predicted_score'] = score
                            sample['reasoning'] = f"[Fast-CoT] {reason}"
                            sample['method'] = "fast_cot_batch"
                            local_results.append(sample)
                    else:
                        sample['trigger_reason'] = "Batch parse failed"
                        local_review.append(sample)
                return local_results, local_review
            except Exception as e:
                print(f"Batch Failed: {e}")
                # All to review
                for s in batch_items:
                    s['trigger_reason'] = f"Batch Error: {e}"
                return [], batch_items

    # Main Batch Loop (Async)
    review_queue = []
    
    if to_predict:
        # Create batches
        num_batches = math.ceil(len(to_predict) / BATCH_SIZE)
        batches = [to_predict[i*BATCH_SIZE : (i+1)*BATCH_SIZE] for i in range(num_batches)]
        
        # Parallel Execution
        # Concurrency for LLM Batches (don't set too high to avoid Rate Limit)
        batch_sem = asyncio.Semaphore(5) 
        
        tasks_batch = [process_batch_task(b, batch_sem) for b in batches]
        
        # We need an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        print(f"Processing {num_batches} batches with concurrency=5...")
        # Use tqdm
        batch_results = []
        for f in tqdm(asyncio.as_completed(tasks_batch), total=len(batches), desc="Async Batches"):
            res, rev = await f
            batch_results.extend(res)
            review_queue.extend(rev)
            
        final_processed_results.extend(batch_results)
    
    # 4. Process Review Queue
    print(f"\nPhase 3: Reviewing {len(review_queue)} uncertain samples with Full Agents (Async)...")
    
    if review_queue:
        # Already inside an event loop, just await the coroutine
        reviewed_results = await run_review_phase(review_queue, app, sim_agent, graph_agent)
        final_processed_results.extend(reviewed_results)
    else:
        print("No samples needed review.")

            
    # 5. Top-K Selection (Global)
    print("\nPhase 4: Selecting Review Candidates for Report...")
    
    pos_samples = [x for x in final_processed_results if x.get('label')==1]
    neg_samples = [x for x in final_processed_results if x.get('label')==0]
    
    pos_samples.sort(key=lambda x: x.get('predicted_score', 0), reverse=True)
    neg_samples.sort(key=lambda x: x.get('predicted_score', 0)) # Ascending
    
    top_k_candidates = pos_samples[:TOP_K] + neg_samples[:TOP_K]
    
    final_report_list = []
    
    print("Generating/Ensuring Detailed Explanations for Top-K...")
    # Import locally to avoid circular imports if any
    # MOVED OUTSIDE of try/except block or rely on global imports
    # The error UnboundLocalError 'SimilarityAgent' suggests we are shadowing it in local scope 
    # but not initializing it, or the import inside try/except failed silently?
    # Actually, SimilarityAgent is imported at top level. 
    # If we re-import it here inside a try/except, Python might treat it as a local variable.
    # Let's REMOVE the local imports since they are already at the top of the file.
    
    import copy

    # Re-initialize agents for single processing if needed (or reuse existing)
    # We reuse app, sim_agent, graph_agent from main scope
    
    # Helper to run single pair sync-to-async bridge
    async def upgrade_explanation(itm):
        # Construct state
        m_name = itm['mirna']
        d_name = itm['disease']
        
        # We need to re-run feature extraction for this single pair to be sure?
        # Or just use what we have. 
        # Better to re-run single extraction to get FULL details (e.g. all paths) 
        # because batch extractor might have simplified things.
        
        # But for speed, let's use existing features if available, 
        # or minimal re-fetch.
        
        # Actually, let's just use the 'process_single_pair_async' logic 
        # but force it to run even if score is high.
        
        # We can reuse the process_single_pair_async function!
        # It does exactly what we want: Sim+Graph+LLM.
        # But we need to make sure it doesn't just return the old score.
        # It constructs state from scratch, so it WILL generate new reasoning.
        
        # Create a dummy semaphore
        sem = asyncio.Semaphore(1)
        
        # We process it again
        # Note: process_single_pair_async modifies item in-place
        res = await process_single_pair_async(itm, app, sim_agent, graph_agent, sem)
        return res

    # Identify which items need upgrade
    # 1. Any 'auto_accept' (template reasoning)
    # 2. Any 'hard_filter' (template reasoning) - though unlikely to be in Top-K Pos
    # 3. Any 'fast_cot_batch' (short reasoning)
    
    # Basically EVERYTHING in Top-K needs upgrade to be high-quality.
    # Because even Full-Review results from Phase 3 might have been 
    # stripped of reasoning if we implemented that optimization.
    
    items_to_upgrade = []
    for item in top_k_candidates:
        # Check if reasoning is "Simple/Template"
        r = item.get('reasoning', '')
        if "Auto-Accepted" in r or "Hard Filtered" in r or "[Fast-CoT]" in r:
           items_to_upgrade.append(item)
        # If it is [Full-Agent Review], we might keep it, or upgrade if we want consistency.
        # Let's upgrade EVERYTHING to ensure uniform high quality for the report.
        # EXCEPT if it was JUST processed in Phase 3 (saves time).
        elif "[Full-Agent Review]" in r:
             # It's already good.
             pass
        else:
             items_to_upgrade.append(item)
             
    if items_to_upgrade:
        print(f"Upgrading explanations for {len(items_to_upgrade)} Top-K samples (this may take a moment)...")
        # Run async
        tasks_upgrade = [upgrade_explanation(it) for it in items_to_upgrade]
        # Use existing loop
        try:
             loop_up = asyncio.get_running_loop()
        except RuntimeError:
             loop_up = asyncio.new_event_loop()
             asyncio.set_event_loop(loop_up)
             
        # Run concurrently
        # Limit concurrency to avoid rate limits
        sem_up = asyncio.Semaphore(10)
        
        async def run_with_sem(task, sem):
            async with sem:
                res = await task
                # Mark as upgraded so evaluation script can find it easily
                res['is_upgraded_explanation'] = True
                return res

        await asyncio.gather(*(run_with_sem(t, sem_up) for t in tasks_upgrade))
        
    # Sort lists again to make sure top_k_candidates are truly the top ones
    # Since we might have modified items in place during upgrade_explanation
    pos_samples.sort(key=lambda x: x.get('predicted_score', 0), reverse=True)
    neg_samples.sort(key=lambda x: x.get('predicted_score', 0))

    final_report_list = top_k_candidates

    # CRITICAL FIX: Ensure the upgraded samples are actually saved back to the full list!
    # top_k_candidates contains references to objects in final_processed_results?
    # Yes, Python lists store references. But let's be double sure.
    
    # Save statistics
    print(f"Top-K Explanations Upgraded. Saving all {len(final_processed_results)} results...")

    # Save metrics
    valid_scores = [s for s in final_processed_results if 'predicted_score' in s]
    tp = len([s for s in valid_scores if s['label']==1 and s['predicted_score']>=0.5])
    tn = len([s for s in valid_scores if s['label']==0 and s['predicted_score']<0.5])
    if len(valid_scores) > 0:
        acc = (tp+tn) / len(valid_scores)
    else:
        acc = 0.0
    
    print("\n" + "="*60)
    print(f"HYBRID PIPELINE COMPLETE")
    print(f"Total Samples: {len(final_processed_results)}")
    print(f"Review Rate: {len(review_queue)/len(to_predict)*100:.1f}% ({len(review_queue)} samples)")
    print(f"Estimated Accuracy: {acc:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Execution Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("="*60)

    # Save full results (for analysis)
    with open("results/hybrid_full_scores_gpt1.json", 'w', encoding='utf-8') as f:
        json.dump(final_processed_results, f, indent=2, ensure_ascii=False)

    # Save Deep Explanation Report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_report_list, f, indent=2, ensure_ascii=False)
        
    print(f"Full scores saved to results/hybrid_full_scores_gpt1.json")
    print(f"Top-K Explanations saved to {OUTPUT_FILE}")
    
    # Print Token Usage and Cost
    # from utils.token_counter import global_token_counter
    # model_name = os.getenv("MODEL_NAME", "xdf-gp-3.0")
    # global_token_counter.print_stats(model_name=model_name)

if __name__ == "__main__":
    asyncio.run(main())
