"""
Controller Agent
核心控制器，调度所有子 agent，整合结果，实现批处理单次LLM调用
重构版本：批处理推理，单次LLM调用
"""
from typing import Dict, List, Optional, Tuple
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.similarity_agent import SimilarityAgent
from agents.graph_agent import GraphAgent
from agents.rwr_agent import RWRAgent
from agents.web_agent import WebAgent
from agents.fusion_agent import FusionAgent
from agents.validation_agent import ValidationAgent
from llm_client import llm_client


class ControllerAgent:
    """控制器 Agent - 批处理推理"""
    
    def __init__(self, data_loader, max_refinement_iterations: int = 2, fast_mode: bool = False, ultra_fast: bool = False):
        """
        初始化 Controller Agent
        
        Args:
            data_loader: DataLoader 实例
            max_refinement_iterations: 最大重新推理迭代次数（批处理模式下不使用）
            fast_mode: 快速模式，跳过RWR Agent
            ultra_fast: 超快模式，只使用Graph Agent，跳过LLM调用
        """
        self.data_loader = data_loader
        self.max_refinement_iterations = max_refinement_iterations
        self.fast_mode = fast_mode
        self.ultra_fast = ultra_fast
        self.batch_size = 10  # 批处理大小
        
        # 初始化所有子 Agent
        self.similarity_agent = SimilarityAgent(data_loader)
        self.graph_agent = GraphAgent(data_loader)
        self.rwr_agent = RWRAgent(data_loader)
        self.web_agent = WebAgent(data_loader)
        self.fusion_agent = FusionAgent(data_loader)
        self.validation_agent = ValidationAgent(data_loader)
    
    def _generate_detailed_explanation(
        self, 
        final_score: float,
        sim_score: float,
        graph_score: float,
        rwr_score: float,
        web_score: float,
        graph_features: Dict,
        rwr_features: Dict,
        sim_features: Dict,
        web_features: Dict
    ) -> str:
        """
        生成详细、一致的解释文本
        
        Args:
            final_score: 最终分数
            sim_score: 相似度分数
            graph_score: 图分数
            rwr_score: RWR分数
            web_score: Web分数
            graph_features: 图特征
            rwr_features: RWR特征
            sim_features: 相似度特征
            web_features: Web特征
        
        Returns:
            详细的解释文本
        """
        explanation_parts = []
        
        # 提取特征值
        direct_link = graph_features.get("direct_link", 0)
        num_paths = graph_features.get("num_paths", 0)
        rwr_prob = rwr_features.get("rwr_probability", 0.0)
        rwr_rank = rwr_features.get("rank", 9999)
        total_diseases = rwr_features.get("total_diseases", 2077)
        mirna_assoc = sim_features.get("mirna_assoc_count", 0)
        disease_assoc = sim_features.get("disease_assoc_count", 0)
        evidence_level = web_features.get("evidence_level", "unknown")
        
        # 1. 图证据描述（与graph_score一致，大幅增加多样性）
        import random
        if direct_link > 0:
            variants = [
                "Direct link exists in the network",
                "A direct connection is present in the network",
                "Direct association found in the network structure",
                "Network topology shows a direct link",
                "A direct pathway connects these entities",
                "Direct network connection identified",
                "The network contains a direct link between these entities"
            ]
            explanation_parts.append(random.choice(variants))
        elif num_paths > 10:
            variants = [
                f"Multiple indirect paths found ({num_paths} paths)",
                f"Network analysis reveals {num_paths} indirect connections",
                f"Found {num_paths} indirect pathways in the network",
                f"Extensive network connectivity: {num_paths} indirect paths detected",
                f"Network structure shows {num_paths} connecting pathways",
                f"Multiple network routes identified ({num_paths} paths)",
                f"Rich network connectivity with {num_paths} indirect paths"
            ]
            explanation_parts.append(random.choice(variants))
        elif num_paths > 5:
            variants = [
                f"Several indirect paths found ({num_paths} paths)",
                f"Network shows {num_paths} indirect connections",
                f"Identified {num_paths} indirect pathways",
                f"Moderate network connectivity: {num_paths} paths present",
                f"Network analysis indicates {num_paths} connecting paths",
                f"Several network routes detected ({num_paths} paths)",
                f"Network structure contains {num_paths} indirect pathways"
            ]
            explanation_parts.append(random.choice(variants))
        elif num_paths > 0:
            variants = [
                f"Few indirect paths found ({num_paths} paths)",
                f"Limited network connections detected ({num_paths} paths)",
                f"Found {num_paths} indirect path(s) in the network",
                f"Sparse network connectivity: {num_paths} path(s) identified",
                f"Network shows minimal connectivity with {num_paths} path(s)",
                f"Limited network routes: {num_paths} path(s) detected",
                f"Few connecting pathways in the network ({num_paths} paths)"
            ]
            explanation_parts.append(random.choice(variants))
        else:
            variants = [
                "No network paths found",
                "No connections detected in the network",
                "Network analysis shows no connecting paths",
                "Network topology reveals no connecting pathways",
                "No network routes identified between these entities",
                "Network structure shows no connectivity",
                "Absence of network paths detected"
            ]
            explanation_parts.append(random.choice(variants))
        
        # 2. RWR证据描述（与rwr_score一致，大幅增加多样性）
        if rwr_prob > 1e-4:
            rank_pct = (rwr_rank / total_diseases * 100) if total_diseases > 0 else 100
            if rank_pct < 5:
                variants = [
                    f"High RWR probability ({rwr_prob:.2e}, top {rank_pct:.1f}%)",
                    f"Random walk analysis indicates high probability ({rwr_prob:.2e}, ranking in top {rank_pct:.1f}%)",
                    f"Strong RWR signal detected ({rwr_prob:.2e}, top {rank_pct:.1f}% rank)",
                    f"Elevated random walk probability ({rwr_prob:.2e}, top {rank_pct:.1f}% ranking)",
                    f"High network proximity signal ({rwr_prob:.2e}, top {rank_pct:.1f}%)",
                    f"Strong random walk evidence ({rwr_prob:.2e}, ranking in top {rank_pct:.1f}%)",
                    f"Prominent RWR signal ({rwr_prob:.2e}, top {rank_pct:.1f}% rank)"
                ]
                explanation_parts.append(random.choice(variants))
            elif rank_pct < 20:
                variants = [
                    f"Moderate RWR probability ({rwr_prob:.2e}, top {rank_pct:.1f}%)",
                    f"Random walk shows moderate probability ({rwr_prob:.2e}, top {rank_pct:.1f}%)",
                    f"Moderate RWR signal ({rwr_prob:.2e}, ranking in top {rank_pct:.1f}%)",
                    f"Moderate random walk probability ({rwr_prob:.2e}, top {rank_pct:.1f}% ranking)",
                    f"Decent network proximity ({rwr_prob:.2e}, top {rank_pct:.1f}%)",
                    f"Moderate random walk evidence ({rwr_prob:.2e}, top {rank_pct:.1f}%)",
                    f"Fair RWR signal strength ({rwr_prob:.2e}, top {rank_pct:.1f}%)"
                ]
                explanation_parts.append(random.choice(variants))
            else:
                variants = [
                    f"Low RWR probability ({rwr_prob:.2e}, rank {rwr_rank})",
                    f"Random walk indicates low probability ({rwr_prob:.2e}, rank {rwr_rank})",
                    f"Weak RWR signal ({rwr_prob:.2e}, rank {rwr_rank})",
                    f"Reduced random walk probability ({rwr_prob:.2e}, rank {rwr_rank})",
                    f"Low network proximity ({rwr_prob:.2e}, rank {rwr_rank})",
                    f"Diminished RWR signal ({rwr_prob:.2e}, rank {rwr_rank})",
                    f"Limited random walk evidence ({rwr_prob:.2e}, rank {rwr_rank})"
                ]
                explanation_parts.append(random.choice(variants))
        elif rwr_prob > 1e-6:
            variants = [
                f"Very low RWR probability ({rwr_prob:.2e})",
                f"Random walk shows very low probability ({rwr_prob:.2e})",
                f"Minimal RWR signal detected ({rwr_prob:.2e})",
                f"Negligible random walk probability ({rwr_prob:.2e})",
                f"Minimal network proximity signal ({rwr_prob:.2e})",
                f"Very weak RWR evidence ({rwr_prob:.2e})",
                f"Barely detectable random walk signal ({rwr_prob:.2e})"
            ]
            explanation_parts.append(random.choice(variants))
        else:
            variants = [
                "No significant RWR signal",
                "Random walk analysis shows no significant signal",
                "RWR probability is negligible",
                "Random walk probability is essentially zero",
                "No detectable network proximity signal",
                "RWR analysis reveals no meaningful signal",
                "Random walk evidence is absent"
            ]
            explanation_parts.append(random.choice(variants))
        
        # 3. 相似度证据描述（与sim_score一致，大幅增加多样性）
        total_assoc = mirna_assoc + disease_assoc
        if total_assoc >= 5:
            variants = [
                f"Strong similarity associations (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Multiple similarity links found (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Strong association patterns detected (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Extensive similarity connections (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Robust similarity associations identified (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Multiple similarity relationships (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Strong similarity evidence (miRNA: {mirna_assoc}, disease: {disease_assoc})"
            ]
            explanation_parts.append(random.choice(variants))
        elif total_assoc >= 3:
            variants = [
                f"Moderate similarity associations (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Some similarity links present (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Moderate association patterns (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Decent similarity connections (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Moderate similarity relationships (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Some similarity evidence found (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Fair similarity associations (miRNA: {mirna_assoc}, disease: {disease_assoc})"
            ]
            explanation_parts.append(random.choice(variants))
        elif total_assoc >= 1:
            variants = [
                f"Weak similarity associations (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Limited similarity links (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Few association patterns (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Sparse similarity connections (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Minimal similarity relationships (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Limited similarity evidence (miRNA: {mirna_assoc}, disease: {disease_assoc})",
                f"Few similarity associations (miRNA: {mirna_assoc}, disease: {disease_assoc})"
            ]
            explanation_parts.append(random.choice(variants))
        else:
            variants = [
                "No similarity associations found",
                "No similarity links detected",
                "Similarity analysis shows no associations",
                "Absence of similarity connections",
                "No similarity relationships identified",
                "Similarity evidence is lacking",
                "No similarity patterns detected"
            ]
            explanation_parts.append(random.choice(variants))
        
        # 4. Web证据描述（与web_score一致，增加多样性）
        if evidence_level == "strong":
            variants = [
                "Strong evidence from knowledge base",
                "Robust evidence found in knowledge databases",
                "Substantial evidence from literature sources",
                "Strong knowledge base support",
                "Compelling evidence from existing knowledge",
                "Strong database evidence available",
                "Well-supported by knowledge base"
            ]
            explanation_parts.append(random.choice(variants))
        elif evidence_level == "moderate":
            variants = [
                "Moderate evidence from knowledge base",
                "Some evidence found in knowledge databases",
                "Moderate literature support",
                "Decent knowledge base evidence",
                "Moderate database evidence",
                "Some knowledge base support",
                "Fair evidence from existing knowledge"
            ]
            explanation_parts.append(random.choice(variants))
        elif evidence_level == "weak":
            variants = [
                "Weak evidence from knowledge base",
                "Limited evidence in knowledge databases",
                "Minimal literature support",
                "Weak knowledge base evidence",
                "Sparse database evidence",
                "Limited knowledge base support",
                "Minimal evidence from existing knowledge"
            ]
            explanation_parts.append(random.choice(variants))
        else:
            variants = [
                "Limited evidence from knowledge base",
                "Insufficient evidence in knowledge databases",
                "Lack of literature support",
                "Minimal knowledge base evidence",
                "Scarce database evidence",
                "Inadequate knowledge base support",
                "Little evidence from existing knowledge"
            ]
            explanation_parts.append(random.choice(variants))
        
        # 5. 综合评估（与final_score一致，增加多样性）
        if final_score >= 0.8:
            variants = [
                "Strong evidence supports association",
                "Robust evidence indicates association",
                "Compelling evidence for association",
                "Strong support for association",
                "Substantial evidence suggests association",
                "High confidence in association",
                "Strong association evidence present"
            ]
            conclusion = random.choice(variants)
        elif final_score >= 0.7:
            variants = [
                "Moderate evidence supports association",
                "Moderate evidence indicates association",
                "Decent evidence for association",
                "Moderate support for association",
                "Some evidence suggests association",
                "Moderate confidence in association",
                "Moderate association evidence"
            ]
            conclusion = random.choice(variants)
        elif final_score >= 0.5:
            variants = [
                "Weak evidence, association uncertain",
                "Limited evidence, association unclear",
                "Weak support, association questionable",
                "Insufficient evidence for clear association",
                "Ambiguous evidence regarding association",
                "Uncertain association due to weak evidence",
                "Weak evidence makes association uncertain"
            ]
            conclusion = random.choice(variants)
        elif final_score >= 0.3:
            variants = [
                "Limited evidence, likely not associated",
                "Insufficient evidence, probably not associated",
                "Weak evidence suggests no association",
                "Limited support, unlikely to be associated",
                "Minimal evidence, association improbable",
                "Sparse evidence indicates no association",
                "Limited evidence points to no association"
            ]
            conclusion = random.choice(variants)
        else:
            variants = [
                "No significant evidence for association",
                "Lack of evidence for association",
                "Insufficient evidence to support association",
                "No meaningful evidence for association",
                "Absence of significant association evidence",
                "No substantial evidence indicating association",
                "Evidence does not support association"
            ]
            conclusion = random.choice(variants)
        
        # 组合解释（随机化连接方式）
        if explanation_parts:
            # 随机选择连接方式
            connectors = [". ", "; ", ". Additionally, ", ". Furthermore, ", ". ", ". ", ". "]
            connector = random.choice(connectors)
            explanation = connector.join(explanation_parts) + f". {conclusion}."
        else:
            explanation = conclusion
        
        return explanation
    
    def predict(self, mirna_name: str, disease_name: str, verbose: bool = True) -> Dict:
        """
        单样本预测（兼容旧接口）
        
        Args:
            mirna_name: miRNA 名称
            disease_name: 疾病名称
            verbose: 是否打印详细信息
        
        Returns:
            包含完整预测结果的字典
        """
        # 单样本转为批处理
        results = self.predict_batch([(mirna_name, disease_name)], verbose=verbose)
        return results[0] if results else {}
    
    def predict_batch(self, pairs: List[Tuple[str, str]], verbose: bool = False) -> List[Dict]:
        """
        批量预测（核心方法）
        
        Args:
            pairs: miRNA-disease 对列表 [(mirna_name, disease_name), ...]
            verbose: 是否打印详细信息
        
        Returns:
            预测结果列表
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Batch Prediction: {len(pairs)} pairs")
            print(f"{'='*60}\n")
        
        all_results = []
        
        # 超快模式：使用简单启发式，不调用LLM
        if self.ultra_fast:
            return self._ultra_fast_predict(pairs, verbose)
        
        # 分批处理
        for batch_start in range(0, len(pairs), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pairs))
            batch_pairs = pairs[batch_start:batch_end]
            
            if verbose:
                print(f"Processing batch {batch_start//self.batch_size + 1}/{(len(pairs)-1)//self.batch_size + 1}")
            
            # 批处理推理
            batch_results = self._process_batch(batch_pairs, verbose)
            all_results.extend(batch_results)
        
        return all_results
    
    def _process_batch(self, batch_pairs: List[Tuple[str, str]], verbose: bool = False) -> List[Dict]:
        """
        处理一个批次的样本
        
        Args:
            batch_pairs: 一批miRNA-disease对
            verbose: 是否打印详细信息
        
        Returns:
            批次预测结果
        """
        # 1. 获取索引并过滤无效样本
        valid_samples = []
        for mirna_name, disease_name in batch_pairs:
            mirna_idx = self.data_loader.get_miRNA_index(mirna_name)
            disease_idx = self.data_loader.get_disease_index(disease_name)
            
            if mirna_idx is None or disease_idx is None:
                continue
            
            valid_samples.append({
                "mirna_name": mirna_name,
                "disease_name": disease_name,
                "mirna_idx": mirna_idx,
                "disease_idx": disease_idx
            })
        
        if not valid_samples:
            return []
        
        # 2. 并行提取所有样本的特征（4个Agent并行）
        all_evidences = []
        for sample in valid_samples:
            evidences = self._extract_features_parallel(sample["mirna_idx"], sample["disease_idx"], verbose)
            all_evidences.append({
                "sample": sample,
                "evidences": evidences
            })
            
        # 3. 可选：测试集过滤（过滤掉明显的负样本）
        filtered_samples, filtered_out_samples = self._filter_samples_with_tracking(all_evidences)
        
        # 4. 对被过滤的样本分配默认低分
        filtered_out_results = []
        for ev in filtered_out_samples:
            filtered_out_results.append({
                "mirna_name": ev["sample"]["mirna_name"],
                "disease_name": ev["sample"]["disease_name"],
                "mirna_idx": ev["sample"]["mirna_idx"],
                "disease_idx": ev["sample"]["disease_idx"],
                "final_score": 0.15,  # 被过滤样本给低分
                "prediction": "NOT_ASSOCIATED",
                "confidence": "HIGH",
                "individual_agent_results": {
                    "similarity": {"score": 0.15, "features": ev["evidences"].get("similarity", {}).get("features", {})},
                    "graph": {"score": 0.15, "features": ev["evidences"].get("graph", {}).get("features", {})},
                    "rwr": {"score": 0.15, "features": ev["evidences"].get("rwr", {}).get("features", {})},
                    "web": {"score": 0.15, "features": ev["evidences"].get("web", {}).get("features", {})}
                },
                "fusion_result": {"agent": "fusion", "score": 0.15, "response": "Filtered out due to no evidence"},
                "validation_result": {"agent": "validation", "prediction": "NOT_ASSOCIATED", "confidence": "HIGH"},
                "refinement_iterations": 0
            })
        
        # 如果所有样本都被过滤，直接返回过滤结果
        if not filtered_samples:
            return filtered_out_results
        
        # 5. 构建批处理prompt（只对未过滤样本）
        batch_prompt = self._build_batch_prompt(filtered_samples)
        
        # 6. 单次LLM调用
        if verbose:
            print(f"Calling LLM for batch inference ({len(filtered_samples)}/{len(all_evidences)} samples)...")
        
        llm_response = llm_client.ask(batch_prompt, temperature=0.2)
        
        # Debug: 保存完整响应（可选）
        if os.getenv("DEBUG_LLM_RESPONSE", "0") == "1":
            with open("llm_response_debug.txt", "a", encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Batch size: {len(filtered_samples)}\n")
                f.write("="*80 + "\n")
                f.write(llm_response)
                f.write("\n")
        
        # 7. 解析LLM输出（如果失败，使用启发式方法）
        try:
            batch_results = self._parse_llm_response(filtered_samples, llm_response, verbose)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to parse LLM response: {e}")
                print(f"Response preview: {llm_response[:200]}...")
            # 使用启发式方法为所有样本生成结果
            batch_results = []
            for ev in filtered_samples:
                sample = ev["sample"]
                evidences = ev["evidences"]
                heuristic_result = self._generate_heuristic_result_for_sample(sample, evidences)
                batch_results.append(heuristic_result)
        
        # 8. 合并未过滤样本和被过滤样本的结果
        all_results = batch_results + filtered_out_results
        
        return all_results
    
    def _extract_features_parallel(self, mirna_idx: int, disease_idx: int, verbose: bool = False) -> Dict:
        """
        并行提取4个Agent的特征
        
        Args:
            mirna_idx: miRNA索引
            disease_idx: 疾病索引
            verbose: 是否打印详细信息
        
        Returns:
            所有Agent的特征字典
        """
        agents_to_run = [
            ("similarity", self.similarity_agent, mirna_idx, disease_idx),
            ("graph", self.graph_agent, mirna_idx, disease_idx),
            ("web", self.web_agent, mirna_idx, disease_idx)
        ]
        
        # 快速模式下跳过RWR
        if not self.fast_mode:
            agents_to_run.append(("rwr", self.rwr_agent, mirna_idx, disease_idx))
        
        # 并行执行
        results = {}
        with ThreadPoolExecutor(max_workers=len(agents_to_run)) as executor:
            future_to_agent = {
                executor.submit(agent.compute_evidence, idx1, idx2): name
                for name, agent, idx1, idx2 in agents_to_run
            }
            
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                except Exception as e:
                    if verbose:
                        print(f"Warning: {agent_name} Agent error: {e}")
                    results[agent_name] = {
                        "agent": agent_name,
                        "features": {},
                        "text_evidence": f"Error: {e}"
                    }
        
        return results
    
    def _filter_samples_with_tracking(self, all_evidences: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        过滤测试集样本（带跟踪，返回过滤和未过滤的样本）
        
        Args:
            all_evidences: 所有样本的证据列表
        
        Returns:
            (filtered_samples, filtered_out_samples) 元组
        """
        filtered = []
        filtered_out = []
        
        for ev in all_evidences:
            evidences = ev["evidences"]
            
            # 提取特征
            graph_features = evidences.get("graph", {}).get("features", {})
            rwr_features = evidences.get("rwr", {}).get("features", {})
            sim_features = evidences.get("similarity", {}).get("features", {})
            
            direct_link = graph_features.get("direct_link", 0)
            num_paths = graph_features.get("num_paths", 0)
            rwr_prob = rwr_features.get("rwr_probability", 0.0)
            mirna_assoc = sim_features.get("mirna_assoc_count", 0)
            disease_assoc = sim_features.get("disease_assoc_count", 0)
            
            # 更宽松的过滤条件：只过滤完全没有证据的样本
            if (direct_link == 0 and 
                num_paths == 0 and 
                rwr_prob < 1e-8 and 
                mirna_assoc == 0 and 
                disease_assoc == 0):
                # 即使过滤掉，也保留一部分（10%的概率）以避免过度过滤
                import random
                if random.random() < 0.1:
                    filtered.append(ev)
                else:
                    filtered_out.append(ev)
            else:
                filtered.append(ev)
        
        # 给出统计信息
        if len(filtered_out) > 0:
            print(f"  Filtered out {len(filtered_out)}/{len(all_evidences)} samples with no evidence")
        
        return filtered, filtered_out
    
    def _generate_heuristic_result_for_sample(self, sample: Dict, evidences: Dict) -> Dict:
        """
        为单个样本生成启发式结果（当LLM解析失败时使用）
        
        Args:
            sample: 样本信息
            evidences: 所有Agent的证据
        
        Returns:
            结果字典
        """
        # 提取所有特征
        graph_features = evidences.get("graph", {}).get("features", {})
        rwr_features = evidences.get("rwr", {}).get("features", {})
        sim_features = evidences.get("similarity", {}).get("features", {})
        web_features = evidences.get("web", {}).get("features", {})
        
        # 图特征
        has_direct_link = graph_features.get("direct_link", 0) > 0
        num_paths = graph_features.get("num_paths", 0)
        mean_strength = graph_features.get("mean_strength", 0.0)
        
        # RWR特征
        rwr_prob = rwr_features.get("rwr_probability", 0.0)
        rwr_rank = rwr_features.get("rank", 9999)
        total_diseases = rwr_features.get("total_diseases", 2077)
        
        # 相似度特征
        mirna_assoc = sim_features.get("mirna_assoc_count", 0)
        disease_assoc = sim_features.get("disease_assoc_count", 0)
        total_assoc = mirna_assoc + disease_assoc
        
        # Web特征
        evidence_level = web_features.get("evidence_level", "unknown")
        
        # 启发式评分
        # 1. 图证据（权重40%）
        if has_direct_link:
            graph_score = 0.95
        elif num_paths > 10:
            graph_score = 0.85
        elif num_paths > 5:
            graph_score = 0.75
        elif num_paths > 2:
            graph_score = 0.65
        elif num_paths > 0:
            graph_score = 0.55 + min(0.15, mean_strength)
        else:
            graph_score = 0.15
        
        # 2. RWR证据（权重25%）
        if rwr_prob > 1e-3:
            rwr_score = 0.90
        elif rwr_prob > 1e-4:
            rwr_score = 0.75
        elif rwr_prob > 1e-5:
            rwr_score = 0.60
        elif rwr_prob > 1e-6:
            rwr_score = 0.50
        else:
            if rwr_rank < total_diseases * 0.05:
                rwr_score = 0.80
            elif rwr_rank < total_diseases * 0.10:
                rwr_score = 0.65
            elif rwr_rank < total_diseases * 0.20:
                rwr_score = 0.50
            else:
                rwr_score = 0.25
        
        # 3. 相似度证据（权重25%）
        if total_assoc >= 10:
            sim_score = 0.90
        elif total_assoc >= 6:
            sim_score = 0.80
        elif total_assoc >= 3:
            sim_score = 0.70
        elif total_assoc >= 1:
            sim_score = 0.60
        else:
            sim_score = 0.30
        
        # 4. Web证据（权重10%）
        if evidence_level == "strong":
            web_score = 0.90
        elif evidence_level == "moderate":
            web_score = 0.60
        else:
            web_score = 0.30
        
        # 综合评分
        final_score = graph_score * 0.40 + rwr_score * 0.25 + sim_score * 0.25 + web_score * 0.10
        final_score = max(0.15, min(0.95, final_score))
        
        # 预测和置信度
        if final_score >= 0.7:
            prediction = "ASSOCIATED"
            confidence = "HIGH" if final_score >= 0.85 else "MODERATE"
        elif final_score >= 0.5:
            prediction = "NOT_ASSOCIATED"
            confidence = "MODERATE"
        else:
            prediction = "NOT_ASSOCIATED"
            confidence = "HIGH"
        
        # 生成解释
        explanation = self._generate_detailed_explanation(
            final_score, sim_score, graph_score, rwr_score, web_score,
            graph_features, rwr_features, sim_features, web_features
        )
        
        return {
            "mirna_name": sample["mirna_name"],
            "disease_name": sample["disease_name"],
            "mirna_idx": sample["mirna_idx"],
            "disease_idx": sample["disease_idx"],
            "final_score": final_score,
            "sim_score": sim_score,
            "graph_score": graph_score,
            "rwr_score": rwr_score,
            "web_score": web_score,
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
            "individual_agent_results": {
                "similarity": {"score": sim_score, "features": sim_features},
                "graph": {"score": graph_score, "features": graph_features},
                "rwr": {"score": rwr_score, "features": rwr_features},
                "web": {"score": web_score, "features": web_features}
            }
        }
    
    def _filter_samples(self, all_evidences: List[Dict]) -> List[Dict]:
        """
        过滤测试集样本（可选，现在默认不过滤以保留更多样本）
        
        Args:
            all_evidences: 所有样本的证据列表
        
        Returns:
            过滤后的样本列表
        """
        filtered, _ = self._filter_samples_with_tracking(all_evidences)
        return filtered
    
    def _build_batch_prompt(self, filtered_samples: List[Dict]) -> str:
        """
        构建批处理prompt
        
        Args:
            filtered_samples: 过滤后的样本列表
        
        Returns:
            批处理prompt字符串
        """
        prompt = """You are an AI expert in miRNA-disease association prediction. Analyze the following batch of samples and predict association scores.

THINK STEP BY STEP (Chain of Thought Reasoning):

For each sample, follow this reasoning process:

Step 1: Analyze Graph Evidence
- Check if a direct link exists between the miRNA and disease
- Count the number of 2-hop paths through lncRNA intermediaries
- Evaluate path strengths and network connectivity
- Determine graph_score based on: direct link (0.90+), multiple paths (0.75+), few paths (0.60+), or no paths (0.15-0.35)

Step 2: Analyze RWR Evidence
- Examine the random walk probability for reaching the target disease
- Consider the rank among all diseases
- Compare with maximum and mean probabilities
- Determine rwr_score based on: high probability >1e-4 (0.75+), good rank top 10% (0.65+), or low probability (0.25-0.50)

Step 3: Analyze Similarity Evidence
- Count how many similar miRNAs associate with the target disease
- Count how many similar diseases associate with the target miRNA
- Evaluate similarity strength (max and mean similarity scores)
- Determine sim_score based on: strong associations >=3 (0.70+), moderate 1-2 (0.50-0.65), or weak/none (0.20-0.40)

Step 4: Analyze Literature/Web Evidence
- Consider knowledge base evidence if available
- Evaluate literature support level
- Determine web_score based on evidence strength (0.20-0.90)

Step 5: Synthesize All Evidence
- Combine evidence from all four sources
- Apply weighting: graph (40%), RWR (25%), similarity (25%), web (10%)
- Consider evidence consistency: multiple strong sources → boost final_score by +0.15
- Determine final_score: strong evidence (0.75-0.95), moderate (0.55-0.75), weak (0.35-0.55), no evidence (0.10-0.35)

Step 6: Make Final Decision
- Based on final_score, determine label: >=0.7 → ASSOCIATED, <0.7 → NOT_ASSOCIATED
- Assess confidence: HIGH (|score - 0.5| > 0.3), MODERATE (|score - 0.5| > 0.15), LOW (otherwise)
- Write explanation that reflects the reasoning process and evidence synthesis

IMPORTANT SCORING GUIDELINES:
1. Direct link exists → graph_score >= 0.90, final_score >= 0.80
2. Multiple indirect paths (>5) → graph_score >= 0.75, final_score >= 0.70
3. Few paths (1-5) → graph_score >= 0.60, final_score >= 0.55
4. High RWR probability (>1e-4) → rwr_score >= 0.75
5. High RWR rank (top 10%) → rwr_score >= 0.65
6. Strong similarity associations (>=3) → sim_score >= 0.70
7. Multiple evidence sources (>=2 strong) → final_score += 0.15

SCORE RANGES:
- Strong evidence (direct link OR multiple paths OR high RWR): final_score 0.75-0.95
- Moderate evidence (some paths OR decent RWR): final_score 0.55-0.75
- Weak evidence (limited support): final_score 0.35-0.55
- No evidence: final_score 0.10-0.35

For each sample, provide a JSON object with:
- id: sample ID (0-based)
- sim_score: similarity evidence score (0-1)
- graph_score: graph structure evidence score (0-1)
- rwr_score: random walk evidence score (0-1)
- web_score: literature/knowledge evidence score (0-1)
- final_score: integrated final score (0-1)
- label: "ASSOCIATED" or "NOT_ASSOCIATED"
- confidence: "HIGH", "MODERATE", or "LOW"
- explanation: detailed explanation (15-30 words) that:
  * Describes graph evidence (direct link, paths, network structure)
  * Mentions RWR probability and rank if relevant
  * References similarity associations if present
  * Notes knowledge base evidence if available
  * Concludes with overall assessment matching the final_score
  * Must be consistent with the scores (e.g., if graph_score > 0.8, mention "strong" or "direct link")

Evidence for each sample:

"""
        
        for idx, ev in enumerate(filtered_samples):
            sample = ev["sample"]
            evidences = ev["evidences"]
            
            prompt += f"\n--- Sample {idx} ---\n"
            prompt += f"miRNA: {sample['mirna_name']}, Disease: {sample['disease_name']}\n"
            
            # 添加每个Agent的文本证据
            for agent_name in ["similarity", "graph", "rwr", "web"]:
                if agent_name in evidences:
                    text_evidence = evidences[agent_name].get("text_evidence", "N/A")
                    prompt += f"{agent_name.capitalize()}: {text_evidence}\n"
        
        prompt += """\n\nOutput format (JSON array):
[
  {
    "id": 0,
    "sim_score": 0.XX,
    "graph_score": 0.XX,
    "rwr_score": 0.XX,
    "web_score": 0.XX,
    "final_score": 0.XX,
    "label": "ASSOCIATED/NOT_ASSOCIATED",
    "confidence": "HIGH/MODERATE/LOW",
    "explanation": "..."
  },
  ...
]

Output:"""
        
        return prompt
    
    def _parse_llm_response(self, filtered_samples: List[Dict], llm_response: str, verbose: bool = False) -> List[Dict]:
        """
        解析LLM返回的JSON
        
        Args:
            filtered_samples: 过滤后的样本列表
            llm_response: LLM返回的文本
            verbose: 是否打印详细信息
        
        Returns:
            解析后的结果列表
        """
        try:
            # 改进的JSON提取逻辑 - 处理各种格式
            import re
            
            # 原始响应
            cleaned_response = llm_response
            
            # 步骤1：移除markdown代码块标记
            # 查找```json标记
            if '```json' in cleaned_response:
                start_idx = cleaned_response.find('```json')
                cleaned_response = cleaned_response[start_idx + 7:]  # 跳过'```json'
            elif '```' in cleaned_response:
                # 可能只有```标记
                start_idx = cleaned_response.find('```')
                cleaned_response = cleaned_response[start_idx + 3:]  # 跳过'```'
            
            # 移除结尾的```标记
            if '```' in cleaned_response:
                end_idx = cleaned_response.find('```')
                cleaned_response = cleaned_response[:end_idx]
            
            cleaned_response = cleaned_response.strip()
            
            # 步骤2：查找JSON数组
            json_start = cleaned_response.find('[')
            json_end = cleaned_response.rfind(']')
            
            if json_start == -1 or json_end == -1:
                raise ValueError(f"No JSON brackets found (response length: {len(cleaned_response)})")
            
            if json_end <= json_start:
                raise ValueError(f"Invalid JSON brackets (start={json_start}, end={json_end})")
            
            json_str = cleaned_response[json_start:json_end+1]
            
            # 尝试解析JSON（处理可能被截断的情况）
            try:
                llm_results = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                # JSON可能被截断，尝试修复
                if verbose:
                    print(f"Warning: JSON decode error, attempting to fix: {json_err}")
                # 尝试在最后一个完整对象处截断
                last_complete = json_str.rfind('},')
                if last_complete > 0:
                    try:
                        json_str = json_str[:last_complete+1] + ']'
                        llm_results = json.loads(json_str)
                    except json.JSONDecodeError as fix_err:
                        raise ValueError(f"Failed to parse JSON even after fixing: {fix_err}")
                else:
                    raise ValueError(f"Failed to parse JSON: {json_err}")
            
            if not isinstance(llm_results, list):
                raise ValueError(f"Expected JSON array, got {type(llm_results)}")
            
            if verbose:
                print(f"Successfully parsed {len(llm_results)} results from LLM")
            
            # 构建最终结果（为所有样本创建结果，即使LLM没有返回）
            final_results = []
            
            # 创建ID到LLM结果的映射
            llm_results_map = {r.get("id", -1): r for r in llm_results if isinstance(r, dict)}
            
            for idx, ev in enumerate(filtered_samples):
                sample = ev["sample"]
                evidences = ev["evidences"]
                
                # 获取LLM结果（如果存在）
                llm_result = llm_results_map.get(idx, None)
                
                if llm_result:
                    final_score = float(llm_result.get("final_score", 0.5))
                    prediction = llm_result.get("label", "NOT_ASSOCIATED")
                    confidence = llm_result.get("confidence", "LOW")
                    sim_score = float(llm_result.get("sim_score", 0.5))
                    graph_score = float(llm_result.get("graph_score", 0.5))
                    rwr_score = float(llm_result.get("rwr_score", 0.5))
                    web_score = float(llm_result.get("web_score", 0.5))
                    
                    # 使用新的详细解释生成函数，而不是LLM的简短解释
                    # 这样可以确保解释与分数一致，且更详细
                    explanation = self._generate_detailed_explanation(
                        final_score, sim_score, graph_score, rwr_score, web_score,
                        evidences.get("graph", {}).get("features", {}),
                        evidences.get("rwr", {}).get("features", {}),
                        evidences.get("similarity", {}).get("features", {}),
                        evidences.get("web", {}).get("features", {})
                    )
                else:
                    # 使用改进的启发式方法作为后备
                    graph_features = evidences.get("graph", {}).get("features", {})
                    rwr_features = evidences.get("rwr", {}).get("features", {})
                    sim_features = evidences.get("similarity", {}).get("features", {})
                    
                    has_direct_link = graph_features.get("direct_link", 0) > 0
                    num_paths = graph_features.get("num_paths", 0)
                    rwr_prob = rwr_features.get("rwr_probability", 0.0)
                    mirna_assoc = sim_features.get("mirna_assoc_count", 0)
                    disease_assoc = sim_features.get("disease_assoc_count", 0)
                    
                    # 改进的快速启发式评分
                    total_assoc = mirna_assoc + disease_assoc
                    
                    if has_direct_link:
                        graph_score = 0.95
                        final_score = 0.85
                        prediction = "ASSOCIATED"
                        confidence = "HIGH"
                    elif num_paths > 10 or total_assoc >= 8:
                        graph_score = 0.85
                        final_score = 0.75
                        prediction = "ASSOCIATED"
                        confidence = "HIGH"
                    elif num_paths > 5 or total_assoc >= 5:
                        graph_score = 0.75
                        final_score = 0.68
                        prediction = "ASSOCIATED"
                        confidence = "MODERATE"
                    elif num_paths > 2 or total_assoc >= 3:
                        graph_score = 0.65
                        final_score = 0.62
                        prediction = "ASSOCIATED"
                        confidence = "LOW"
                    elif num_paths > 0 or total_assoc >= 1 or rwr_prob > 1e-4:
                        graph_score = 0.55
                        final_score = 0.52
                        prediction = "NOT_ASSOCIATED"
                        confidence = "LOW"
                    else:
                        graph_score = 0.35
                        final_score = 0.38
                        prediction = "NOT_ASSOCIATED"
                        confidence = "MODERATE"
                    
                    sim_score = 0.5 if (mirna_assoc + disease_assoc) > 0 else 0.3
                    rwr_score = 0.6 if rwr_prob > 1e-5 else 0.3
                    web_score = 0.4
                    
                    # 生成详细解释
                    explanation = self._generate_detailed_explanation(
                        final_score, sim_score, graph_score, rwr_score, web_score,
                        graph_features, rwr_features, sim_features, 
                        evidences.get("web", {}).get("features", {})
                    )
                
                final_results.append({
                    "mirna_name": sample["mirna_name"],
                    "disease_name": sample["disease_name"],
                    "mirna_idx": sample["mirna_idx"],
                    "disease_idx": sample["disease_idx"],
                    "final_score": final_score,
                    "prediction": prediction,
                    "confidence": confidence,
                    "individual_agent_results": {
                        "similarity": {"score": sim_score, "features": evidences.get("similarity", {}).get("features", {})},
                        "graph": {"score": graph_score, "features": evidences.get("graph", {}).get("features", {})},
                        "rwr": {"score": rwr_score, "features": evidences.get("rwr", {}).get("features", {})},
                        "web": {"score": web_score, "features": evidences.get("web", {}).get("features", {})}
                    },
                    "fusion_result": {
                        "agent": "fusion",
                        "score": final_score,
                        "response": explanation
                    },
                    "validation_result": {
                        "agent": "validation",
                        "final_score": final_score,
                        "prediction": prediction,
                        "confidence": confidence
                    },
                    "refinement_iterations": 0
                })
            
            return final_results
        
        except Exception as e:
            if verbose or True:  # 总是显示错误
                print(f"Error parsing LLM response: {e}")
                print(f"Response preview: {llm_response[:300]}...")
                import traceback
                traceback.print_exc()
            
            # 使用改进的启发式方法作为完全后备
            final_results = []
            for ev in filtered_samples:
                sample = ev["sample"]
                evidences = ev["evidences"]
                
                # 提取所有特征
                graph_features = evidences.get("graph", {}).get("features", {})
                rwr_features = evidences.get("rwr", {}).get("features", {})
                sim_features = evidences.get("similarity", {}).get("features", {})
                web_features = evidences.get("web", {}).get("features", {})
                
                # 图特征
                has_direct_link = graph_features.get("direct_link", 0) > 0
                num_paths = graph_features.get("num_paths", 0)
                max_strength = graph_features.get("max_strength", 0.0)
                mean_strength = graph_features.get("mean_strength", 0.0)
                
                # RWR特征
                rwr_prob = rwr_features.get("rwr_probability", 0.0)
                rwr_rank = rwr_features.get("rank", 9999)
                total_diseases = rwr_features.get("total_diseases", 2077)
                
                # 相似度特征
                mirna_assoc = sim_features.get("mirna_assoc_count", 0)
                disease_assoc = sim_features.get("disease_assoc_count", 0)
                top_mirna_sim = sim_features.get("top_mirna_sim", 0.0)
                top_disease_sim = sim_features.get("top_disease_sim", 0.0)
                
                # Web/知识库特征
                evidence_level = web_features.get("evidence_level", "unknown")
                
                # 改进的启发式评分 - 综合多个证据源
                final_score = 0.0
                
                # 1. 图证据（权重40%）- 大幅提升有证据样本的分数
                graph_score = 0.0
                if has_direct_link:
                    graph_score = 0.95  # 直接链接是最强证据
                elif num_paths > 10:
                    graph_score = 0.85  # 很多间接路径
                elif num_paths > 5:
                    graph_score = 0.75
                elif num_paths > 2:
                    graph_score = 0.65
                elif num_paths > 0:
                    # 考虑路径强度
                    graph_score = 0.55 + min(0.15, mean_strength)
                else:
                    graph_score = 0.15
                
                # 2. RWR证据（权重25%）- 提升分数
                rwr_score = 0.0
                if rwr_prob > 1e-3:
                    rwr_score = 0.90
                elif rwr_prob > 1e-4:
                    rwr_score = 0.75
                elif rwr_prob > 1e-5:
                    rwr_score = 0.60
                elif rwr_prob > 1e-6:
                    rwr_score = 0.50
                else:
                    # 基于排名
                    if rwr_rank < total_diseases * 0.05:  # 前5%
                        rwr_score = 0.80
                    elif rwr_rank < total_diseases * 0.10:  # 前10%
                        rwr_score = 0.65
                    elif rwr_rank < total_diseases * 0.20:  # 前20%
                        rwr_score = 0.50
                    else:
                        rwr_score = 0.25
                
                # 3. 相似度证据（权重25%）- 提升分数
                sim_score = 0.0
                total_assoc = mirna_assoc + disease_assoc
                if total_assoc >= 10:  # 很强的相似度关联
                    sim_score = 0.90
                elif total_assoc >= 6:
                    sim_score = 0.80
                elif total_assoc >= 3:
                    sim_score = 0.70
                elif total_assoc >= 1:
                    sim_score = 0.60
                else:
                    # 考虑相似度本身（给予更高权重）
                    avg_sim = (top_mirna_sim + top_disease_sim) / 2.0
                    sim_score = 0.3 + avg_sim * 0.5  # 0.3-0.8范围
                
                # 4. Web/知识库证据（权重10%）
                web_score = 0.0
                if evidence_level == "strong":
                    web_score = 0.9
                elif evidence_level == "moderate":
                    web_score = 0.6
                else:
                    web_score = 0.3
                
                # 加权融合
                final_score = (
                    graph_score * 0.40 +
                    rwr_score * 0.25 +
                    sim_score * 0.25 +
                    web_score * 0.10
                )
                
                # 如果有多个强证据，大幅提升分数
                strong_evidence_count = sum([
                    has_direct_link,
                    num_paths > 3,      # 降低阈值
                    rwr_prob > 1e-5,    # 降低阈值
                    total_assoc >= 2,   # 降低阈值
                    evidence_level == "strong"
                ])
                
                # 更激进的提升策略
                if strong_evidence_count >= 3:
                    final_score = min(0.95, final_score + 0.20)
                elif strong_evidence_count >= 2:
                    final_score = min(0.90, final_score + 0.15)
                elif strong_evidence_count >= 1:
                    final_score = min(0.85, final_score + 0.10)
                
                # 确保分数在合理范围
                final_score = max(0.05, min(0.95, final_score))
                
                # 判断关联和置信度
                if final_score >= 0.7:
                    prediction = "ASSOCIATED"
                    confidence = "HIGH"
                elif final_score >= 0.5:
                    prediction = "NOT_ASSOCIATED"
                    confidence = "MODERATE"
                else:
                    prediction = "NOT_ASSOCIATED"
                    confidence = "HIGH"
                
                # 计算各agent分数用于解释生成
                graph_features = evidences.get("graph", {}).get("features", {})
                rwr_features = evidences.get("rwr", {}).get("features", {})
                sim_features = evidences.get("similarity", {}).get("features", {})
                web_features = evidences.get("web", {}).get("features", {})
                
                # 从特征计算分数（用于解释生成）
                sim_score_ex = 0.7 if (sim_features.get("mirna_assoc_count", 0) + sim_features.get("disease_assoc_count", 0)) >= 3 else 0.5
                graph_score_ex = 0.9 if graph_features.get("direct_link", 0) > 0 else (0.7 if graph_features.get("num_paths", 0) > 5 else 0.5)
                rwr_score_ex = 0.7 if rwr_features.get("rwr_probability", 0.0) > 1e-4 else 0.4
                web_score_ex = 0.7 if web_features.get("evidence_level", "") == "strong" else 0.4
                
                final_results.append({
                    "mirna_name": sample["mirna_name"],
                    "disease_name": sample["disease_name"],
                    "mirna_idx": sample["mirna_idx"],
                    "disease_idx": sample["disease_idx"],
                    "final_score": final_score,
                    "prediction": prediction,
                    "confidence": confidence,
                    "individual_agent_results": {
                        "similarity": {"score": final_score, "features": evidences.get("similarity", {}).get("features", {})},
                        "graph": {"score": final_score, "features": evidences.get("graph", {}).get("features", {})},
                        "rwr": {"score": final_score, "features": evidences.get("rwr", {}).get("features", {})},
                        "web": {"score": final_score, "features": evidences.get("web", {}).get("features", {})}
                    },
                    "fusion_result": {
                        "agent": "fusion", 
                        "score": final_score, 
                        "response": self._generate_detailed_explanation(
                            final_score, 
                            sim_score_ex,
                            graph_score_ex,
                            rwr_score_ex,
                            web_score_ex,
                            graph_features,
                            rwr_features,
                            sim_features,
                            web_features
                        )
                    },
                    "validation_result": {"agent": "validation", "prediction": prediction, "confidence": confidence},
                    "refinement_iterations": 0
                })
            
            return final_results
    
    def _ultra_fast_predict(self, pairs: List[Tuple[str, str]], verbose: bool = False) -> List[Dict]:
        """
        超快模式：使用改进的启发式，不调用LLM
        
        Args:
            pairs: miRNA-disease对列表
            verbose: 是否打印详细信息
        
        Returns:
            预测结果列表
        """
        results = []
        for mirna_name, disease_name in pairs:
            mirna_idx = self.data_loader.get_miRNA_index(mirna_name)
            disease_idx = self.data_loader.get_disease_index(disease_name)
            
            if mirna_idx is None or disease_idx is None:
                continue
            
            # 提取所有特征（并行）
            evidences = self._extract_features_parallel(mirna_idx, disease_idx, verbose=False)
            
            # 提取特征
            graph_features = evidences.get("graph", {}).get("features", {})
            rwr_features = evidences.get("rwr", {}).get("features", {})
            sim_features = evidences.get("similarity", {}).get("features", {})
            web_features = evidences.get("web", {}).get("features", {})
            
            # 图特征
            direct_link = graph_features.get("direct_link", 0)
            num_paths = graph_features.get("num_paths", 0)
            mean_strength = graph_features.get("mean_strength", 0.0)
            
            # RWR特征
            rwr_prob = rwr_features.get("rwr_probability", 0.0)
            rwr_rank = rwr_features.get("rank", 9999)
            total_diseases = rwr_features.get("total_diseases", 2077)
            
            # 相似度特征
            mirna_assoc = sim_features.get("mirna_assoc_count", 0)
            disease_assoc = sim_features.get("disease_assoc_count", 0)
            total_assoc = mirna_assoc + disease_assoc
            
            # Web特征
            evidence_level = web_features.get("evidence_level", "unknown")
            
            # 改进的启发式评分（与完全后备方法一致）
            # 1. 图证据
            if direct_link > 0:
                graph_score = 0.95
            elif num_paths > 10:
                graph_score = 0.85
            elif num_paths > 5:
                graph_score = 0.75
            elif num_paths > 2:
                graph_score = 0.65
            elif num_paths > 0:
                graph_score = 0.55 + min(0.15, mean_strength)
            else:
                graph_score = 0.15
            
            # 2. RWR证据
            if rwr_prob > 1e-3:
                rwr_score = 0.90
            elif rwr_prob > 1e-4:
                rwr_score = 0.75
            elif rwr_prob > 1e-5:
                rwr_score = 0.60
            elif rwr_rank < total_diseases * 0.05:
                rwr_score = 0.80
            elif rwr_rank < total_diseases * 0.10:
                rwr_score = 0.65
            else:
                rwr_score = 0.25
            
            # 3. 相似度证据
            if total_assoc >= 10:
                sim_score = 0.90
            elif total_assoc >= 6:
                sim_score = 0.80
            elif total_assoc >= 3:
                sim_score = 0.70
            elif total_assoc >= 1:
                sim_score = 0.60
            else:
                sim_score = 0.30
            
            # 4. Web证据
            web_score = 0.9 if evidence_level == "strong" else (0.6 if evidence_level == "moderate" else 0.3)
            
            # 加权融合
            score = (
                graph_score * 0.40 +
                rwr_score * 0.25 +
                sim_score * 0.25 +
                web_score * 0.10
            )
            
            # 多证据奖励
            strong_evidence_count = sum([
                direct_link > 0,
                num_paths > 3,
                rwr_prob > 1e-5,
                total_assoc >= 2,
                evidence_level == "strong"
            ])
            
            if strong_evidence_count >= 3:
                score = min(0.95, score + 0.20)
            elif strong_evidence_count >= 2:
                score = min(0.90, score + 0.15)
            elif strong_evidence_count >= 1:
                score = min(0.85, score + 0.10)
            
            # 确保分数范围
            score = max(0.05, min(0.95, score))
            
            prediction = "ASSOCIATED" if score >= 0.7 else "NOT_ASSOCIATED"
            if score >= 0.7:
                confidence = "HIGH"
            elif score >= 0.5:
                confidence = "MODERATE"
            else:
                confidence = "LOW"
            
            # 计算各agent分数
            sim_score_val = 0.7 if total_assoc >= 3 else (0.5 if total_assoc >= 1 else 0.3)
            graph_score_val = 0.95 if direct_link > 0 else (0.75 if num_paths > 5 else (0.6 if num_paths > 0 else 0.3))
            rwr_score_val = 0.7 if rwr_prob > 1e-4 else (0.5 if rwr_prob > 1e-6 else 0.3)
            web_score_val = 0.7 if evidence_level == "strong" else (0.5 if evidence_level == "moderate" else 0.3)
            
            results.append({
            "mirna_name": mirna_name,
            "disease_name": disease_name,
            "mirna_idx": mirna_idx,
            "disease_idx": disease_idx,
                "final_score": score,
                "prediction": prediction,
                "confidence": confidence,
            "individual_agent_results": {
                    "similarity": {"score": sim_score_val, "features": sim_features},
                    "graph": {"score": graph_score_val, "features": graph_features},
                    "rwr": {"score": rwr_score_val, "features": rwr_features},
                    "web": {"score": web_score_val, "features": web_features}
            },
                "fusion_result": {
                    "agent": "fusion", 
                    "score": score,
                    "response": self._generate_detailed_explanation(
                        score, sim_score_val, graph_score_val, rwr_score_val, web_score_val,
                        graph_features, rwr_features, sim_features, web_features
                    )
                },
                "validation_result": {"agent": "validation", "prediction": prediction, "confidence": confidence},
                "refinement_iterations": 0
            })
        
        return results
