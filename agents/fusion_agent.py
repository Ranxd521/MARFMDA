"""
Fusion/Generator Agent Module
Responsible for generating the initial prediction and refining it based on critique.
Acts as the 'Generator' and 'Refiner' node in the LangGraph.
"""
import sys
import os
import json
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.config import get_llm
from agents.state import AgentState

class FusionAgent:
    """
    Legacy Fusion Agent class for ControllerAgent compatibility.
    Used by batch prediction mode.
    """
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def prepare_evidence(self, agent_results: List[Dict], mirna_name: str, disease_name: str) -> Dict:
        all_features = {}
        text_evidences = []
        
        for result in agent_results:
            agent_name = result.get("agent", "unknown")
            all_features[agent_name] = result.get("features", {})
            text_evidence = result.get("text_evidence", "")
            if text_evidence:
                text_evidences.append(f"{agent_name.capitalize()}: {text_evidence}")
        
        heuristic_result = self._heuristic_judgment(all_features)
        
        return {
            "agent": "fusion",
            "mirna_name": mirna_name,
            "disease_name": disease_name,
            "all_features": all_features,
            "text_evidences": text_evidences,
            "heuristic": heuristic_result
        }
    
    def _heuristic_judgment(self, all_features: Dict[str, Dict]) -> Dict:
        graph_features = all_features.get("graph", {})
        has_direct_link = graph_features.get("direct_link", 0) > 0
        
        rwr_features = all_features.get("rwr", {})
        rwr_prob = rwr_features.get("rwr_probability", 0.0)
        
        sim_features = all_features.get("similarity", {})
        mirna_assoc = sim_features.get("mirna_assoc_count", 0)
        disease_assoc = sim_features.get("disease_assoc_count", 0)
        
        if has_direct_link:
            return {"can_skip_llm": True, "suggested_score": 0.9, "reason": "Direct link exists"}
        elif mirna_assoc == 0 and disease_assoc == 0 and rwr_prob < 1e-6:
            return {"can_skip_llm": True, "suggested_score": 0.1, "reason": "All evidence very weak"}
        else:
            return {"can_skip_llm": False, "suggested_score": None, "reason": "Need LLM judgment"}

# Template for Initial Generation
INITIAL_GEN_TEMPLATE = """You are an expert Biologist and Data Scientist.
Your task is to predict the association between a miRNA and a Disease based on computational evidence.

Target: {mirna_id} - {disease_id}

=== COMPUTATIONAL EVIDENCE ===
{feature_context}

=== SCORING RULES - CRITICAL ===
[CRITICAL SCORING CALIBRATION]
You must be DECISIVE. Avoid scores around 0.5 unless evidence is truly contradictory.
Use the following Evidence-to-Score Mapping:

1. **High Confidence (0.75 - 0.99)**:
   - **RWR Rank Top 100** (Top 5%). (e.g. Rank 1-100)
   - OR RWR Rank Top 300 AND (Disease Similarity > 0.7 OR miRNA Similarity > 0.7).
   - OR Any Direct Graph Path exists.

2. **Moderate Confidence (0.55 - 0.74)**:
   - **RWR Rank 101-300** with weak similarity.
   - OR **RWR Rank 301-800** BUT with **Strong Similarity** (>0.7 for either).
   - Do NOT penalize just because there are no direct paths IF the RWR Rank is good.

3. **Low Confidence (0.01 - 0.45)**:
   - **RWR Rank > 300** AND Similarity < 0.7.
   - If Rank > 1000, score MUST be < 0.2.
   - If RWR Probability is exactly 0.0, score < 0.15.
   - Note: RWR Probability values can be numerically small (e.g., 1e-4). Do NOT downrank just because the number is small. TRUST THE RANK.

**KEY DIRECTIVE**: 
- If Rank is TOP 200 -> Give **0.8** or higher (HIGH CONFIDENCE).
- If Rank is decent (e.g., 200-500) -> Give **0.75** (Lean Positive - Trust the Rank).
- If Rank is poor (e.g., 600+) but Similarity is High (>0.7) -> Give **0.65** (Lean Positive).
- If Rank is poor (>800) and Similarity is Low/Moderate -> Give **0.1** (Strictly Negative).

**Consistence Check**:
If you say "Low Confidence", your score MUST be < 0.45.
If you say "High Confidence", your score MUST be > 0.75.
If you say "Moderate Confidence", your score typically falls 0.55 - 0.74.

=== INSTRUCTIONS ===
1. Analyze the RWR (Random Walk) scores.
2. Analyze the Similarity scores.
3. Synthesize a coherent reasoning chain.
4. Conclude with a prediction strength (Low/Moderate/High).

Output your reasoning primarily in natural language.

IMPORTANT: At the very end of your response, output the final predicted probability score (0 to 1) strictly in this JSON format: {{"score": 0.xx}}.
"""

# Template for Refinement
REFINE_GEN_TEMPLATE = """You are an expert Biologist.
You previously wrote a draft analysis for a miRNA-Disease association, but it was criticized by a reviewer.
Your task is to REWRITE the analysis to address the critique.

Target: {mirna_id} - {disease_id}

=== COMPUTATIONAL EVIDENCE ===
{feature_context}

=== PREVIOUS DRAFT ===
{previous_draft}

=== REVIEWER CRITIQUE ===
{critique_feedback}

=== SCORING RULES - CRITICAL ===
[CRITICAL SCORING CALIBRATION]
You must be DECISIVE. Avoid scores around 0.5 unless evidence is truly contradictory.
Use the following Evidence-to-Score Mapping:

1. **High Confidence (0.75 - 0.99)**:
   - **RWR Rank Top 100** (Top 5%). (e.g. Rank 1-100)
   - OR RWR Rank Top 300 AND (Disease Similarity > 0.7 OR miRNA Similarity > 0.7).
   - OR Any Direct Graph Path exists.

2. **Moderate Confidence (0.55 - 0.74)**:
   - **RWR Rank 101-300** with weak similarity.
   - OR **RWR Rank 301-800** BUT with **Strong Similarity** (>0.7 for either).
   - Do NOT penalize just because there are no direct paths IF the RWR Rank is good.

3. **Low Confidence (0.01 - 0.45)**:
   - **RWR Rank > 300** AND Similarity < 0.7.
   - If Rank > 1000, score MUST be < 0.2.
   - If RWR Probability is exactly 0.0, score < 0.15.
   - Note: RWR Probability values can be numerically small (e.g., 1e-4). Do NOT downrank just because the number is small. TRUST THE RANK.

**KEY DIRECTIVE**: 
- If Rank is TOP 200 -> Give **0.8** or higher (HIGH CONFIDENCE).
- If Rank is decent (e.g., 200-500) -> Give **0.75** (Lean Positive - Trust the Rank).
- If Rank is poor (e.g., 600+) but Similarity is High (>0.7) -> Give **0.65** (Lean Positive).
- If Rank is poor (>800) and Similarity is Low/Moderate -> Give **0.1** (Strictly Negative).

**Consistence Check**:
If you say "Low Confidence", your score MUST be < 0.45.
If you say "High Confidence", your score MUST be > 0.75.
If you say "Moderate Confidence", your score typically falls 0.55 - 0.74.

=== INSTRUCTIONS ===
1. Read the critique carefully.
2. Fix any hallucinations.
3. Resolve any contradictions.
4. Improve clarity.

IMPORTANT: At the very end of your response, output the final predicted probability score (0 to 1) strictly in this JSON format: {{"score": 0.xx}}.
"""

def fusion_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Generator/Refiner
    
    Generates the initial reasoning or refines it based on feedback.
    
    Args:
        state (AgentState): Current graph state.
        
    Returns:
        dict: Update to 'draft_reasoning' and potentially 'revision_count'
    """
    print(f"--- [Fusion Agent] Generating (Revision {state.get('revision_count', 0)}) ---")
    
    # Get config
    mirna_id = state["mirna_id"]
    disease_id = state["disease_id"]
    features = state["feature_data"]
    revision_count = state.get("revision_count", 0)
    
    # Format features for prompt
    feature_context = json.dumps(features, indent=2)
    
    llm = get_llm(temperature=0.4) # Slightly creative for writing, but grounded
    parser = StrOutputParser()
    
    if revision_count == 0:
        # Initial Generation
        prompt = ChatPromptTemplate.from_template(INITIAL_GEN_TEMPLATE)
        chain = prompt | llm | parser
        
        response = chain.invoke({
            "mirna_id": mirna_id,
            "disease_id": disease_id,
            "feature_context": feature_context
        })
    else:
        # Refinement
        critique = state.get("critique_feedback", "Please improve the text.")
        previous_draft = state.get("draft_reasoning", "")
        
        prompt = ChatPromptTemplate.from_template(REFINE_GEN_TEMPLATE)
        chain = prompt | llm | parser
        
        response = chain.invoke({
            "mirna_id": mirna_id,
            "disease_id": disease_id,
            "feature_context": feature_context,
            "previous_draft": previous_draft,
            "critique_feedback": critique
        })
        
    return {
        "draft_reasoning": response,
        # increment logic might be handled in the edge, but we can ensure it's tracked here if needed.
        # usually update the state with the new draft.
    }

