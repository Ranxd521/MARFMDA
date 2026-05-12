"""
LangGraph Workflow Definition
Assembles the Generator (Fusion) and Critic agents into a stateful graph.
Implements the Critic-Refine loop with a maximum retry limit.
"""
import sys
import os
import re
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState
from agents.fusion_agent import fusion_node
from agents.critic_agent import critic_node

def increment_revision(state: AgentState) -> dict:
    """Helper node to increment the revision counter."""
    return {"revision_count": state.get("revision_count", 0) + 1}

def should_continue(state: AgentState) -> Literal["end", "retry"]:
    """
    Conditional logic to determine if the workflow should stop or refine.
    """
    passed = state.get("critique_passed", False)
    count = state.get("revision_count", 0)
    reasoning = state.get("draft_reasoning", "")
    
    # Smart Critic Optimization: Skip critique for extreme scores in the first round
    if count == 0:
        score = 0.5
        try:
            # Extract score using regex (same logic as batch_predict)
            match = re.search(r'\{\s*"score"\s*:\s*(\d+\.?\d*)\s*\}', reasoning)
            if match:
                score = float(match.group(1))
            else:
                match_loose = re.search(r'score.*?(\d+\.?\d*)', reasoning, re.IGNORECASE)
                if match_loose:
                    score = float(match_loose.group(1))
            
            # Extreme score check: > 0.85 (confident positive) or < 0.15 (confident negative)
            if score > 0.85 or score < 0.15:
                print(f">>> SMART CRITIC: Score {score:.4f} is extreme/confident. Skipping critique (Save Cost).")
                return "end"
        except Exception:
            pass # Fallback to normal flow if extraction fails

    # Condition 1: Critique passed -> Success
    if passed:
        print(">>> CRITIQUE PASSED. Finishing.")
        return "end"
    
    # Condition 2: Max retries exceeded (Optimized to 1 retry)
    if count >= 1:
        print(f">>> MAX RETRIES ({count}) REACHED. Finishing with best effort.")
        return "end"
    
    # Condition 3: Continue to refinement
    print(f">>> CRITIQUE FAILED. Retrying (Count: {count})...")
    return "retry"

def create_workflow():
    """Compiles the LangGraph workflow."""
    
    # Initialize Graph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("generator", fusion_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("update_count", increment_revision)
    
    # Define Flow
    # 1. Start at Generator
    workflow.set_entry_point("generator")
    
    # 2. Generator -> Critic
    workflow.add_edge("generator", "critic")
    
    # 3. Critic -> Conditional Check
    # If "retry", go to update_count, then back to generator
    # If "end", go to END
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "retry": "update_count"
        }
    )
    
    # 4. update_count -> Generator (Loop back)
    workflow.add_edge("update_count", "generator")
    
    # Compile
    # using MemorySaver to support state persistence if needed, though for batch it might be transient
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

if __name__ == "__main__":
    # Test visualization or compilation
    app = create_workflow()
    print("Workflow Graph compiled successfully.")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("Could not draw graph (requires dependencies).")
