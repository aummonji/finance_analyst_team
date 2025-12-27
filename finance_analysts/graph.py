"""
Financial Analyst Team Graph

Simple orchestrator-driven design using fan-out pattern.
No Send API - just conditional edges and parallel execution.
"""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END, START

from finance_analysts.state import FinancialState
from finance_analysts.agents import (
    manage_conversation_length,
    orchestrator,
    price_analyst,
    fundamental_analyst,
    trader,
    reporter,
    synthesize_final_response
)
from finance_analysts.portfolio_advisor import portfolio_advisor

logger = logging.getLogger(__name__)


def create_graph():
    """
    Build financial analyst team graph.
    
    Flow: orchestrator → agents → orchestrator → agents → ... → done → synthesize
    """
    logger.info("🗓️ Building graph...")
    
    workflow = StateGraph(FinancialState)
    
    # Nodes
    workflow.add_node("manage_length", manage_conversation_length)
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("price", price_analyst)
    workflow.add_node("fundamental", fundamental_analyst)
    workflow.add_node("trader", trader)
    workflow.add_node("portfolio_advisor", portfolio_advisor)
    workflow.add_node("reporter", reporter)
    workflow.add_node("synthesize", synthesize_final_response)
    
    # Parallel execution helper (single fan-out node for price + fundamental)
    workflow.add_node("both_analysts", lambda s: s)
    
    # Entry
    workflow.add_edge(START, "manage_length")
    workflow.add_edge("manage_length", "orchestrator")
    
    # Orchestrator routing
    def route_from_orchestrator(state):
        """
        Route based on orchestrator's decision.
        
        Handles:
        - Single agent calls
        - Parallel execution for price + fundamental
        - Done state
        """
        agents = state.get("next_agents", [])
        
        # No agents = done
        if not agents:
            return "done"
        
        # Check for failed trading (don't retry)
        results = state.get("results", {})
        if "trader" in agents and results.get("trader", "").startswith("❌"):
            logger.warning("⚠️ Trading failed, skipping retry")
            return "done"
        
        # Single agent - route directly
        if len(agents) == 1:
            return agents[0]
        
        # Multiple agents - check for parallel patterns
        agents_set = set(agents)
        
        # Both price and fundamental requested (with or without reporter)
        # Route to parallel execution - reporter will be called in next iteration
        if {"price", "fundamental"}.issubset(agents_set):
            return "both_analysts"
        
        # Default: call first agent (orchestrator will loop for others)
        return agents[0]
    
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "price": "price",
            "fundamental": "fundamental",
            "trader": "trader",
            "portfolio_advisor": "portfolio_advisor",
            "reporter": "reporter",
            "both_analysts": "both_analysts",
            "done": "synthesize"
        }
    )
    
    # Fan-out edges for parallel execution
    workflow.add_edge("both_analysts", "price")
    workflow.add_edge("both_analysts", "fundamental")
    
    # All agents loop back to orchestrator
    workflow.add_edge("price", "orchestrator")
    workflow.add_edge("fundamental", "orchestrator")
    workflow.add_edge("trader", "orchestrator")
    workflow.add_edge("portfolio_advisor", "orchestrator")
    workflow.add_edge("reporter", "orchestrator")
    
    # End
    workflow.add_edge("synthesize", END)
    
    logger.info("✅ Graph built")
    
    return workflow.compile()