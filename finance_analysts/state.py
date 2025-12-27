"""
State Schema for Financial Analyst Team

Clean design:
- Messages with rolling summary for token management
- Results dict where agents write their outputs
- Orchestrator reads results to decide what to call next
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage, SystemMessage, RemoveMessage
from langgraph.graph.message import add_messages


class FinancialState(TypedDict):
    """
    State for financial analyst team.
    
    Fields:
    -------
    messages : List[AnyMessage]
        Conversation history. After 10 messages, creates rolling summary
        to manage token usage while preserving context.
    
    results : Dict[str, str]
        Outputs from each agent. Keys are agent names:
        - "price": Price analyst output
        - "fundamental": Fundamental analyst output  
        - "trader": Trader output
        - "reporter": Reporter synthesis
        
        Orchestrator reads this to see what's been done and decide what's next.
    
    next_agents : List[str]
        List of agent names to call next. Set by orchestrator.
        Examples: ["price"], ["price", "fundamental"], []
    """
    
    # Conversation with automatic message appending
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Agent outputs - reducer merges parallel updates
    results: Annotated[Dict[str, str], lambda x, y: {**x, **y}]
    
    # Next agents to call
    next_agents: List[str]