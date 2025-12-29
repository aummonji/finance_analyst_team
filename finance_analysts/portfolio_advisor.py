
""""
Agent that analyzes user's portfolio and gives investment advice using RAG + data tools.
Outputs SELL:/BUY: lines when user wants to execute trades.
"""

from typing import Dict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from .state import FinancialState
from .investment_knowledge import get_investment_advice
from .trading import get_portfolio
from .tools import get_company_overview, get_stock_quote
import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


def portfolio_advisor(state: FinancialState) -> Dict:
    """Smart portfolio advisor using investment principles + data analysis."""
    
    logger.info("💼 Portfolio advisor analyzing...")
    
    #building context from recent messages
    messages = state.get("messages", [])
    context = "\n".join([
        m.content[:300] for m in messages[-10:]
        if hasattr(m, 'content') and m.content
    ])
    
    tools = [
        get_investment_advice,
        get_portfolio,
        get_company_overview,
        get_stock_quote
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    agent_messages = [
    SystemMessage(content="""You are a portfolio advisor. When asked to review or analyze a portfolio, you MUST call get_portfolio() FIRST.

**CRITICAL: ALWAYS call get_portfolio() as your first action when the user asks about their portfolio.**

DO NOT respond conversationally saying you don't have access. You DO have access via the get_portfolio() tool.

**YOUR PROCESS:**

1. **IMMEDIATELY call get_portfolio()** - Don't ask the user for info, just call the tool
2. **Understand the numbers correctly**:
   - Gross positions = total value of all holdings
   - Net portfolio value = gross positions + cash (cash can be negative if on margin)
   - Calculate percentages based on GROSS POSITIONS, not net value
   - If cash is negative, user is leveraged (borrowed money to invest)
3. **Understand user from conversation**:
   - Risk tolerance (any phrasing)
   - Time horizon 
   - Goals (retirement, growth, income)
4. **Get investment principles**: Call get_investment_advice() with relevant query
5. **Analyze with data**: Compare holdings, check performance
6. **Rate & Advise** (1-10 scale): Give specific recommendations

**IMPORTANT - MARGIN/LEVERAGE:**
- If cash is negative (e.g., -$100K), user has borrowed that amount
- Report ALL holdings with their values and percentage of GROSS positions
- Example: $100K NVDA + $100K SPY = $200K gross, each is 50% (not 100%!)

**EXAMPLE FLOW:**

User: "Review my portfolio"
You: [IMMEDIATELY call get_portfolio()]
[Get results showing AAPL, MSFT, etc.]
You: [Call get_investment_advice("diversification tech stocks")]
You: [Analyze and respond in depth with rating + recommendations]

**WHEN USER WANTS TO EXECUTE/REBALANCE:**

If user says something like "rebalance", "do it", "execute", "make those changes", "rebuild it", "go ahead", 
"make it happen", "rebalance my portfolio", "rebalance accordingly", "yes":

YOU MUST output trade instructions at the END of your response in this EXACT format:

SELL: NVDA 551
SELL: SPY 155
BUY: VTI 200
BUY: BND 50

CRITICAL FORMAT RULES:
- Start each line with SELL: or BUY: (with colon!)
- Then ticker symbol
- Then quantity (whole number of SHARES, not dollars)
- One trade per line
- NO bullet points, NO asterisks, NO dollar amounts
- SELL orders before BUY orders
- These lines trigger automatic execution - without them, no trades happen!

**REMEMBER:**
- ALWAYS call get_portfolio() first - don't ask user for data
- Tech overweight (30-40%) is fine
- Crypto 1-5% for aggressive only
- Gold 5-10% for stability
- Be specific with tickers and amounts"""),
    
    HumanMessage(content=f"Context:\n{context}\n\nAnalyze and respond. If user wants to rebalance/execute, include SELL:/BUY: lines at the end.")
]
    
    # Agent loop - ReAct pattern
    for _ in range(8):
        response = llm_with_tools.invoke(agent_messages)
        agent_messages.append(response)
        
        if not response.tool_calls:
            break
        
        for tool_call in response.tool_calls:
            name = tool_call["name"]

        #tool call structure:
        # tool_call = {
        #     "name": "get_portfolio",
        #     "args": {},  # Arguments (empty for get_portfolio)
        #     "id": "call_123"  # Unique ID to link result back

            if name == "get_investment_advice":
                result = get_investment_advice.invoke(tool_call["args"])
            elif name == "get_portfolio":
                result = get_portfolio.invoke(tool_call["args"])
            elif name == "get_company_overview":
                result = get_company_overview.invoke(tool_call["args"])
            elif name == "get_stock_quote":
                result = get_stock_quote.invoke(tool_call["args"])
            else:
                result = "Unknown tool"
            
            # Match tool name to function
            # Call function with provided arguments
            # Get result (string)

            agent_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"]) #- Claude may call multiple tools
#                                                                               - ID links each result to correct tool call
            )
    
    final = agent_messages[-1].content if isinstance(agent_messages[-1], AIMessage) else "I can help!"
    
    return {"results": {"portfolio_advisor": final}}