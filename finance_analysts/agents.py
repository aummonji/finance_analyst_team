"""
Financial Analyst Team - Clean Implementation

Agents:
- Orchestrator: Reads user query + current results, decides what to call next
- Price Analyst: Gets price data
- Fundamental Analyst: Gets company/news data
- Trader: Executes trades
- Portfolio Advisor: Investment advice
- Reporter: Synthesizes comprehensive reports
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import logging
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage, ToolMessage
from .portfolio_advisor import portfolio_advisor
from .state import FinancialState
from .tools import (
    get_stock_quote,
    get_historical_prices,
    get_stock_news,
    get_company_overview,
    get_financial_statements,
)
from finance_analysts.trading import (
    buy_stock,
    sell_stock,
    get_portfolio,
    get_orders
)

logger = logging.getLogger(__name__)
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


def manage_conversation_length(state: FinancialState) -> dict:
    """
    Create rolling summary after 10 messages to manage tokens.
    
    Keeps recent context while summarizing older messages.
    This prevents token bloat in long conversations.
    """
    messages = state.get("messages", [])
    
    # If less than 10 messages, no action needed
    if len(messages) < 10:
        return {}
    
    logger.info(f"📝 Creating rolling summary ({len(messages)} messages)...")
    
    # Summarize everything except the last 2 messages
    messages_to_summarize = messages[:-2]
    recent_messages = messages[-2:]
    
    # Ask LLM to create summary
    summary_prompt = f"""Summarize this conversation, preserving key context:
- Any tickers/stocks mentioned
- Key facts or data discussed
- User's goals or questions

Keep it concise (3-4 sentences)."""
    
    summary_response = llm.invoke([
        HumanMessage(content=summary_prompt),
        *messages_to_summarize
    ])
    
    # Create summary message
    summary_message = SystemMessage(content=f"Previous conversation summary: {summary_response.content}")
    
    # Remove old messages and add summary
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    logger.info("  Summary created, old messages removed")
    
    return {
        "messages": [summary_message] + messages_to_remove
    }


def orchestrator(state: FinancialState) -> dict:
    """
    Orchestrator: Decides which agent(s) to call next.
    
    Uses LLM to intelligently determine when work is complete.
    """
    logger.info("🎯 Orchestrator analyzing request...")
    
    messages = state.get("messages", [])
    results = state.get("results", {})
    
    # Build context about completed work
    completed = list(results.keys())
    completed_str = ", ".join(completed) if completed else "nothing yet"
    
    # Get original query
    original_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
    
    # Let LLM decide intelligently
    prompt = f"""You are an intelligent orchestrator. Decide if more work is needed.

ORIGINAL QUERY: "{original_query}"
COMPLETED WORK: {completed_str}

YOUR JOB: Determine if the COMPLETED work answers the ORIGINAL query, or if more agents needed.

AGENTS AVAILABLE:
- price: stock prices, historical prices, volatility, volume
- fundamental: news/company data, earnings, financial statements, industry info, competitors  
- trader: execute trades, gets portfolio info, gets account orders
- portfolio_advisor: analyze portfolio, investment advice, asset allocation recommendations
- reporter: create comprehensive formal reports

LOGIC:
- If query asks for PRICE and "price" completed → "done" (answered!)
- If query asks for NEWS and "fundamental" completed → "done" (answered!)
- If query asks to TRADE and "trader" completed → "done" (executed!)
- If query wants PORTFOLIO ADVICE and "portfolio_advisor" completed → "done" (advised!)
- If query wants COMPARISON and "price,fundamental" done but NOT "reporter" → "reporter"
- If query wants COMPARISON and all 3 done → "done"
- If nothing done yet, decide which agents needed for the query

PORTFOLIO ADVISOR triggers:
- "rate my portfolio", "analyze my portfolio", "review my holdings"
- "what should I invest in", "investment advice", "asset allocation"
- mentions risk tolerance, retirement, diversification
- "should I add crypto/bonds/gold"

EXAMPLES:
Query: "what's AAPL price?" | Done: "" → "price"
Query: "what's AAPL price?" | Done: "price" → "done"
Query: "compare MSFT vs AAPL" | Done: "" → "price,fundamental"  
Query: "compare MSFT vs AAPL" | Done: "price,fundamental" → "reporter"
Query: "compare MSFT vs AAPL" | Done: "price,fundamental,reporter" → "done"
Query: "get current portfolio" | Done: "" → "trader"
Query: "get current portfolio" | Done: "trader" → "done"
Query: "rate my portfolio" | Done: "" → "portfolio_advisor"
Query: "rate my portfolio" | Done: "portfolio_advisor" → "done"
Query: "I'm 30 and aggressive, what should I invest in?" | Done: "" → "portfolio_advisor"
Query: "should I add crypto?" | Done: "" → "portfolio_advisor"
Query: "analyze my holdings and give advice" | Done: "" → "portfolio_advisor"

CRITICAL: If work answers the query → return "done". Don't repeat agents!

Respond ONLY with agent names or "done":"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()
    
    logger.info(f"  LLM raw decision: '{decision}'")
    
    # Extract agent names
    agent_names = []
    for word in decision.replace(",", " ").split():
        word = word.strip()
        if word in ["price", "fundamental", "trader", "portfolio_advisor", "reporter"]:
            agent_names.append(word)
    
    if "done" in decision or not agent_names:
        logger.info("  Final decision: DONE")
        return {"next_agents": []}
    
    logger.info(f"  Final decision: {agent_names}")
    return {"next_agents": agent_names}


def price_analyst(state: FinancialState) -> dict:
    """Price Analyst: Fetches and analyzes price data."""
    logger.info("📊 Price analyst working...")
    
    messages = state.get("messages", [])
    context = "\n".join([m.content for m in messages[-5:] if hasattr(m, 'content')])
    
    tools = [get_stock_quote, get_historical_prices]
    llm_with_tools = llm.bind_tools(tools)
    
    agent_messages = [
        SystemMessage(content="You are a price analyst. Extract tickers from context and get recent price data. " \
        "If user asks for current price use get_stock_quote to fetch exact real time price and also get_historical_prices for more context. If user asks " \
        "for all time high or low or max drawdown use get_historical_prices. Also use relevant tools for questions about volume, " \
        "volatility, price change etc."),
        HumanMessage(content=f"Context:\n{context}\n\nGet relevant price data.")
    ]
    
    for _ in range(3):
        response = llm_with_tools.invoke(agent_messages)
        agent_messages.append(response)
        
        if not response.tool_calls:
            break
        
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_stock_quote":
                result = get_stock_quote.invoke(tool_call["args"])
            else:
                result = get_historical_prices.invoke(tool_call["args"])
            
            agent_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
    
    analysis = agent_messages[-1].content if isinstance(agent_messages[-1], AIMessage) else "Price analysis complete"
    
    logger.info(f"  ✓ Price analysis: {len(analysis)} chars")
    
    return {"results": {"price": analysis}}


def fundamental_analyst(state: FinancialState) -> dict:
    """Fundamental Analyst: Fetches and analyzes company data."""
    logger.info("📰 Fundamental analyst working...")
    
    messages = state.get("messages", [])
    context = "\n".join([m.content for m in messages[-5:] if hasattr(m, 'content')])
    
    tools = [get_stock_news, get_company_overview, get_financial_statements]
    llm_with_tools = llm.bind_tools(tools)
    
    agent_messages = [
        SystemMessage(content="""You are a fundamental analyst. Extract tickers from context and get recent company data. Available tools:

get_company_overview - Company fundamentals, market cap, sector, industry, description
get_financial_statements - Income, balance sheet, cash flow, earnings 
get_stock_news - Recent news with sentiment


Be intelligent with tool usage based on the question."""),
        HumanMessage(content=f"Context:\n{context}\n\nGet relevant fundamental data.")
    ]
    
    for _ in range(4):
        response = llm_with_tools.invoke(agent_messages)
        agent_messages.append(response)
        
        if not response.tool_calls:
            break
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name == "get_stock_news":
                result = get_stock_news.invoke(tool_call["args"])
            elif tool_name == "get_company_overview":
                result = get_company_overview.invoke(tool_call["args"])
            elif tool_name == "get_financial_statements":
                result = get_financial_statements.invoke(tool_call["args"])
            else:
                result = get_earnings_data.invoke(tool_call["args"])
            
            agent_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
    
    analysis = agent_messages[-1].content if isinstance(agent_messages[-1], AIMessage) else "Fundamental analysis complete"
    
    logger.info(f"  ✓ Fundamental analysis: {len(analysis)} chars")
    
    return {"results": {"fundamental": analysis}}


def trader(state: FinancialState) -> dict:
    """Trader: Executes trades and manages portfolio."""
    logger.info("💼 Trader working...")
    
    messages = state.get("messages", [])
    
    tools = [buy_stock, sell_stock, get_portfolio, get_orders]
    llm_with_tools = llm.bind_tools(tools)
    
    agent_messages = [
        SystemMessage(content="""You are a trading agent. Execute buy/sell orders based on conversation.

If user says "buy that" or "buy it", look at previous messages for which stock.
This is PAPER TRADING (simulated).

For portfolio/status requests, use get_portfolio.
For recent orders, use get_orders.""")
    ] + messages
    
    for _ in range(2):
        response = llm_with_tools.invoke(agent_messages)
        agent_messages.append(response)
        
        if not response.tool_calls:
            break
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name == "buy_stock":
                result = buy_stock.invoke(tool_call["args"])
            elif tool_name == "sell_stock":
                result = sell_stock.invoke(tool_call["args"])
            elif tool_name == "get_portfolio":
                result = get_portfolio.invoke(tool_call["args"])
            else:
                result = get_orders.invoke(tool_call["args"])
            
            agent_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
    
    summary = agent_messages[-1].content if isinstance(agent_messages[-1], AIMessage) else "Trading complete"
    
    return {"results": {"trader": summary}}


def reporter(state: FinancialState) -> dict:
    """Reporter: Creates comprehensive formal reports."""
    logger.info("📋 Reporter creating formal report...")
    
    messages = state.get("messages", [])
    results = state.get("results", {})
    
    # Get user query
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    # Get conversation context
    context = "\n\n".join([
        m.content[:300] 
        for m in messages[-5:] 
        if hasattr(m, 'content') and m.content
    ])
    
    # Gather all analysis
    analysis_parts = []
    if "price" in results:
        analysis_parts.append(f"PRICE ANALYSIS:\n{results['price']}")
    if "fundamental" in results:
        analysis_parts.append(f"FUNDAMENTAL ANALYSIS:\n{results['fundamental']}")
    if "trader" in results:
        analysis_parts.append(f"TRADING ACTIVITY:\n{results['trader']}")
    
    combined = "\n\n".join(analysis_parts)
    
    prompt = f"""You are a professional financial analyst creating a FORMAL INVESTMENT REPORT.

RECENT CONVERSATION:
{context}

CURRENT REQUEST: "{user_query}"

DATA YOU HAVE:
{combined}

Create a PROFESSIONAL, FORMAL INVESTMENT REPORT with this structure:

------------------------------------------------
# [COMPANY NAME] - Investment Analysis Report
------------------------------------------------

**Date:** December 24, 2024
**Prepared by:** Financial Analysis Team

## Executive Summary
[High-level overview with key recommendation]

## 1. Current Market Position
### Stock Performance
- Current price and movement
- Volume analysis
- 52-week performance

### Market Valuation
- Market capitalization
- P/E ratio analysis

## 2. Financial Analysis
### Profitability Metrics
- Revenue, margins, EPS (with actual numbers from data)

### Balance Sheet Strength
- Assets, equity, debt, cash

## 3. Competitive Position
[Strengths, market position]

## 4. Investment Recommendation
**Rating:** [Strong Buy / Buy / Hold / Sell]
**Rationale:** [Clear reasoning based on data]

---

CRITICAL: Use ALL the actual data provided. Include specific numbers throughout."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"results": {"reporter": response.content}}


def synthesize_final_response(state: FinancialState) -> dict:
    """
    Create final response to user.
    
    Uses reporter output if available (formal report),
    otherwise creates conversational response.
    """
    logger.info("✨ Synthesizing final response...")
    
    messages = state.get("messages", [])
    results = state.get("results", {})
    
    # Get user query
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    # If reporter was called, use that (formal report)
    if "reporter" in results:
        return {"messages": [AIMessage(content=results["reporter"])]}
    
    # Otherwise, create conversational response
    parts = []
    if "price" in results:
        parts.append(results["price"])
    if "fundamental" in results:
        parts.append(results["fundamental"])
    if "trader" in results:
        parts.append(results["trader"])
    if "portfolio_advisor" in results:  # ADD THIS
        parts.append(results["portfolio_advisor"])
    
    if not parts:
        return {"messages": [AIMessage(content="I need more information to help you.")]}
    
    # Get conversation context
    context = "\n\n".join([
        m.content[:150] 
        for m in messages[-5:] 
        if hasattr(m, 'content') and m.content
    ])
    
    # Combine all results
    combined = "\n\n".join(parts)
    
    # Create conversational response
    prompt = f"""You are a financial analyst chatting with a client.

RECENT CONVERSATION:
{context}

CURRENT QUESTION: "{user_query}"

DATA AVAILABLE:
{combined}

CRITICAL: You MUST use the actual data provided above.
Include specific numbers: prices, P/E ratios, percentages, dates.

Respond conversationally:
- Use natural language (not formal report style)
- Be detailed, in-depth and specific
- Use the ACTUAL data from analysts
- Answer directly and helpfully

Response:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [AIMessage(content=response.content)]}