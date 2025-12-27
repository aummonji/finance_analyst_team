"""Alpaca paper trading integration.

This module provides tools for placing paper trades (simulated trades with fake money)
using the Alpaca brokerage API. All trades are paper-only - no real money involved.

Key features:
- Buy/sell stocks with share quantity or dollar amount
- View portfolio positions and P&L
- Track order history
- Full integration with LangChain tool system
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from typing import Dict
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Alpaca API - Try to import, gracefully handle if not installed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    # If alpaca-py not installed, tools will return helpful error messages
    logger.warning("alpaca-py not installed. Run: pip install alpaca-py")
    ALPACA_AVAILABLE = False


def get_alpaca_client():
    """Get Alpaca trading client configured for paper trading.
    
    Paper trading means:
    - Fake money (starts with $100,000)
    - Real market data and prices
    - Orders are simulated, not real
    - Perfect for testing strategies
    
    Returns:
        TradingClient: Authenticated client for paper trading
        
    Raises:
        ValueError: If API keys not configured in .env
    """
    # Load API credentials from environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET required for trading")
    
    # Create client with paper=True (CRITICAL - ensures fake money only!)
    return TradingClient(api_key, secret_key, paper=True)


@tool
def buy_stock(ticker: str, quantity: int = None, dollars: float = None) -> str:
    """Place a paper trade BUY order.
    
    This tool allows buying stocks in two ways:
    1. By share quantity: "buy 5 shares of AAPL" → quantity=5
    2. By dollar amount: "buy $500 of AAPL" → dollars=500.0
    
    The AI will parse the user's request and call this tool with appropriate args.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA", "MSFT")
        quantity: Number of shares to buy (e.g., 5, 10, 100)
        dollars: Dollar amount to spend (e.g., 500.0, 1000.0)
        
    Note: Provide EITHER quantity OR dollars, not both.
    
    Returns:
        Order confirmation with details (order ID, status, amount)
        
    Examples:
        User: "Buy 5 shares of NVDA"
        → AI calls: buy_stock(ticker="NVDA", quantity=5)
        
        User: "Buy $500 of AAPL"
        → AI calls: buy_stock(ticker="AAPL", dollars=500.0)
    """
    # Check if Alpaca library is installed
    if not ALPACA_AVAILABLE:
        return "Alpaca not installed. Run: pip install alpaca-py"
    
    try:
        # Get authenticated trading client
        client = get_alpaca_client()
        
        # Build order request based on what user specified
        if quantity:
            # Share-based order: "Buy 5 shares"
            order_data = MarketOrderRequest(
                symbol=ticker,              # Stock to buy
                qty=quantity,               # Number of shares
                side=OrderSide.BUY,         # BUY vs SELL
                time_in_force=TimeInForce.DAY  # Order expires end of day
            )
            order_type = f"{quantity} shares"
            
        elif dollars:
            # Dollar-based order: "Buy $500 worth"
            # Alpaca calls this "notional" (dollar amount instead of shares)
            order_data = MarketOrderRequest(
                symbol=ticker,
                notional=dollars,           # Dollar amount to spend
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order_type = f"${dollars}"
            
        else:
            # User didn't specify quantity or dollars - invalid
            return "Must specify either quantity (shares) or dollars amount"
        
        # Submit order to Alpaca (this happens instantly in paper trading)
        order = client.submit_order(order_data)
        
        # Log for debugging
        logger.info(f"Paper trade submitted: BUY {order_type} of {ticker}")
        
        # Return user-friendly confirmation
        return f"""✅ PAPER TRADE ORDER PLACED:
Type: BUY
Ticker: {ticker}
Amount: {order_type}
Order ID: {order.id}
Status: {order.status}

Note: This is a PAPER TRADE (not real money)"""
        
    except Exception as e:
        # Handle errors (invalid ticker, insufficient funds, API issues, etc.)
        logger.error(f"Error placing order: {e}")
        return f"❌ Order failed: {e}"


@tool
def sell_stock(ticker: str, quantity: int = None, dollars: float = None) -> str:
    """Place a paper trade SELL order.
    
    This tool allows selling stocks you own in two ways:
    1. By share quantity: "sell 5 shares of AAPL" → quantity=5
    2. By dollar amount: "sell $500 of AAPL" → dollars=500.0
    
    Note: You can only sell stocks you currently own in your portfolio.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA", "MSFT")
        quantity: Number of shares to sell (e.g., 5, 10, 100)
        dollars: Dollar amount to sell (e.g., 500.0, 1000.0)
        
    Note: Provide EITHER quantity OR dollars, not both.
    
    Returns:
        Order confirmation with details (order ID, status, amount)
        
    Examples:
        User: "Sell 5 shares of NVDA"
        → AI calls: sell_stock(ticker="NVDA", quantity=5)
        
        User: "Sell $500 of MSFT"
        → AI calls: sell_stock(ticker="MSFT", dollars=500.0)
    """
    if not ALPACA_AVAILABLE:
        return "Alpaca not installed. Run: pip install alpaca-py"
    
    try:
        client = get_alpaca_client()
        
        # Build order request - same structure as buy, but OrderSide.SELL
        if quantity:
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.SELL,        # SELL instead of BUY
                time_in_force=TimeInForce.DAY
            )
            order_type = f"{quantity} shares"
            
        elif dollars:
            order_data = MarketOrderRequest(
                symbol=ticker,
                notional=dollars,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order_type = f"${dollars}"
            
        else:
            return "Must specify either quantity (shares) or dollars amount"
        
        # Submit sell order
        order = client.submit_order(order_data)
        
        logger.info(f"Paper trade submitted: SELL {order_type} of {ticker}")
        
        return f"""✅ PAPER TRADE ORDER PLACED:
Type: SELL
Ticker: {ticker}
Amount: {order_type}
Order ID: {order.id}
Status: {order.status}

Note: This is a PAPER TRADE (not real money)"""
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return f"❌ Order failed: {e}"


@tool
def get_portfolio() -> str:
    """Get current paper trading portfolio with positions and buying power.
    
    Shows:
    - Total account value (cash + positions)
    - Available cash
    - Buying power (can be 2x cash if margin enabled)
    - All open positions with P&L (profit/loss)
    
    This is like viewing your brokerage account balance and holdings.
    
    Returns:
        Portfolio summary with:
        - Account value
        - Cash balance
        - Buying power
        - Each position with:
          - Shares owned
          - Current price
          - Market value
          - Profit/Loss ($ and %)
          - Cost basis (what you paid)
          
    Example:
        User: "Show my portfolio"
        User: "What stocks do I own?"
        User: "How much money do I have?"
        → All examples that trigger this tool
    """
    if not ALPACA_AVAILABLE:
        return "Alpaca not installed. Run: pip install alpaca-py"
    
    try:
        client = get_alpaca_client()
        
        # Get account information (balances, buying power, etc.)
        account = client.get_account()
        
        # Get all open positions (stocks you currently own)
        positions = client.get_all_positions()
        
        # Build portfolio summary - start with account totals
        portfolio_text = f"""📊 PAPER TRADING PORTFOLIO:

💰 Account Value: ${float(account.equity):,.2f}
💵 Cash: ${float(account.cash):,.2f}
📈 Buying Power: ${float(account.buying_power):,.2f}

Positions ({len(positions)}):
"""
        
        # List each position with details
        if positions:
            for pos in positions:
                # Calculate P&L (profit/loss)
                pnl = float(pos.unrealized_pl)           # Dollar P&L
                pnl_pct = float(pos.unrealized_plpc) * 100  # Percentage P&L
                
                # Choose emoji based on profit or loss
                pnl_emoji = "📈" if pnl >= 0 else "📉"
                
                # Format position details
                portfolio_text += f"""
{pnl_emoji} {pos.symbol}:
  Shares: {pos.qty}
  Current Price: ${float(pos.current_price):,.2f}
  Market Value: ${float(pos.market_value):,.2f}
  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)
  Cost Basis: ${float(pos.cost_basis):,.2f}
"""
        else:
            # No positions - all cash
            portfolio_text += "\n(No positions)\n"
        
        portfolio_text += "\nNote: This is PAPER TRADING (not real money)"
        
        return portfolio_text
        
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return f"❌ Could not fetch portfolio: {e}"


@tool
def get_orders() -> str:
    """Get recent paper trading orders.
    
    Shows order history with status:
    - ✅ Filled: Order executed successfully
    - ⏳ Pending: Order submitted, waiting to fill
    - ❌ Canceled: Order was canceled
    
    Useful for tracking what trades were made and their status.
    
    Returns:
        List of recent orders (up to 10) with:
        - Order type (BUY/SELL)
        - Ticker symbol
        - Quantity or dollar amount
        - Status (filled, pending, canceled)
        - Submission time
        - Order ID
        
    Example:
        User: "Show my recent orders"
        User: "What trades did I make?"
        User: "Did my AAPL order fill?"
        → All trigger this tool
    """
    if not ALPACA_AVAILABLE:
        return "Alpaca not installed. Run: pip install alpaca-py"
    
    try:
        client = get_alpaca_client()
        
        # Get recent orders (limit to 10 most recent)
        orders = list(client.get_orders())[:10]
        
        # Handle no orders case
        if not orders:
            return "📋 No recent orders"
        
        orders_text = "📋 RECENT PAPER TRADE ORDERS:\n\n"
        
        # List each order with details
        for order in orders:
            # Map status to emoji for visual clarity
            status_emoji = {
                "filled": "✅",      # Successfully executed
                "pending": "⏳",     # Waiting to execute
                "canceled": "❌"    # Was canceled
            }.get(order.status.lower(), "⚪")  # Default: white circle
            
            # Format order details
            # order.side.value converts enum to string ("buy" or "sell")
            orders_text += f"""{status_emoji} {order.side.value} {order.symbol}
  Qty: {order.qty if order.qty else f'${order.notional}'}
  Status: {order.status}
  Time: {order.submitted_at}
  Order ID: {order.id}

"""
        
        return orders_text
        
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        return f"❌ Could not fetch orders: {e}"