"""
Financial Data Tools

Price analysis tools (yfinance - free, unlimited)
Fundamental analysis tools (yfinance + Alpha Vantage)
"""

import os
import logging
from langchain_core.tools import tool
import requests

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    logger.error("yfinance not installed. Run: pip install yfinance")
    YF_AVAILABLE = False

#stock = yf.Ticker("NVDA")
# See ALL available fields
# print(stock.info.keys())

@tool
def get_stock_quote(ticker: str) -> str:
    """
    Get current stock price and key metrics.
    
    Returns: Current price, change %, volume, market cap, 52-week high/low
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")
    """
    if not YF_AVAILABLE:
        return "yfinance not installed"
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info   #library builds requests internally
        
        #library parses repsponse into python dict and returns html/json data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', 0)
        volume = info.get('volume', 0)
        market_cap = info.get('marketCap', 0)
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        low_52 = info.get('fiftyTwoWeekLow', 0)
        
        # Calculate metrics
        change = info.get('regularMarketChange', 0)
        change_pct = info.get('regularMarketChangePercent', 0) * 100
        
        # Distance from 52-week high/low
        distance_from_high = ((current_price - high_52) / high_52 * 100) if high_52 else 0
        distance_from_low = ((current_price - low_52) / low_52 * 100) if low_52 else 0
        
        return f"""Stock Quote - {ticker.upper()}:
• Current Price: ${current_price:.2f}
• Change Today: ${change:.2f} ({change_pct:+.2f}%)
• Previous Close: ${previous_close:.2f}
• Volume: {volume:,}
• Market Cap: ${market_cap:,}
• 52-Week High: ${high_52:.2f} (currently {distance_from_high:+.1f}% from high)
• 52-Week Low: ${low_52:.2f} (currently {distance_from_low:+.1f}% from low)"""
        
    except Exception as e:
        logger.error(f"Error fetching quote for {ticker}: {e}")
        return f"Could not fetch quote for {ticker}: {str(e)}"


@tool
def get_historical_prices(ticker: str, period: str = "1mo") -> str:
    """
    Get historical price data with max/min analysis.
    
    Automatically calculates:
    - Period high/low prices
    - Maximum drawdown from peak
    - Performance metrics
    
    Args:
        ticker: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    """
    if not YF_AVAILABLE:
        return "yfinance not installed"
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"No historical data for {ticker}"
        
        # Calculate metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        period_return = ((end_price / start_price) - 1) * 100
        
        period_high = hist['High'].max()
        period_low = hist['Low'].min()
        
        # Maximum drawdown calculation
        # Find peak-to-trough decline
        cumulative_returns = (1 + hist['Close'].pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        avg_volume = hist['Volume'].mean()
        
        return f"""Historical Data - {ticker.upper()} ({period}):
• Starting Price: ${start_price:.2f}
• Ending Price: ${end_price:.2f}
• Period Return: {period_return:+.2f}%
• Period High: ${period_high:.2f}
• Period Low: ${period_low:.2f}
• Maximum Drawdown: {max_drawdown:.2f}% (worst peak-to-trough decline)
• Average Volume: {avg_volume:,.0f}"""
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        return f"Could not fetch historical data for {ticker}: {str(e)}"


@tool
def get_stock_news(ticker: str) -> str:
    """Get recent stock news with sentiment (if available)."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    
    if api_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 5,
                "apikey": api_key
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("feed"):
                news_items = []
                for item in data["feed"][:5]:
                    title = item.get('title', 'No title')
                    source = item.get('source', 'Unknown')
                    sentiment = item.get('overall_sentiment_label', 'Neutral')
                    news_items.append(f"• {title}\n  Source: {source} | Sentiment: {sentiment}")
                
                return f"Recent News - {ticker.upper()}:\n\n" + "\n\n".join(news_items)
        except Exception as e:
            logger.warning(f"Alpha Vantage failed: {e}")
   

@tool
def get_company_overview(ticker: str) -> str:
    """Get company fundamentals and key metrics."""
    if not YF_AVAILABLE:
        return "yfinance not installed"
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        company_name = info.get('longName', 'N/A')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 'N/A')
        eps = info.get('trailingEps', 'N/A')
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        high_52 = info.get('fiftyTwoWeekHigh', 'N/A')
        low_52 = info.get('fiftyTwoWeekLow', 'N/A')
        avg_volume = info.get('averageVolume', 0)
        description = info.get('longBusinessSummary', 'N/A')
        
        if len(description) > 300:
            description = description[:300] + "..."
        
        return f"""Company Overview - {ticker.upper()}:
• Name: {company_name}
• Sector: {sector}
• Industry: {industry}
• Market Cap: ${market_cap:,}
• P/E Ratio: {pe_ratio}
• EPS: ${eps}
• Dividend Yield: {dividend_yield:.2f}%
• 52-Week Range: ${low_52} - ${high_52}
• Average Volume: {avg_volume:,}

Description:
{description}"""
        
    except Exception as e:
        logger.error(f"Error fetching company overview for {ticker}: {e}")
        return f"Could not fetch company overview for {ticker}: {str(e)}"




@tool
def get_financial_statements(ticker: str, quarters_back: int = 0) -> str:
    """
    Get quarterly financial statements (income & balance sheet).
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "NVDA")
        quarters_back: 0 = most recent, 1 = previous quarter, 2 = two quarters ago, etc.
    
    Returns: Revenue, profit, margins, balance sheet for the specified quarter
    """
    if not YF_AVAILABLE:
        return "yfinance not installed"
    
    try:
        stock = yf.Ticker(ticker)
        
        # Use QUARTERLY data, not annual
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        
        if income.empty or balance.empty:
            return f"No financial data for {ticker}"
        
        # Check if requested quarter exists
        if quarters_back >= len(income.columns):
            return f"Only {len(income.columns)} quarters of data available"
        
        # Get the specific quarter
        selected_income = income.iloc[:, quarters_back]
        selected_balance = balance.iloc[:, quarters_back]
        
        # Get the period date
        period_date = income.columns[quarters_back]
        period_str = period_date.strftime('%Y-%m-%d') if hasattr(period_date, 'strftime') else str(period_date)
        
        # Extract income statement data
        total_revenue = selected_income.get('Total Revenue', 0)
        gross_profit = selected_income.get('Gross Profit', 0)
        operating_income = selected_income.get('Operating Income', 0)
        net_income = selected_income.get('Net Income', 0)
        ebitda = selected_income.get('EBITDA', 0)
        
        # Extract balance sheet data
        total_assets = selected_balance.get('Total Assets', 0)
        total_liabilities = selected_balance.get('Total Liabilities Net Minority Interest', 0)
        equity = selected_balance.get('Stockholders Equity', 0)
        cash = selected_balance.get('Cash And Cash Equivalents', 0)
        
        # Calculate margins
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue else 0
        operating_margin = (operating_income / total_revenue * 100) if total_revenue else 0
        net_margin = (net_income / total_revenue * 100) if total_revenue else 0
        
        return f"""Quarterly Financial Statements - {ticker.upper()}
Period Ending: {period_str}

INCOME STATEMENT:
• Total Revenue: ${total_revenue:,.0f}
• Gross Profit: ${gross_profit:,.0f} ({gross_margin:.1f}% margin)
• Operating Income: ${operating_income:,.0f} ({operating_margin:.1f}% margin)
• Net Income: ${net_income:,.0f} ({net_margin:.1f}% margin)
• EBITDA: ${ebitda:,.0f}

BALANCE SHEET:
• Total Assets: ${total_assets:,.0f}
• Total Liabilities: ${total_liabilities:,.0f}
• Stockholders Equity: ${equity:,.0f}
• Cash: ${cash:,.0f}"""
        
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return f"Could not fetch financial statements for {ticker}: {str(e)}"


