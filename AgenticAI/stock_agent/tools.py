"""
Stock Price Tools
Contains all tool definitions for fetching stock data.
"""

import json
from typing import Dict, Any
import yfinance as yf
from langchain_core.tools import tool


@tool
def get_stock_price(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Fetch historical stock price data for a given symbol and date range.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        JSON string with stock price data including Open, High, Low, Close, Volume
    """
    try:
        # Download stock data
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return json.dumps({
                "error": f"No data found for symbol '{symbol}' in the given date range",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            })
        
        # Convert DataFrame to dict and format the response
        result = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "data": []
        }
        
        for date, row in df.iterrows():
            result["data"].append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume'])
            })
        
        # Add summary statistics
        result["summary"] = {
            "total_days": len(df),
            "avg_close": round(float(df['Close'].mean()), 2),
            "max_high": round(float(df['High'].max()), 2),
            "min_low": round(float(df['Low'].min()), 2),
            "total_volume": int(df['Volume'].sum())
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        })


# List of all available tools
ALL_TOOLS = [get_stock_price]
