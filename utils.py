import yfinance as yf
import pandas as pd

def download_prices(tickers: list[str], start_date: str, end_date: str):
    """
    Download historical closing prices for given tickers and date range.
    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
    Returns:
        pd.DataFrame: DataFrame containing closing prices for the tickers.
    """
    return yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'][tickers]

def calculate_returns(prices: pd.DataFrame):
    """
    Calculate daily returns from price data.
    Args:
        prices (pd.DataFrame): DataFrame of historical prices.
    Returns:
        pd.DataFrame: DataFrame of daily returns.
    """
    return prices.pct_change().dropna()