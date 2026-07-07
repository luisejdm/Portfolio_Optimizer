import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import download_prices, calculate_returns
from optimizer import PortfolioOptimizer


def load_data(tickers, benchmark, start_date, end_date):
    """
    Load price and return data for given tickers and benchmark
    Args:
        tickers (list): List of stock ticker symbols.
        benchmark (str): Benchmark ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
    Returns:
        tuple: prices, benchmark_prices, returns, benchmark_returns
    """
    try:
        prices = download_prices(tickers, start_date, end_date)
        benchmark_prices = download_prices([benchmark], start_date, end_date)

        if prices.empty or benchmark_prices.empty:
            return None, None, None, None

        returns = calculate_returns(prices)
        benchmark_returns = calculate_returns(benchmark_prices)

        return prices, benchmark_prices, returns, benchmark_returns

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None


def optimize_portfolio(_returns, rf, _benchmark_returns):
    """
    Optimize portfolio weights using different strategies
    Args:
        _returns (pd.DataFrame): DataFrame of asset returns.
        rf (float): Risk-free rate.
        _benchmark_returns (pd.DataFrame): DataFrame of benchmark returns.
    Returns:
        dict: Dictionary of optimized weights for each strategy.
    """
    optimizer = PortfolioOptimizer(_returns, rf)

    weights = {
        'Minimum Variance': optimizer.min_variance(),
        'Maximum Sharpe': optimizer.max_sharpe(),
        'Minimum Target Semivariance': optimizer.min_target_semivariance(_benchmark_returns),
        'Maximum Target Omega': optimizer.max_target_omega(_benchmark_returns)
    }

    return weights


def calculate_portfolio_metrics(returns, weights, rf):
    """Calculate key portfolio performance metrics"""
    portfolio_return = np.dot(returns.mean(), weights) * 252
    portfolio_std = np.sqrt(weights.T @ returns.cov() @ weights * 252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std if portfolio_std > 0 else 0

    # Sortino Ratio
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_std = np.sqrt(weights.T @ downside_returns.cov() @ weights * 252)
    sortino_ratio = (portfolio_return - rf) / downside_std if downside_std > 0 else 0

    return {
        'Annual Return': portfolio_return,
        'Annual Volatility': portfolio_std,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio
    }


def plot_pie_chart(weights, tickers, title):
    """Create interactive pie chart for portfolio weights"""
    # Filter out negligible weights
    filtered_data = [(w, t) for w, t in zip(weights, tickers) if w > 0.0001]

    if not filtered_data:
        return None

    filtered_weights, filtered_tickers = zip(*filtered_data)

    fig = go.Figure(data=[go.Pie(
        labels=filtered_tickers,
        values=filtered_weights,
        hole=0.3,
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.4f}<extra></extra>',
        marker=dict(
            colors=px.colors.sequential.Blues,
            line=dict(color='white', width=2)
        )
    )])

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        showlegend=True,
        height=400,
        margin=dict(t=80, b=20, l=20, r=20)
    )

    return fig


def calculate_backtest_metrics(history, initial_capital):
    """Calculate comprehensive backtesting metrics"""
    metrics = {}

    for column in history.columns:
        if column == 'Benchmark':
            continue

        values = history[column]
        returns = values.pct_change().dropna()

        # Total Return
        total_return = (values.iloc[-1] - initial_capital) / initial_capital

        # Annualized Return
        annualized_return = returns.mean() * 252

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio (assuming risk-free rate from sidebar)
        sharpe = (annualized_return - st.session_state.get('rf', 0.035)) / volatility if volatility > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (annualized_return - st.session_state.get('rf', 0.035)) / downside_std if downside_std > 0 else 0

        # Calmar Ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Alpha and Beta

        metrics[column] = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe:.3f}",
            'Sortino Ratio': f"{sortino:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar:.3f}",
            'Final Value': f"${values.iloc[-1]:,.2f}"
        }

    return pd.DataFrame(metrics).T


def plot_backtesting_results(history):
    """Plot backtesting results over time"""
    backtest_fig = go.Figure()

    for column in history.columns:
        backtest_fig.add_trace(go.Scatter(
            x=history.index,
            y=history[column],
            mode='lines',
            name=column,
            line=dict(width=2)
        ))

    backtest_fig.update_layout(
        title="Backtesting Results",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode='x unified',
        height=500,
        showlegend=True
    )

    st.plotly_chart(backtest_fig, use_container_width=True)


def plot_rolling_sharpe(history, rf):
    """Plot rolling Sharpe ratio over time"""
    sharpe_fig = go.Figure()
    window = 252

    for column in history.columns:
        if column == 'Benchmark':
            continue

        values = history[column]
        returns_series = values.pct_change().dropna()

        if len(returns_series) >= window:
            rolling_mean = returns_series.rolling(window=window).mean() * 252
            rolling_std = returns_series.rolling(window=window).std() * np.sqrt(252)
            rolling_sharpe = (rolling_mean - rf) / rolling_std

            sharpe_fig.add_trace(go.Scatter(
                x=history.index[window:],
                y=rolling_sharpe[window - 1:],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))

    sharpe_fig.update_layout(
        title="Rolling Sharpe Ratio (252-Day Window)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    st.plotly_chart(sharpe_fig, use_container_width=True)


def plot_rolling_volatility(history):
    """Plot rolling volatility over time"""
    vol_fig = go.Figure()
    window = 252

    for column in history.columns:
        if column == 'Benchmark':
            continue

        values = history[column]
        returns_series = values.pct_change().dropna()

        if len(returns_series) >= window:
            rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(252)

            vol_fig.add_trace(go.Scatter(
                x=history.index[window:],
                y=rolling_vol[window - 1:],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))

    vol_fig.update_layout(
        title="Rolling Volatility (252-Day Window)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    st.plotly_chart(vol_fig, use_container_width=True)


def plot_rolling_return(history):
    """Plot rolling return over time"""
    ret_fig = go.Figure()
    window = 252

    for column in history.columns:
        if column == 'Benchmark':
            continue

        values = history[column]
        returns_series = values.pct_change().dropna()

        if len(returns_series) >= window:
            rolling_ret = returns_series.rolling(window=window).mean() * 252

            ret_fig.add_trace(go.Scatter(
                x=history.index[window:],
                y=rolling_ret[window - 1:],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))

    ret_fig.update_layout(
        title="Rolling Return (252-Day Window)",
        xaxis_title="Date",
        yaxis_title="Expected Return",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    st.plotly_chart(ret_fig, use_container_width=True)


def plot_rolling_sortino(history, rf):
    """Plot rolling Sortino ratio over time"""
    sortino_fig = go.Figure()
    window = 252

    for column in history.columns:
        if column == 'Benchmark':
            continue

        values = history[column]
        returns_series = values.pct_change().dropna()

        if len(returns_series) >= window:
            rolling_mean = returns_series.rolling(window=window).mean() * 252
            downside_returns = returns_series.copy()
            downside_returns[downside_returns > 0] = 0
            rolling_downside_std = downside_returns.rolling(window=window).std() * np.sqrt(252)
            rolling_sortino = (rolling_mean - rf) / rolling_downside_std

            sortino_fig.add_trace(go.Scatter(
                x=history.index[window:],
                y=rolling_sortino[window - 1:],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))

    sortino_fig.update_layout(
        title="Rolling Sortino Ratio (252-Day Window)",
        xaxis_title="Date",
        yaxis_title="Sortino Ratio",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    st.plotly_chart(sortino_fig, use_container_width=True)


def plot_returns_distribution(history):
    dist_fig = go.Figure()

    for column in history.columns:
        if column == 'Benchmark':
            continue

        values = history[column]
        returns_series = values.pct_change().dropna()

        dist_fig.add_trace(go.Histogram(
            x=returns_series,
            name=column,
            opacity=0.7,
            nbinsx=100
        ))

    dist_fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Daily Returns",
        yaxis_title="Frequency",
        barmode='overlay',
        height=600,
        showlegend=True
    )

    st.plotly_chart(dist_fig, use_container_width=True)


def plot_correlation_heatmap(returns):
    corr_matrix = returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Blues',
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation")
    ))

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        xaxis_title="Assets",
        yaxis_title="Assets",
        height=900,
        autosize=True,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left')
    )
    st.plotly_chart(fig, use_container_width=True)