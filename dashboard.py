import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dashboard_utils import *
from backtesting import DynamicBacktester

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
    h1 {
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Ticker Selection
st.sidebar.subheader("Assets")
default_tickers = ['AAPL', 'MSFT', 'GOOG', 'JPM']
tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated)",
    value=", ".join(default_tickers),
    help="Enter stock tickers separated by commas"
)
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

# Benchmark Selection
benchmark = st.sidebar.text_input(
    "Benchmark Ticker",
    value="^GSPC",
    help="S&P 500 index (^GSPC)"
)

# Date Range
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2020, 1, 1),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Risk-Free Rate
st.sidebar.subheader("Parameters")
rf = st.sidebar.number_input(
    "Risk-Free Rate (annual)",
    min_value=0.0,
    max_value=0.20,
    value=0.035,
    step=0.001,
    format="%.3f",
    help="Annual risk-free rate (e.g., 0.035 for 3.5%)"
)

# Store in session state
st.session_state['rf'] = rf

# Backtesting Parameters
st.sidebar.subheader("Backtesting")
rebalance_months = st.sidebar.slider(
    "Rebalancing Period (months)",
    min_value=1,
    max_value=12,
    value=12,
    help="How often to reoptimize the portfolio"
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=100_000_000,
    value=1_000_000,
    step=100_000,
    format="%d"
)

# Run button
run_analysis = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# Main content
st.title("Portfolio Optimization Dashboard")
st.markdown("---")

if run_analysis:
    if len(tickers) < 2:
        st.error("Please select at least 2 tickers for portfolio optimization.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        with st.spinner("Loading market data..."):
            prices, benchmark_prices, returns, benchmark_returns = load_data(
                tickers, benchmark, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            )

        if prices is None or returns is None:
            st.error("Failed to load data. Please check your tickers and date range.")
        else:
            st.success(f"Loaded data for {len(tickers)} assets from {start_date} to {end_date}")

            # Create tabs
            tab1, tab2 = st.tabs([
                "Portfolio Optimization",
                "Backtesting Results"
            ])

            # Tab 1: Portfolio Optimization
            with tab1:
                st.header("Optimal Portfolio Weights")
                st.info(f"Optimization using price data from {start_date} to {end_date}.")

                with st.spinner("Optimizing portfolios..."):
                    weights = optimize_portfolio(returns, rf, benchmark_returns)

                # Create 2x2 grid for pie charts
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Minimum Variance")
                    fig = plot_pie_chart(weights['Minimum Variance'], tickers, "Minimum Variance Portfolio")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Show weights table
                    with st.expander("View Weights"):
                        df_weights = pd.DataFrame({
                            'Ticker': tickers,
                            'Weight': weights['Minimum Variance']
                        })
                        df_weights = df_weights[df_weights['Weight'] > 0.0001].sort_values('Weight', ascending=False)
                        st.dataframe(df_weights.style.format({'Weight': '{:.2%}'}), use_container_width=True)

                with col2:
                    st.subheader("Maximum Sharpe Ratio")
                    fig = plot_pie_chart(weights['Maximum Sharpe'], tickers, "Maximum Sharpe Ratio Portfolio")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("View Weights"):
                        df_weights = pd.DataFrame({
                            'Ticker': tickers,
                            'Weight': weights['Maximum Sharpe']
                        })
                        df_weights = df_weights[df_weights['Weight'] > 0.0001].sort_values('Weight', ascending=False)
                        st.dataframe(df_weights.style.format({'Weight': '{:.2%}'}), use_container_width=True)


                st.markdown("---")

                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Minimum Target Semivariance")
                    fig = plot_pie_chart(weights['Minimum Target Semivariance'], tickers,
                                         "Minimum Target Semivariance Portfolio")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("View Weights"):
                        df_weights = pd.DataFrame({
                            'Ticker': tickers,
                            'Weight': weights['Minimum Target Semivariance']
                        })
                        df_weights = df_weights[df_weights['Weight'] > 0.0001].sort_values('Weight', ascending=False)
                        st.dataframe(df_weights.style.format({'Weight': '{:.2%}'}), use_container_width=True)

                with col4:
                    st.subheader("Maximum Target Omega")
                    fig = plot_pie_chart(weights['Maximum Target Omega'], tickers, "Maximum Target Omega Portfolio")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("View Weights"):
                        df_weights = pd.DataFrame({
                            'Ticker': tickers,
                            'Weight': weights['Maximum Target Omega']
                        })
                        df_weights = df_weights[df_weights['Weight'] > 0.0001].sort_values('Weight', ascending=False)
                        st.dataframe(df_weights.style.format({'Weight': '{:.2%}'}), use_container_width=True)

                st.header("Strategy Comparison")

                # Create comparison table
                comparison_data = []
                for strategy_name, strategy_weights in weights.items():
                    metrics = calculate_portfolio_metrics(returns, strategy_weights, rf)
                    comparison_data.append({
                        'Strategy': strategy_name,
                        'Annual Return': metrics['Annual Return'],
                        'Annual Volatility': metrics['Annual Volatility'],
                        'Sharpe Ratio': metrics['Sharpe Ratio'],
                        'Sortino Ratio': metrics['Sortino Ratio']
                    })

                df_comparison = pd.DataFrame(comparison_data)
                df_comparison = df_comparison.set_index('Strategy')

                st.dataframe(
                    df_comparison.style.format({
                        'Annual Return': '{:.2%}',
                        'Annual Volatility': '{:.2%}',
                        'Sharpe Ratio': '{:.3f}',
                        'Sortino Ratio': '{:.3f}'
                    }),
                    use_container_width=True
                )

                st.markdown("---")

                # Correlation Heatmap
                st.subheader("Portfolio Correlation Heatmap")
                plot_correlation_heatmap(returns)


            # Tab 2: Backtesting Results
            with tab2:
                st.header("Historical Performance Simulation")
                st.info(
                    f"Backtesting with {rebalance_months}-month rebalancing period and ${initial_capital:,.0f} initial capital")

                with st.spinner("Running backtest simulation..."):
                    backtester = DynamicBacktester(
                        prices, benchmark_prices, rf,
                        months=rebalance_months,
                        cash=initial_capital
                    )
                    history = backtester.run_backtest()

                    # Plot backtesting results
                    plot_backtesting_results(history)

                # Summary statistics
                st.subheader("Performance Summary")

                col1, col2, col3, col4 = st.columns(4)

                final_values = history.iloc[-1]

                with col1:
                    best_strategy = final_values.drop('Benchmark').idxmax()
                    best_value = final_values.drop('Benchmark').max()
                    st.metric(
                        "Best Strategy",
                        best_strategy,
                        f"${best_value:,.2f}"
                    )

                with col2:
                    worst_strategy = final_values.drop('Benchmark').idxmin()
                    worst_value = final_values.drop('Benchmark').min()
                    st.metric(
                        "Worst Strategy",
                        worst_strategy,
                        f"${worst_value:,.2f}"
                    )

                with col3:
                    benchmark_value = final_values['Benchmark']
                    st.metric(
                        "Benchmark Final Value",
                        f"${benchmark_value:,.2f}",
                        f"{((benchmark_value - initial_capital) / initial_capital):.2%}"
                    )

                with col4:
                    avg_value = final_values.drop('Benchmark').mean()
                    st.metric(
                        "Average Portfolio Value",
                        f"${avg_value:,.2f}",
                        f"{((avg_value - initial_capital) / initial_capital):.2%}"
                    )

                # Download button
                st.markdown("---")
                csv = history.to_csv()
                st.download_button(
                    label="Download Backtest Results (CSV)",
                    data=csv,
                    file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

                st.subheader("Comprehensive Performance Analysis")

                metrics_df = calculate_backtest_metrics(history, initial_capital)

                st.dataframe(
                    metrics_df.style.set_properties(**{
                        'text-align': 'center',
                        'font-weight': 'bold'
                    }),
                    use_container_width=True
                )

                st.markdown("---")

                # Drawdown analysis
                st.subheader("Drawdown Analysis")

                drawdown_fig = go.Figure()

                for column in history.columns:
                    if column == 'Benchmark':
                        continue

                    values = history[column]
                    returns_series = values.pct_change().dropna()
                    cumulative = (1 + returns_series).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max

                    drawdown_fig.add_trace(go.Scatter(
                        x=history.index[1:],
                        y=drawdown * 100,
                        mode='lines',
                        name=column,
                        fill='tozeroy',
                        line=dict(width=1)
                    ))

                drawdown_fig.update_layout(
                    title="Portfolio Drawdowns Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    hovermode='x unified',
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(drawdown_fig, use_container_width=True)

                # Returns Distribution
                st.subheader("Returns Distribution")
                plot_returns_distribution(history)

                # Rolling Expected Return
                st.subheader("Rolling Expected Return (1-Year Window)")
                plot_rolling_return(history)

                # Rolling Volatility
                st.subheader("Rolling Volatility (1-Year Window)")
                plot_rolling_volatility(history)

                # Rolling Sharpe Ratio
                st.subheader("Rolling Sharpe Ratio (1-Year Window)")
                plot_rolling_sharpe(history, rf)

                # Rolling Sortino Ratio
                st.subheader("Rolling Sortino Ratio (1-Year Window)")
                plot_rolling_sortino(history, rf)



else:
    # Welcome screen
    st.info("Configure your portfolio parameters in the sidebar and click **Run Analysis** to begin.")

    st.markdown("""
    ## Features

    - **Portfolio Optimization**: Four different optimization strategies
        - Minimum Variance
        - Maximum Sharpe Ratio
        - Minimum Target Semivariance
        - Maximum Target Omega

    - **Interactive Visualizations**: Dynamic pie charts and line plots

    - **Backtesting Engine**: Historical simulation with configurable rebalancing

    - **Performance Metrics**: Comprehensive risk-adjusted return analysis

    ## How to Use

    1. Select your assets (tickers) in the sidebar
    2. Choose a benchmark for comparison
    3. Set your date range and parameters
    4. Click "Run Analysis" to optimize and backtest
    5. Explore results across different tabs

    ## Optimization Strategies Explained

    - **Minimum Variance**: Minimizes portfolio volatility
    - **Maximum Sharpe**: Maximizes risk-adjusted returns
    - **Minimum Semivariance**: Minimizes downside risk relative to benchmark
    - **Maximum Omega**: Maximizes probability-weighted gains over losses relative to benchmark
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Portfolio Optimization Dashboard</div>",
    unsafe_allow_html=True
)