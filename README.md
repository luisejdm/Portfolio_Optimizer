# Portfolio Optimizer

Portfolio Optimizer is a Streamlit application for building, comparing, and backtesting equity portfolios using historical market data.

It downloads price data from Yahoo Finance, calculates returns, optimizes allocations with multiple portfolio strategies, and simulates how those strategies would have performed over time.

## Features

- Interactive Streamlit dashboard
- Historical price download with `yfinance`
- Portfolio optimization with four strategies:
	- Minimum Variance
	- Maximum Sharpe Ratio
	- Minimum Target Semivariance
	- Maximum Target Omega
- Dynamic backtesting with periodic rebalancing
- Portfolio metrics such as return, volatility, Sharpe ratio, Sortino ratio, and drawdown
- Visualizations for portfolio weights, correlation heatmaps, and performance curves

## Project Structure

- `dashboard.py`: main Streamlit dashboard
- `dashboard_utils.py`: data loading, optimization helpers, metrics, and charts
- `optimizer.py`: optimization logic based on SciPy
- `backtesting.py`: dynamic backtesting engine
- `utils.py`: price download and return calculations
- `visualization.py`: matplotlib-based charts used in the console demo
- `main.py`: simple script that runs a backtest example

## Requirements

The project uses:

- Python 3.10+
- streamlit
- pandas
- numpy
- scipy
- yfinance
- plotly
- matplotlib
- seaborn

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy scipy yfinance plotly matplotlib seaborn
```

If you prefer, you can also create a `requirements.txt` file with those packages and install them with:

```bash
pip install -r requirements.txt
```

## Usage

### Launch the dashboard

```bash
streamlit run dashboard.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

### Run the console example

```bash
python main.py
```

This script downloads sample data, runs a dynamic backtest, and displays the results with matplotlib.

## How It Works

1. Select a set of tickers and a benchmark.
2. Choose the date range, risk-free rate, rebalancing period, and initial capital.
3. Load historical prices from Yahoo Finance.
4. Optimize the portfolio using the selected strategies.
5. Compare the resulting allocations and performance metrics.
6. Run a historical backtest to evaluate strategy behavior over time.

## Notes

- The dashboard expects at least two tickers.
- The benchmark defaults to `^GSPC` (S&P 500).
- Results depend on the quality and availability of market data from Yahoo Finance.

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice.
