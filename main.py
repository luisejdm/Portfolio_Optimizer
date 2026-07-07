from utils import download_prices, calculate_returns
from visualization import plot_weights, plot_backtesting
from optimizer import *
from backtesting import DynamicBacktester

tickers = ['AAPL', 'MSFT', 'GOOG', 'JPM']
benchmark_ticker = ['^GSPC']
start_date = '2020-01-01'
end_date = '2025-07-01'
rf = 0.035

prices = download_prices(tickers, start_date, end_date)
benchmark = download_prices(benchmark_ticker, start_date, end_date)
returns = calculate_returns(prices)
benchmark_returns = calculate_returns(benchmark)

def main():
    backtester = DynamicBacktester(prices, benchmark, rf, months=12, cash=1_000_000)
    history = backtester.run_backtest()
    plot_backtesting(history)

    # optimizer = PortfolioOptimizer(returns, rf)
    # w_min_v = optimizer.min_variance()
    # w_max_sh = optimizer.max_sharpe()
    # w_min_semivar = optimizer.min_target_semivariance(benchmark_returns)
    # w_max_omega = optimizer.max_target_omega(benchmark_returns)
    #
    # plot_weights(w_min_v, tickers, 'Minimum Variance')
    # plot_weights(w_max_sh, tickers, 'Maximum Sharpe Ratio')
    # plot_weights(w_min_semivar, tickers, 'Minimum Target Semivariance')
    # plot_weights(w_max_omega, tickers, 'Maximum Target Omega')
if __name__ == "__main__":
    main()