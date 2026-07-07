import pandas as pd
from optimizer import PortfolioOptimizer

class DynamicBacktester:
    def __init__(self,
                 prices: pd.DataFrame,
                 prices_benchmark: pd.DataFrame,
                 rf: float,
                 months: int,
                 cash: float):
        self.prices = prices
        self.prices_benchmark = prices_benchmark
        self.rf = rf
        self.months = months
        self.cash = cash


    def _optimize_weights(self, prices, ndays, periods):
        start_idx = int(ndays*periods)
        end_idx = int(ndays*(periods+1))
        temp_data = prices.iloc[start_idx:end_idx, :]
        temp_bm = self.prices_benchmark.copy().iloc[start_idx:end_idx, :]
        temp_rt = temp_data.pct_change().dropna()
        bm_rt = temp_bm.pct_change().dropna()

        optimizer = PortfolioOptimizer(temp_rt, self.rf)
        w_min_v = optimizer.min_variance()
        w_max_sh = optimizer.max_sharpe()
        w_min_semi = optimizer.min_target_semivariance(bm_rt)
        w_max_om = optimizer.max_target_omega(bm_rt)
        return w_min_v, w_max_sh, w_min_semi, w_max_om


    def run_backtest(self):
        n_days = round(len(self.prices) / round(len(self.prices) / 252 / (self.months / 12)), 0)
        cash = self.cash
        opt_data = self.prices.copy().iloc[:int(n_days), :]
        backtest_data = self.prices.copy().iloc[int(n_days):, :]
        backtest_rt = backtest_data.pct_change().dropna()

        day_counter, period_counter = 0, 0
        min_v, max_sh, min_semi, max_om, bm = [cash], [cash], [cash], [cash], [cash]
        w_min_v, w_max_sh, w_min_semi, w_max_om = self._optimize_weights(
            opt_data, n_days, 0
        )

        for day in range(len(backtest_data)-1):
            if day_counter < n_days:
                min_v.append(min_v[-1] * (1 + (backtest_rt.iloc[day, :] @ w_min_v)))
                max_sh.append(max_sh[-1] * (1 + (backtest_rt.iloc[day, :] @ w_max_sh)))
                min_semi.append(min_semi[-1] * (1 + (backtest_rt.iloc[day, :] @ w_min_semi)))
                max_om.append(max_om[-1] * (1 + (backtest_rt.iloc[day, :] @ w_max_om)))
                bm.append(bm[-1] * (1 + backtest_rt.iloc[day, :].mean()))

            else:
                w_min_v, w_max_sh, w_min_semi, w_max_om = self._optimize_weights(
                    backtest_data, n_days, period_counter
                )
                min_v.append(min_v[-1] * (1 + (backtest_rt.iloc[day, :] @ w_min_v)))
                max_sh.append(max_sh[-1] * (1 + (backtest_rt.iloc[day, :] @ w_max_sh)))
                min_semi.append(min_semi[-1] * (1 + (backtest_rt.iloc[day, :] @ w_min_semi)))
                max_om.append(max_om[-1] * (1 + (backtest_rt.iloc[day, :] @ w_max_om)))
                bm.append(bm[-1] * (1 + backtest_rt.iloc[day, :].mean()))

                day_counter = 0
                period_counter += 1

            day_counter += 1

        df = pd.DataFrame()
        df['Date'] = backtest_data.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Min Variance'] = min_v
        df['Max Sharpe'] = max_sh
        df['Min Target Semivariance'] = min_semi
        df['Max Target Omega'] = max_om
        df['Benchmark'] = bm
        df.set_index('Date', inplace=True)
        return df


