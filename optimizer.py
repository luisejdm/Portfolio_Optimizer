from scipy.optimize import minimize
import pandas as pd
import numpy as np


class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, rf: float):
        self.returns = returns
        self.rf = rf
        self.n_assets = len(returns.columns)
        self.cov_matrix = returns.cov()
        self.mean_rt = returns.mean()


    def _calculate_minimum_bound(self):
        n_assets = len(self.returns.columns)
        minimum = 0.25 / n_assets
        return minimum


    def min_variance(self):
        variance = lambda w: w.T @ self.cov_matrix @ w
        x0 = np.ones(self.n_assets) / self.n_assets
        bounds = [(self._calculate_minimum_bound(), 3)]*self.n_assets
        constraints = {'fun': lambda w: np.sum(w) - 1, 'type': 'eq'}
        result = minimize(variance, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
        return result.x


    def max_sharpe(self):
        returns = self.returns
        rend = returns.mean()
        cov_matrix = self.cov_matrix
        rf = self.rf

        sharpe = lambda w: -((np.dot(rend, w) - rf/252) / np.sqrt(w @ cov_matrix @ w))
        x0 = np.ones(self.n_assets) / self.n_assets
        bounds = [(self._calculate_minimum_bound(), 3)]*self.n_assets
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        result = minimize(sharpe, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
        return result.x


    def min_target_semivariance(self, benchmark):
        returns = self.returns
        corr = returns.corr()
        differences = returns - benchmark.values
        below_zero_target = differences[differences < 0].fillna(0)
        target_downside = np.array(below_zero_target.std())
        target_semivariance = np.multiply(target_downside.reshape(len(target_downside), 1), target_downside) * corr

        semivariance = lambda w: w.T @ target_semivariance @ w
        x0 = np.ones(self.n_assets) / self.n_assets
        bounds = [(self._calculate_minimum_bound(), 3)]*self.n_assets
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        result = minimize(semivariance, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
        return result.x


    def max_target_omega(self, benchmark):
        returns = self.returns
        differences = returns - benchmark.values
        below_zero_target = differences[differences < 0].fillna(0)
        above_zero_target = differences[differences > 0].fillna(0)
        target_downside = np.array(below_zero_target.std())
        target_upside = np.array(above_zero_target.std())

        omega = lambda w: -(target_upside / target_downside).dot(w)
        x0 = np.ones(self.n_assets) / self.n_assets
        bounds = [(self._calculate_minimum_bound(), 3)]*self.n_assets
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        result = minimize(omega, x0, bounds=bounds, constraints=constraints, tol=1e-16, method='SLSQP')
        return result.x