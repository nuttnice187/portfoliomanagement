import pandas as pd
import numpy as np

class EfficientFrontierModel:
    percent_change: pd.DataFrame
    mean_returns: pd.Series
    covariance_matrix: pd.DataFrame
    returns: float
    sigma: float
    def __init__(self, adjusted_close: pd.DataFrame):
        self.percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.covariance_matrix = percent_change.cov()
    def get_portfolio_performance(self, weights, trading_days: int=252):
        self.returns = np.sum(self.mean_returns*weights)*trading_days
        self.sigma = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            )*np.sqrt(trading_days)
