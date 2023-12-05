import pandas as pd
import numpy as np
import numpy.typing as npt

class EfficientFrontierModel:
    percent_change: pd.DataFrame
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    returns: float
    std_dev: float
    def __init__(self, adjusted_close: pd.DataFrame) -> None:
        self.percent_change = adjusted_close.pct_change()
        self.mean_returns = self.percent_change.mean()
        self.cov_matrix = self.percent_change.cov()
    def get_portfolio_performance(self, weights: npt.NDArray[np.float64],
        trading_days: int=252) -> None:
        self.returns = np.sum(self.mean_returns*weights)*trading_days
        self.std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix,
            weights)))*np.sqrt(trading_days)
