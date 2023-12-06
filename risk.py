import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple

class EfficientFrontierModel:
    percent_change: pd.DataFrame
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    def __init__(self, adjusted_close: pd.DataFrame) -> None:
        self.percent_change = adjusted_close.pct_change()
        self.mean_returns = self.percent_change.mean()
        self.cov_matrix = self.percent_change.cov()
    def get_portfolio_performance(self, weights: npt.NDArray[np.float64],
        trading_days: int=252) -> Tuple[float, float]:
        returns_p: float= np.sum(self.mean_returns*weights)*trading_days
        std_dev_p: float= np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix,
            weights)))*np.sqrt(trading_days)
        return returns_p, std_dev_p
    def get_neg_sharp_ratio(self, weights: npt.NDArray[np.float64],
        risk_free_rate: float) -> float:
        returns_p, std_dev_p = self.get_portfolio_performance(weights)
        return - (returns_p - risk_free_rate) / std_dev_p
