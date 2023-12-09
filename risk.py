import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, List, Tuple, Union, Callable
from scipy.optimize import minimize, OptimizeResult

def get_returns_p(weights: npt.NDArray[np.float64],
    mean_returns: pd.Series, trading_days: int) -> float:
    return np.sum(mean_returns*weights)*trading_days
def get_std_dev_p(weights: npt.NDArray[np.float64],cov_matrix: pd.DataFrame,
    trading_days: int) -> float:
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(
        trading_days)
def get_neg_sharpe_ratio(weights: npt.NDArray[np.float64],
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, trading_days: int,
    risk_free_rate: float) -> float:
    returns_p = get_returns_p(weights, mean_returns, trading_days)
    std_dev_p = get_std_dev_p(weights, cov_matrix, trading_days)
    return - (returns_p - risk_free_rate) / std_dev_p

class EfficientFrontierModel:
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    risk_free_rate: float
    trading_days: int
    asset_len: int
    max_sharpe_ratio_portfolio: OptimizeResult
    def __init__(self, adjusted_close: pd.DataFrame, risk_free_rate: float=0.04
        ) -> None:
        self.trading_days, self.asset_len = adjusted_close.shape
        percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.cov_matrix = percent_change.cov()
        self.risk_free_rate = risk_free_rate
        self.max_sharpe_ratio_portfolio = self.__get_optimal_portfolio(
            get_neg_sharpe_ratio, (self.mean_returns, self.cov_matrix,
                self.trading_days, self.risk_free_rate))
    def __get_optimal_portfolio(self, fun: Callable, *args,
        weight_limit: Tuple[float, float]=(0, 1)) -> OptimizeResult:
        constraints: Dict[str, Union[str, function]]= {"type": 'eq',
            "fun": lambda x: np.sum(x) - 1}
        bounds: Tuple[Tuple[float, float]]= tuple(
            weight_limit for i in range(self.asset_len))
        initial_weights: List[float]= self.asset_len*[1/self.asset_len]
        self.max_sharpe_ratio_portfolio = minimize(fun,
            initial_weights, args=args, method='SLSQP', bounds=bounds,
            constraints=constraints)
