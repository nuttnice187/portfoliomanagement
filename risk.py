import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize, OptimizeResult

def get_returns_p(weights: NDArray[np.float64], mean_returns: pd.Series,
    trading_days: int) -> float:
    return np.sum(mean_returns*weights)*trading_days
def get_std_dev_p(weights: NDArray[np.float64], cov_matrix: pd.DataFrame,
    trading_days: int) -> float:
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(
        trading_days)
def get_neg_sharpe_ratio(weights: NDArray[np.float64], mean_returns: pd.Series,
    cov_matrix: pd.DataFrame, trading_days: int, risk_free_rate: float
    ) -> float:
    returns_p = get_returns_p(weights, mean_returns, trading_days)
    std_dev_p = get_std_dev_p(weights, cov_matrix, trading_days)
    return - (returns_p - risk_free_rate) / std_dev_p
def get_weight_allocation(symbols: Union[List[str], pd.Index],
    weights: NDArray[np.float64]) -> Dict[str, str]:
    res = {symbols[0]: "{:.2%}".format(weights[0])}
    for i in range(1, len(symbols)):
        s = symbols[i]
        weight = "{:.2%}".format(weights[i])
        res[s] = weight
    return res

class Portfolio:
    returns: float
    std_dev: float
    sharpe_ratio: float
    weights: NDArray[np.float64]
    symbols: pd.Index
    def __init__(self, weights: NDArray[np.float64], mean_returns: pd.Series, 
        cov_matrix: pd.DataFrame, trading_days: int, risk_free_rate: float):
        self.returns = get_returns_p(weights, mean_returns, trading_days)
        self.std_dev = get_std_dev_p(weights, cov_matrix, trading_days)
        self.sharpe_ratio = (self.returns_p - risk_free_rate) / self.std_dev_p
        self.weights = weights
        self.symbols = mean_returns.index
    def __repr__(self):
        res = []
        for s in self.symbols:
            res.append('    Returns: {:.2%}'.format(self.returns))
            res.append('    Standard Deviation: {:.2%}'.format(self.std_dev))
            res.append('    Sharpe Ratio: {:.2}'.format(self.sharpe_ratio))
            res.append('    Weight Allocation:')
            for k, v in get_weight_allocation(self.symbols, self.weights
                ).items():
                res.append('        {}: {}'.format(k, v))
        return '\n'.join(res)

class EfficientFrontier:
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    risk_free_rate: float
    trading_days: int
    asset_len: int
    max_sharpe_ratio_portfolio: OptimizeResult
    min_risk_portfolio: OptimizeResult
    bound: Tuple[float, float]
    def __init__(self, adjusted_close: pd.DataFrame, risk_free_rate: float=0.04,
        bound: Tuple[float, float]=(0, 1)) -> None:
        self.trading_days, self.asset_len = adjusted_close.shape
        percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.cov_matrix = percent_change.cov()
        self.risk_free_rate = risk_free_rate
        self.bound = bound
        self.max_sharpe_p = self.get_optimal_portfolio(*(
            get_neg_sharpe_ratio, self.mean_returns, self.cov_matrix,
            self.trading_days, self.risk_free_rate))
        self.min_risk_p = self.get_optimal_portfolio(*(get_std_dev_p,
            self.cov_matrix, self.trading_days))
    def __repr__(self) -> str:
        portfolios: Tuple[Tuple[str, Portfolio]]= (
            ('Maximum Sharpe Ratio:', self.max_sharpe_p),
            ('Minimum Risk:', self.min_risk_p))
        res: List= []
        for description, p in portfolios:
            res.append(description)
            res.append(p.__repr__())
        return '\n'.join(res)
    def get_optimal_portfolio(self, fun: Union[
            Callable[[NDArray[np.float64], pd.DataFrame, int], float],
            Callable[[NDArray[np.float64], pd.Series, pd.DataFrame, int, float],
                float]],
        *args: Union[pd.Series, pd.DataFrame, int, float], **kwargs: float
        ) -> Portfolio:        
        constraints = {"type": 'eq', "fun": lambda x: np.sum(x) - 1}
        if 'target_return' in kwargs and kwargs['target_return']:
            return_p_constraints = {"type": 'eq',
                "fun": lambda x: get_returns_p(x, self.mean_returns,
                    self.trading_days) - kwargs['target_return']}
            constraints = (return_p_constraints, constraints)
        bounds: Tuple[Tuple[float, float]]= tuple(
            self.bound for i in range(self.asset_len))
        initial_weights: List[float]= self.asset_len*[1/self.asset_len]
        opt_res = minimize(fun, initial_weights, args=args, method='SLSQP',
            bounds=bounds, constraints=constraints)
        return Portfolio(opt_res.x, self.mean_returns, self.cov_matrix,
            self.trading_days, self.risk_free_rate)
