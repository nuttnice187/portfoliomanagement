import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Dict, List, Tuple, Union, Callable
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
    portfolio: OptimizeResult) -> Dict[str, str]:
    res = {symbols[0]: "{:.2%}".format(portfolio.x[0])}
    for i in range(1, len(symbols)):
        s = symbols[i]
        weight = "{:.2%}".format(portfolio.x[i])
        res[s] = weight
    return res

class EfficientFrontierModel:
    symbols: pd.Index
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
        self.symbols = adjusted_close.columns
        self.trading_days, self.asset_len = adjusted_close.shape
        percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.cov_matrix = percent_change.cov()
        self.risk_free_rate = risk_free_rate
        self.bound = bound
        self.max_sharpe_ratio_portfolio = self.__get_optimal_portfolio(
            *(get_neg_sharpe_ratio, self.mean_returns, self.cov_matrix,
                self.trading_days, self.risk_free_rate))
        self.min_risk_portfolio = self.__get_optimal_portfolio(*(get_std_dev_p,
            self.cov_matrix, self.trading_days))
    def __repr__(self) -> str:
        portfolios: Tuple[Tuple[str, OptimizeResult]]= (
            ('Maximum Sharpe Ratio: {:.2}'
                 .format(-self.max_sharpe_ratio_portfolio.fun),
                 self.max_sharpe_ratio_portfolio),
            ('Minimum Risk:', self.min_risk_portfolio))
        res: List= []
        for description, optimization_result in portfolios:
            res.append(description)
            res.append('    Returns: {:.2%}'.format(get_returns_p(
                optimization_result.x, self.mean_returns, self.trading_days)))
            res.append('    Risk Volatility: {:.2%}'.format(get_std_dev_p(
                optimization_result.x, self.cov_matrix, self.trading_days)))
            res.append('    Weight Allocation:')
            for k, v in get_weight_allocation(self.symbols,
                optimization_result).items():
                res.append('        {}: {}'.format(k, v))
        return '\n'.join(res)
    def __get_optimal_portfolio(self, fun: Union[
            Callable[[NDArray[np.float64], pd.DataFrame, int], float],
            Callable[[NDArray[np.float64], pd.Series, pd.DataFrame, int, float],
                float]],
        *args: Union[pd.Series, pd.DataFrame, int, float]) -> OptimizeResult:
        constraints: Dict[str, Union[str, function]]= {"type": 'eq',
            "fun": lambda x: np.sum(x) - 1}
        bounds: Tuple[Tuple[float, float]]= tuple(
            self.bound for i in range(self.asset_len))
        initial_weights: List[float]= self.asset_len*[1/self.asset_len]
        return minimize(fun, initial_weights, args=args, method='SLSQP',
            bounds=bounds, constraints=constraints)
