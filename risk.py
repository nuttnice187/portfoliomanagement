import numpy as np
import pandas as pd
from numpy.typing import NDArray
from plotly.graph_objects import Scatter, Layout, Figure
from typing import Callable, Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize, OptimizeResult

def get_return_p(weights: NDArray[np.float64], mean_returns: pd.Series,
    trading_days: int) -> float:
    return np.sum(mean_returns*weights)*trading_days
def get_std_dev_p(weights: NDArray[np.float64], cov_matrix: pd.DataFrame,
    trading_days: int) -> float:
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(
        trading_days)
def get_neg_sharpe_ratio(weights: NDArray[np.float64], mean_returns: pd.Series,
    cov_matrix: pd.DataFrame, trading_days: int, risk_free_rate: float
    ) -> float:
    returns_p = get_return_p(weights, mean_returns, trading_days)
    std_dev_p = get_std_dev_p(weights, cov_matrix, trading_days)
    return - (returns_p - risk_free_rate) / std_dev_p

class Portfolio:
    p_return: float
    std_dev: float
    sharpe_ratio: float
    weights: NDArray[np.float64]
    symbols: pd.Index
    def __init__(self, weights: NDArray[np.float64], mean_returns: pd.Series, 
        cov_matrix: pd.DataFrame, trading_days: int, risk_free_rate: float
        ) -> None:
        self.p_return = get_return_p(weights, mean_returns, trading_days)
        self.std_dev = get_std_dev_p(weights, cov_matrix, trading_days)
        self.sharpe_ratio = (self.p_return - risk_free_rate) / self.std_dev
        self.weights = weights
        self.symbols = mean_returns.index
    def __repr__(self, sep: str='\n') -> str:
        res = []
        res.append('    Returns: {:.2%}'.format(self.p_return))
        res.append('    Standard Deviation: {:.2%}'.format(self.std_dev))
        res.append('    Sharpe Ratio: {:.2}'.format(self.sharpe_ratio))
        res.append('    Weight Allocation:')
        for k, v in self.get_weight_allocation().items():
            res.append('        {}: {:.2%}'.format(k, v))
        return sep.join(res)
    def get_weight_allocation(self) -> Dict[str, float]:        
        res = {self.symbols[0]: self.weights[0]}
        for i in range(1, len(self.symbols)):
            s = self.symbols[i]
            weight = self.weights[i]
            res[s] = weight
        return res

class EfficientFrontier:
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    risk_free_rate: float
    trading_days: int
    asset_len: int
    max_sharpe_p: Portfolio
    min_risk_p: Portfolio
    bound: Tuple[float, float]
    fig: Figure
    def __init__(self, adjusted_close: pd.DataFrame, risk_free_rate: float=0.04,
        bound: Tuple[float, float]=(0, 1)) -> None:
        self.trading_days, self.asset_len = adjusted_close.shape
        percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.cov_matrix = percent_change.cov()
        self.risk_free_rate = risk_free_rate
        self.bound = bound
        self.max_sharpe_p = self.predict(max_sharpe=True)
        self.min_risk_p = self.predict(min_risk=True)
        self.fig = self.__plot_frontier_curve()
    def __repr__(self) -> str:
        portfolios: Tuple[Tuple[str, Portfolio]]= (
            ('Maximum Sharpe Ratio:', self.max_sharpe_p),
            ('Minimum Risk:', self.min_risk_p))
        res: List= []
        for description, p in portfolios:
            res.append(description)
            res.append(p.__repr__())
        return '\n'.join(res)
    def __get_optimal_portfolio(self, fun: Union[
            Callable[[NDArray[np.float64], pd.DataFrame, int], float],
            Callable[[NDArray[np.float64], pd.Series, pd.DataFrame, int, float],
                float]],
        constraints: Union[Dict[str,
            Union[str, Callable[[NDArray[np.float64]], float]]],Tuple[
                    Dict[str, Union[str, Callable[[NDArray[np.float64]],
                        float]]],
                    Dict[str, Union[str, Callable[[NDArray[np.float64]],
                        float]]]
                ]],
        *args: Union[pd.Series, pd.DataFrame, int, float]) -> Portfolio:
        bounds: Tuple[Tuple[float, float]]= tuple(
            self.bound for i in range(self.asset_len))
        initial_weights: List[float]= self.asset_len*[1/self.asset_len]
        opt_res: OptimizeResult= minimize(fun, initial_weights, args=args,
            method='SLSQP', bounds=bounds, constraints=constraints)
        return Portfolio(opt_res.x, self.mean_returns, self.cov_matrix,
            self.trading_days, self.risk_free_rate)
    def __get_frontier_returns(self, n: int=20) -> NDArray:
        return np.linspace(self.min_risk_p.p_return, self.max_sharpe_p.p_return,
            n)
    def __get_frontier_std_devs_hover_text(self,
        frontier_returns: NDArray[np.float64]
        ) -> Tuple[List[float], List[float]]:
        frontier_std_devs, hover_text = [], []
        for r in frontier_returns:
            portfolio = self.predict(target_return=r)
            frontier_std_devs.append(portfolio.std_dev)
            hover_text.append(portfolio.__repr__(sep='<br>'))
        return frontier_std_devs, hover_text
    def __plot_frontier_curve(self) -> Figure:
        frontier_returns: NDArray= self.__get_frontier_returns()
        frontier_std_devs, frontier_hovertexts = (self
            .__get_frontier_std_devs_hover_text(frontier_returns))
        max_sharpe_ratio_marker = Scatter(name='Maximum Sharpe Ratio',
            mode='markers', x=[self.max_sharpe_p.std_dev],
            y=[self.max_sharpe_p.p_return], marker={"color": 'red', "size": 14,
                "line": {"width": 3, "color": 'black'}},
            hovertext=self.max_sharpe_p.__repr__(sep='<br>'))
        min_std_dev_marker = Scatter(name='Minimum Standard Deviation',
            mode='markers', x=[self.min_risk_p.std_dev],
            y=[self.min_risk_p.p_return], marker={"color": 'green', "size": 14,
                "line": {"width": 3, "color": 'black'}},
            hovertext=self.min_risk_p.__repr__(sep='<br>'))
        frontier_curve = Scatter(name='Efficient Frontier', mode='lines', 
            x=frontier_std_devs, y=frontier_returns, line={"width": 4,
                "color": 'black', "dash": 'dashdot'},
            hovertext=frontier_hovertexts)
        data: List[Scatter]= [max_sharpe_ratio_marker, min_std_dev_marker,
            frontier_curve]
        layout = Layout(title='Portfolio Optimization', yaxis={
                "title": 'Return', "tickformat": ',.0%'},
            xaxis={"title": 'Standard Deviation', "tickformat": ',.0%'},
            showlegend=True, legend={"x": .75, "y": 0, "traceorder": 'normal',
                "bgcolor": '#E2E2E2', "bordercolor": 'black', "borderwidth": 2
                }, width=800, height=600)
        return Figure(data=data, layout=layout)
    def predict(self, target_return: Optional[float]=None, 
        target_std_dev: Optional[float]=None, max_sharpe: Optional[bool]=None,
        min_risk: Optional[bool]=None) -> Portfolio:
        c = {"type": 'eq', "fun": lambda x: np.sum(x) - 1}
        return_p_constraint = {"type": 'eq',
            "fun": lambda x: get_return_p(x, self.mean_returns,
                self.trading_days) - target_return}
        std_dev_constraint = {"type": 'eq',
            "fun": lambda x: get_std_dev_p(x, self.cov_matrix,
                self.trading_days) - target_std_dev}
        if target_return and target_std_dev:
            c = (c, return_p_constraint, std_dev_constraint)
        elif target_return:
            c = (return_p_constraint, c)
        elif target_std_dev:
            c = (std_dev_constraint, c)
        if max_sharpe or target_std_dev:
            portfolio_res: Portfolio = self.__get_optimal_portfolio(*(
                get_neg_sharpe_ratio, c, self.mean_returns,
                self.cov_matrix, self.trading_days, self.risk_free_rate))
        else:
            portfolio_res: Portfolio = self.__get_optimal_portfolio(*(
                get_std_dev_p, c, self.cov_matrix, self.trading_days)
                )
        return portfolio_res
