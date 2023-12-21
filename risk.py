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
    name: Optional[str]
    def __init__(self, weights: NDArray[np.float64], mean_returns: pd.Series, 
        cov_matrix: pd.DataFrame, trading_days: int, risk_free_rate: float,
        name: Optional[str]=None) -> None:
        self.p_return = get_return_p(weights, mean_returns, trading_days)
        self.std_dev = get_std_dev_p(weights, cov_matrix, trading_days)
        self.sharpe_ratio = (self.p_return - risk_free_rate) / self.std_dev
        self.weights, self.symbols = weights, mean_returns.index
        self.name = name
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

class Marker:
    name: str
    mode: str
    x: List[float]
    y: List[float]
    marker: Dict[str, Union[str, int, Dict[str, Union[int, str]]]]
    hovertext: str
    def __init__(self, portfolio: Portfolio, color: str,
        outline: Optional[bool]=None) -> None:
        self.name, self.mode = portfolio.name, 'markers'
        self.x, self.y = [portfolio.std_dev], [portfolio.p_return]
        self.marker = {"color": color, "size": 14}
        if outline:
            self.marker["line"] = {"width": 3, "color": 'black'}
        self.hovertext = portfolio.__repr__(sep='<br>')

class Lines:
    name: str
    mode: str
    x: List[float]
    y: NDArray
    line: Dict[str, Union[int, str]]
    hovertexts: List[str]
    def __init__(self, x: List[float], y: NDArray, hovertexts: List[str],
        name: str='Efficient Frontier') -> None:
        self.name, self.mode, self.x, self.y = name, 'lines', x, y
        self.line = {"width": 4, "color": 'black', "dash": 'dashdot'}
        self.hovertext = hovertexts

class FrontierLayout:
    title: str
    yaxis: Dict[str, str]
    xaxis: Dict[str, str]
    showlegend: bool
    legend: Dict[str, Union[float, str, int]]
    width: int
    height: int
    def __init__(self, title: str='Portfolio Optimization') -> None:
        self.title = title
        self.yaxis = {"title": 'Return', "tickformat": ',.0%'}
        self.xaxis = {"title": 'Standard Deviation', "tickformat": ',.0%'}
        self.showlegend = True
        self.legend = {"x": .75, "y": 0, "traceorder": 'normal',
            "bgcolor": '#E2E2E2', "bordercolor": 'black', "borderwidth": 2}
        self.width = 800
        self.height = 600

class Constraints:
    weight: Dict[str, Union[str, Callable[[NDArray], float]]]
    return_p: Dict[str, Union[str, Callable[[NDArray], float]]]
    std_dev: Dict[str, Union[str, Callable[[NDArray], float]]]
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
        trading_days: int, target_return: Optional[float]=None,
        target_std_dev: Optional[float]=None) -> None:
        self.weight = {"type": 'eq', "fun": lambda x: np.sum(x) - 1}
        if target_return:
            self.return_p = {"type": 'eq',
                "fun": lambda x: get_return_p(x, mean_returns, trading_days
                    ) - target_return}
        if target_std_dev:
            self.std_dev = {"type": 'eq',
                "fun": lambda x: get_std_dev_p(x, cov_matrix, trading_days
                    ) - target_std_dev}

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
        self.risk_free_rate, self.bound = risk_free_rate, bound
        self.max_sharpe_p = self.predict(max_sharpe=True)
        self.min_risk_p = self.predict(min_risk=True)
        self.fig = self.__plot_frontier_curve()
    def __repr__(self) -> str:
        portfolios: Tuple[Portfolio]= (self.max_sharpe_p,  self.min_risk_p)
        res: List= []
        for p in portfolios:
            res.append(p.name)
            res.append(p.__repr__())
        return '\n'.join(res)
    def __get_optimal_portfolio(self, fun: Callable, constraints:  Tuple[Dict[
            str, Union[str, Callable[[NDArray[np.float64]], float]]]],
        *args: Union[pd.Series, pd.DataFrame, int, float], **kwargs: str
        ) -> Portfolio:
        name = None
        if 'name' in kwargs and kwargs['name']:
            name = kwargs['name']
        bounds: Tuple[Tuple[float, float]]= tuple(self.bound for i in range(
            self.asset_len))
        initial_weights: List[float]= self.asset_len*[1/self.asset_len]
        opt_res: OptimizeResult= minimize(fun, initial_weights, args=args,
            method='SLSQP', bounds=bounds, constraints=constraints)
        return Portfolio(opt_res.x, self.mean_returns, self.cov_matrix,
            self.trading_days, self.risk_free_rate, name=name)
    def __get_frontier_returns(self, n: int=20) -> NDArray:
        return np.linspace(self.min_risk_p.p_return,
            self.max_sharpe_p.p_return, n)
    def __get_frontier_std_devs_hover_text(self, frontier_returns: NDArray[
            np.float64]) -> Tuple[List[float], List[str]]:
        frontier_std_devs, hover_text = [], []
        for r in frontier_returns:
            portfolio = self.predict(target_return=r)
            frontier_std_devs.append(portfolio.std_dev)
            hover_text.append(portfolio.__repr__(sep='<br>'))
        return frontier_std_devs, hover_text
    def __plot_frontier_curve(self) -> Figure:
        y = self.__get_frontier_returns()
        x, hovertexts = self.__get_frontier_std_devs_hover_text(y)
        sharpe_ratio_marker = Scatter(**Marker(self.max_sharpe_p, 'red',
            outline=True).__dict__)
        std_dev_marker = Scatter(**Marker(self.min_risk_p, 'green',
            outline=True).__dict__)
        curve = Scatter(**Lines(x, y, hovertexts).__dict__)
        data: List[Scatter]= [sharpe_ratio_marker, std_dev_marker, curve]
        layout = Layout(**FrontierLayout().__dict__)
        return Figure(data=data, layout=layout)
    def __name_portfolio(self, max_sharpe: Optional[bool]=None,
        min_risk: Optional[bool]=None, name: Optional[str]=None) -> str:
        if max_sharpe:
            name = 'Maximum Sharpe Ratio'
        elif min_risk:
            name = 'Minimum Risk'
        return name
    def __check_optimize_type(self, max_sharpe: Optional[bool]=None,
        target_return: Optional[float]=None,
        target_std_dev: Optional[float]=None, name: Optional[str]=None
        ) -> Portfolio:
        c = Constraints(self.mean_returns, self.cov_matrix, self.trading_days,
            target_return, target_std_dev).__dict__.values()
        if max_sharpe or target_std_dev:
            res = self.__get_optimal_portfolio(*(get_neg_sharpe_ratio, c,
                    self.mean_returns, self.cov_matrix, self.trading_days,
                    self.risk_free_rate),
                **{"name": name})
        else:
            res = self.__get_optimal_portfolio(*(get_std_dev_p, c,
                    self.cov_matrix, self.trading_days),
                **{"name": name})
        return res
    def predict(self, target_return: Optional[float]=None, 
        target_std_dev: Optional[float]=None, max_sharpe: Optional[bool]=None,
        min_risk: Optional[bool]=None, name: Optional[str]=None) -> Portfolio:
        option_assertions = iter([max_sharpe, min_risk,
            (target_return or target_std_dev)])
        msg = "Too many options."
        assert any(option_assertions) and not any(option_assertions), msg
        name = self.__name_portfolio(max_sharpe, min_risk, name)
        return self.__check_optimize_type(max_sharpe=max_sharpe,
            target_std_dev=target_std_dev, target_return=target_return,
            name=name)