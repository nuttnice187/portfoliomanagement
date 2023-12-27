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
        if self.name:
            res.append(self.name)
        res.append('    Returns: {:.2%}'.format(self.p_return))
        res.append('    Standard Deviation: {:.2%}'.format(self.std_dev))
        res.append('    Sharpe Ratio: {:.4}'.format(self.sharpe_ratio))
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

class RandomPortfolios:
    x: List[float]
    y: List[float]
    marker: Dict[str, Union[List[float], bool, int, Dict[str, int], str, 
            Dict[str, str]]]
    hovertext: List[str]
    mode: str
    name: str
    def __init__(self, x: List[float], y: List[float], hovertext: List[str],
        sharpe_ratios: List[float]):        
        self.x, self.y, self.hovertext = x, y, hovertext
        self.marker = {"color": sharpe_ratios, "showscale": True, "size": 7,
            "line":{"width": 1}, "colorscale": "RdGy", "colorbar": {
                "title":'Sharpe<br>Ratio'}}
        self.mode, self.name = 'markers', 'Random Portfolios'

class Point:
    name: str
    mode: str
    x: List[float]
    y: List[float]
    marker: Dict[str, Union[str, int, Dict[str, Union[int, str]]]]
    hovertext: str
    def __init__(self, portfolio: Portfolio, color: str) -> None:
        self.name, self.mode = portfolio.name, 'markers'
        self.x, self.y = [portfolio.std_dev], [portfolio.p_return]
        self.marker = {"color": 'white', "size": 14, "line": {
            "width": 3, "color": color}}
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
    def __init__(self, trading_days: int) -> None:
        self.title = 'Portfolio Optimization: Risk, Return Simulator'
        if trading_days == 252:
            title = 'Annualized'
        else:
            title = '{}-Day'.format(trading_days)
        self.yaxis = {"title": '{} Return'.format(title), "tickformat": ',.0%'}
        self.xaxis = {"title": '{} Risk (Standard Deviation)'.format(title),
            "tickformat": ',.0%'}
        self.showlegend, self.legend = True, {"orientation": "h",
            "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
        self.width, self.height = 800, 600

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

class OptimizeArgs:
    fun: Union[Callable, Callable]
    name: str
    constraints: Tuple[Dict[str, Union[str, Callable[[NDArray], float]]]]
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    trading_days: int
    risk_free_rate: float
    def __init__(self, cov_matrix: pd.DataFrame, trading_days: int,
        mean_returns: pd.Series, risk_free_rate: float, name: Optional[str],
        max_sharpe: Optional[bool], target_return: Optional[float],
        target_std_dev: Optional[float]) -> None:
        is_sharpe_optimization: bool= bool((max_sharpe or target_std_dev))
        if is_sharpe_optimization:
            self.fun = get_neg_sharpe_ratio
        else:
            self.fun = get_std_dev_p
        self.name = name
        self.constraints = Constraints(mean_returns, cov_matrix, trading_days,
            target_return, target_std_dev).__dict__.values()
        if is_sharpe_optimization:
            self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.trading_days = trading_days
        if is_sharpe_optimization:
            self.risk_free_rate = risk_free_rate

class FrontierTraces:
    rand_portfolios: Scatter
    curve: Scatter
    sharpe_ratio_marker: Scatter
    std_dev_marker: Scatter
    def __init__(self, rand_points: RandomPortfolios, curve: Lines,
        min_point: Point, max_point: Point) -> None:
        self.rand_portfolios = Scatter(**rand_points.__dict__)
        self.curve = Scatter(**curve.__dict__)
        self.sharpe_ratio_marker = Scatter(**max_point.__dict__)
        self.std_dev_marker = Scatter(**min_point.__dict__)

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
        bound: Tuple[float, float]=(0, 1), trading_days: int=252) -> None:
        self.trading_days = trading_days
        self.asset_len = len(adjusted_close.columns)
        percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.cov_matrix = percent_change.cov()
        self.risk_free_rate, self.bound = risk_free_rate, bound
        self.max_sharpe_p = self.predict(max_sharpe=True,
            name='Maximum Sharpe Ratio')
        self.min_risk_p = self.predict(min_risk=True, name='Minimum Risk')
        self.fig = self.__plot_frontier_curve()
    def __repr__(self) -> str:
        res: List= []
        for p in (self.max_sharpe_p,  self.min_risk_p):
            res.append(p.__repr__())
        return '\n'.join(res)
    def __optimize_portfolio(self, fun: Callable, name: Optional[str],
        constraints:  Tuple[Dict[str, Union[str, Callable[[NDArray[np.float64
                ]], float]]]],
        *args: Union[pd.Series, pd.DataFrame, int, float]) -> Portfolio:
        bounds: Tuple[Tuple[float, float]]= tuple(self.bound for i in range(
            self.asset_len))
        initial_weights: List[float]= self.asset_len*[1/self.asset_len]
        opt_res: OptimizeResult= minimize(fun, initial_weights, args=args,
            method='SLSQP', bounds=bounds, constraints=constraints)
        return Portfolio(opt_res.x, self.mean_returns, self.cov_matrix,
            self.trading_days, self.risk_free_rate, name=name)
    def __get_frontier_returns(self, n: int=20) -> NDArray[np.float]:
        return np.linspace(self.min_risk_p.p_return,
            self.max_sharpe_p.p_return, n)
    def __get_frontier_std_devs_hover_text(self, frontier_returns: NDArray[
            np.float64]) -> Tuple[List[float], List[str]]:
        frontier_std_devs, hover_text = [], []
        for r in frontier_returns:
            p = self.predict(target_return=r)
            frontier_std_devs.append(p.std_dev)
            hover_text.append(p.__repr__(sep='<br>'))
        return frontier_std_devs, hover_text
    def __get_rand_points(self, n = 1500) -> Tuple[List[float], List[float],
            List[str], List[float]]:
        x, y, hovertext, sharpe_ratios = [], [], [], []
        for i in range(n):
            random_weights = np.random.rand(self.asset_len)
            random_weights = random_weights/sum(random_weights)
            p = Portfolio(random_weights, self.mean_returns, self.cov_matrix,
                self.trading_days, self.risk_free_rate)
            x.append(p.std_dev)
            y.append(p.p_return)
            hovertext.append(p.__repr__(sep='<br>'))
            sharpe_ratios.append(p.sharpe_ratio)
        return x, y, hovertext, sharpe_ratios
    def __plot_frontier_curve(self) -> Figure:
        y = self.__get_frontier_returns()
        x, hovertexts = self.__get_frontier_std_devs_hover_text(y)
        data = list(FrontierTraces(RandomPortfolios(*self.__get_rand_points()),
                Lines(x, y, hovertexts), Point(self.min_risk_p, 'red'),
                Point(self.max_sharpe_p, 'black')).__dict__.values())
        layout = Layout(**FrontierLayout(self.trading_days).__dict__)
        return Figure(data=data, layout=layout)
    def predict(self, target_return: Optional[float]=None, 
        target_std_dev: Optional[float]=None, max_sharpe: Optional[bool]=None,
        min_risk: Optional[bool]=None, name: Optional[str]=None) -> Portfolio:
        options = iter(
            [max_sharpe, min_risk, (target_return or target_std_dev)])
        assert any(options) and not any(options), ' '.join(("Options over",
            "loaded: too many or too few options. Target return, risk should",
            "be greater than zero"))
        return self.__optimize_portfolio(*OptimizeArgs(self.cov_matrix,
            self.trading_days, self.mean_returns, self.risk_free_rate, name,
            max_sharpe, target_return, target_std_dev).__dict__.values())