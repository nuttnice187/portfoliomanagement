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
        res.extend(('    Returns: {:.2%}'.format(self.p_return),
            '    Standard Deviation: {:.2%}'.format(self.std_dev),
            '    Sharpe Ratio: {:.4}'.format(self.sharpe_ratio),
            '    Weight Allocation:'))
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

class RandPoints:
    x: List[float]
    y: List[float]
    marker: Dict[str, Union[List[float], bool, int, Dict[str, int], str, 
            Dict[str, str]]]
    hovertext: List[str]
    mode: str
    name: str
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
        trading_days: int, risk_free_rate: float, asset_len: int) -> None:
        self.x, self.y, self.hovertext, ratios = self.__get_rand_points(
            mean_returns, cov_matrix, trading_days, risk_free_rate, asset_len)
        self.marker = {"color": ratios, "showscale": True, "size": 7,
            "line":{"width": 1}, "colorscale": "RdGy", "colorbar": {
                "title":'Sharpe<br>Ratio'}}
        self.mode, self.name = 'markers', 'Random Portfolios'
    def __get_rand_points(self, mean_returns: pd.Series,
        cov_matrix: pd.DataFrame, trading_days: int, risk_free_rate: float,
        asset_len: int, n: int= 1500) -> Tuple[List[float], List[float],
            List[str], List[float]]:
        x, y, hovertext, sharpe_ratios = [], [], [], []
        for i in range(n):
            random_weights = np.random.rand(asset_len)
            random_weights = random_weights/sum(random_weights)
            p = Portfolio(random_weights, mean_returns, cov_matrix,
                trading_days, risk_free_rate)
            x.append(p.std_dev)
            y.append(p.p_return)
            hovertext.append(p.__repr__(sep='<br>'))
            sharpe_ratios.append(p.sharpe_ratio)
        return x, y, hovertext, sharpe_ratios

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

class Curve:
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

class Plot:
    title: str
    yaxis: Dict[str, str]
    xaxis: Dict[str, str]
    showlegend: bool
    legend: Dict[str, Union[float, str, int]]
    width: int
    height: int
    layout: Layout
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

class Optimization:
    fun: Union[Callable, Callable]
    x0: List[float]
    args: Tuple[Union[pd.Series, pd.DataFrame, int, float]]
    method: str
    bounds: Tuple[Tuple[float]]
    constraints: Tuple[Dict[str, Union[str, Callable[[NDArray], float]]]]
    opt_res: OptimizeResult
    portfolio: Portfolio
    def __init__(self, cov_matrix: pd.DataFrame, trading_days: int,
        mean_returns: pd.Series, risk_free_rate: float, asset_len: int,
        name: Optional[str], max_sharpe: Optional[bool],
        target_return: Optional[float], target_std_dev: Optional[float],
        bound: Tuple[float]=(0, 1)) -> None:
        if (max_sharpe or target_std_dev):
            self.fun, self.args = get_neg_sharpe_ratio,  (mean_returns,
                cov_matrix, trading_days, risk_free_rate)
        else:
            self.fun, self.args = get_std_dev_p, (cov_matrix, trading_days)
        self.x0, self.method = asset_len*[1/asset_len], 'SLSQP'
        self.bounds = tuple(bound for i in range(asset_len))
        self.constraints = Constraints(mean_returns, cov_matrix, trading_days,
            target_return, target_std_dev).__dict__.values()
        self.opt_res = minimize(**self.__dict__)
        self.portfolio = Portfolio(self.opt_res.x, mean_returns,
            cov_matrix, trading_days, risk_free_rate, name=name)

class Traces:
    rand_portfolios: Scatter
    curve: Scatter
    sharpe_ratio_marker: Scatter
    std_dev_marker: Scatter
    def __init__(self, rand_points: RandPoints, curve: Curve,
        min_point: Point, max_point: Point) -> None:
        self.rand_portfolios = Scatter(**rand_points.__dict__)
        self.curve = Scatter(**curve.__dict__)
        self.sharpe_ratio_marker = Scatter(**max_point.__dict__)
        self.std_dev_marker = Scatter(**min_point.__dict__)

class TracePlot:
    data: List[Scatter]
    layout: Layout
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
        trading_days: int, risk_free_rate: float, asset_len: int,
        frontier: Tuple[List[float], List[float], List[str]],
        min_risk_p: Portfolio, max_sharpe_p: Portfolio) -> None:
        self.data = list(Traces(RandPoints(mean_returns, cov_matrix,
                    trading_days, risk_free_rate, asset_len),
                Curve(*frontier), Point(min_risk_p, 'red'),
                Point(max_sharpe_p, 'black')).__dict__.values())
        self.layout = Layout(**Plot(trading_days).__dict__)

class EfficientFrontier:
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    risk_free_rate: float
    trading_days: int
    asset_len: int
    max_sharpe_p: Portfolio
    min_risk_p: Portfolio
    fig: Figure
    def __init__(self, adjusted_close: pd.DataFrame, risk_free_rate: float=0.04,
        trading_days: int=252) -> None:
        self.trading_days = trading_days
        self.asset_len = len(adjusted_close.columns)
        percent_change = adjusted_close.pct_change()
        self.mean_returns = percent_change.mean()
        self.cov_matrix = percent_change.cov()
        self.risk_free_rate = risk_free_rate
        self.max_sharpe_p = self.predict(max_sharpe=True,
            name='Maximum Sharpe Ratio')
        self.min_risk_p = self.predict(min_risk=True, name='Minimum Risk')
        self.fig = Figure(**TracePlot(frontier=self.__get_frontier(),
                **self.__dict__).__dict__)
    def __repr__(self) -> str:
        res: List= []
        for p in (self.max_sharpe_p,  self.min_risk_p):
            res.append(p.__repr__())
        return '\n'.join(res)
    def __get_frontier(self, n: int=20) -> Tuple[List[float], List[float], 
        List[str]]:
        std_devs, returns, hover_text = [], [], []
        for r in np.linspace(
            self.min_risk_p.p_return, self.max_sharpe_p.p_return, n):
            p = self.predict(target_return=r)
            std_devs.append(p.std_dev)
            returns.append(p.p_return)
            hover_text.append(p.__repr__(sep='<br>'))
        return std_devs, returns, hover_text
    def predict(self, target_return: Optional[float]=None, 
        target_std_dev: Optional[float]=None, max_sharpe: Optional[bool]=None,
        min_risk: Optional[bool]=None, name: Optional[str]=None) -> Portfolio:
        options = iter(
            [max_sharpe, min_risk, (target_return or target_std_dev)])
        assert any(options) and not any(options), ' '.join(("Options over",
            "loaded: too many or too few options. Target return, risk should",
            "be greater than zero"))
        return Optimization(self.cov_matrix, self.trading_days,
            self.mean_returns, self.risk_free_rate, self.asset_len, name,
            max_sharpe, target_return, target_std_dev).portfolio