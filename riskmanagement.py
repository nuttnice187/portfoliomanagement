import pandas as pd

class EfficientFrontierModel:
    mean_returns: pd.Series
    covariance_matrix: pd.DataFrame
    def __init__(self, adjusted_close: pd.DataFrame):
        returns = adjusted_close.pct_change()
        self.mean_returns = returns.mean()
        self.covariance_matrix = returns.cov()
