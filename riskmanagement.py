import pandas as pd

class EfficientFrontierModel:
    returns: pd.DataFrame
    mean_returns: pd.DataFrame
    covariance_matrix: pd.DataFrame
    def __init__(self, adjusted_close: pd.DataFrame):
        self.returns = adjusted_close.pct_change()
        self.mean_returns = self.returns.mean()
        self.covariance_matrix = self.returns.cov()
