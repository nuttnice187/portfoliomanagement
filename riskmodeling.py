import pandas as pd

class EfficientFrontier:
    def __init__(self, adjusted_close: pd.DataFrame):
        self.returns: pd.DataFrame= adjusted_close.pct_change()
        self.mean_returns = self.returns.mean()
        self.covariance_matric = self.returns.cov()
