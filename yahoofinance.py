import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union, Optional

def get_daily_yf(symbols: Union[str|List[str]], start: Optional[datetime]=None,
    end: Optional[datetime]=None, column: str='Adj Close') -> pd.DataFrame:
    if not end:
        end: datetime= datetime.today()
    if not start:
        start: datetime= end - timedelta(days=365)
    if not isinstance(symbols, list):
        symbols: List[str]= [symbols]
    res_df = pd.DataFrame()
    for s in symbols:
        symbol_df: pd.DataFrame= yf.download(s, start=start, end=end)
        res_df[s] = symbol_df[column]        
    return res_df
