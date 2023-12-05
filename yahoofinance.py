import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union, Optional

def get_daily_yf(symbols: Union[str|List[str]], start: Optional[datetime.datetime]=None,
    end: Optional[datetime.datetime]=None, column: str='Adj Close') -> pd.DataFrame:
    if not end:
        end: datetime.datetime= datetime.today()
    if not start:
        start: datetime.datetime= end - timedelta(days=2*365)
    if not isinstance(symbols, List[str]):
        symbols: List[str]= [symbols]
    res_df = pd.DataFrame()
    for s in symbols:
        symbol_df = yf.download(s, start=start_date, end=end_date)
        df[s] = symbol_df[column]        
    return res_df
