import pandas as pd

from typing import Tuple, List
from yfinance import download


def count_consecutive_same_color(df: pd.DataFrame, color_col: str) -> pd.DataFrame:
    """Counts consecutive days with the same candle color."""
    count = 0
    color_streak = []
    for i in range(len(df)):
        if i == 0:
            count = 0
        elif df[color_col].iloc[i] == df[color_col].iloc[i-1] and df[color_col].iloc[i] != '':
            count += 1
        else:
            count = 0
        color_streak.append(count)
    df['{}_streak'.format(color_col.split('_')[0])] = color_streak
    return df
    
    
def get_close_prices(symbol: str, prices: pd.DataFrame) -> pd.DataFrame:
    close_prices = prices['Close'].reset_index()
    close_prices[f'{symbol}_pct_change'] = close_prices['Close'].pct_change()
    close_prices[f'{symbol}_candle_color'] = close_prices[f'{symbol}_pct_change'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else ''))
    close_prices = count_consecutive_same_color(close_prices, f'{symbol}_candle_color')
    close_prices['lead_color'] = close_prices[f'{symbol}_candle_color'].shift(-1)
    return close_prices


def get_symbol_markov(symbol: str, close_prices: pd.DataFrame) -> pd.DataFrame:
    symbol_markov = close_prices[[f'{symbol}_candle_color', f'{symbol}_streak', 'lead_color']].copy()
    symbol_markov = symbol_markov[symbol_markov[f'{symbol}_candle_color'] != ''].copy()
    return symbol_markov


def get_prediction_row(
        symbol_markov: pd.DataFrame,
        close_prices: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
    lead_color_counts = symbol_markov.groupby([f'{symbol}_candle_color', f'{symbol}_streak', 'lead_color']).size().unstack(fill_value=0).reset_index()
    lead_color_counts['prediction'] = lead_color_counts.apply(lambda row: 'green' if row.get('green', 0) > row.get('red', 0) else 'red', axis=1)

    last_row = close_prices.iloc[-1]
    last_color = last_row[f'{symbol}_candle_color']
    last_streak = last_row[f'{symbol}_streak']

    return lead_color_counts[(lead_color_counts[f'{symbol}_candle_color'] == last_color) & (lead_color_counts[f'{symbol}_streak'] == last_streak)]


def get_symbols(*symbols: str) -> List[str]:
    """
    Returns a subset of symbols where the prediction for the most recent date is green.

    Args:
        symbols: A list of ticker symbols.

    Returns:
        A list of ticker symbols with a green prediction on the most recent date.
    """
    result = []
    for symbol in symbols:
        prices = download(symbol, period='2y', multi_level_index=False)
        close_prices = get_close_prices(symbol, prices)

        symbol_markov = get_symbol_markov(symbol, close_prices)

        prediction_row = get_prediction_row(symbol_markov, close_prices, symbol)

        if not prediction_row.empty:
            recent_prediction = prediction_row['prediction'].iloc[0]
        else:
            recent_prediction = 'unknown'

        if recent_prediction == 'green':
            result.append(symbol)

    return result
