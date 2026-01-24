import pandas as pd
import yfinance as yf
from typing import Any, Dict


def get_historical_prices(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download historical prices for `ticker`.

    Returns a pandas DataFrame indexed by Timestamp with columns Open/High/Low/Close/Adj Close/Volume.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None:
        return pd.DataFrame()

    # If yfinance returns MultiIndex columns (e.g., when multiple tickers), flatten to first level
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

    df.index = pd.to_datetime(df.index)
    return df


def get_current_price_and_change(ticker: str) -> Dict[str, Any]:
    """Return the latest close price and daily percent change for `ticker`.

    Uses the last two daily closes to compute the percent change. Returns an empty dict if data
    cannot be retrieved.
    """
    # fetch last 2 daily closes (reliable fallback)
    df = yf.download(ticker, period="2d", interval="1d", progress=False)
    if df is None or df.empty:
        # still attempt to get some info via ticker.info
        df = pd.DataFrame()

    closes = df['Close'].dropna() if 'Close' in df.columns else pd.Series(dtype=float)
    # If closes is a DataFrame (unexpected), pick first column
    if isinstance(closes, pd.DataFrame):
        if closes.shape[1] > 0:
            closes = closes.iloc[:, 0].dropna()
        else:
            closes = pd.Series(dtype=float)

    # ensure numeric; handle non-iterable values defensively
    try:
        closes = pd.to_numeric(closes, errors='coerce').dropna()
    except TypeError:
        # closes may be a scalar or non-iterable; wrap into Series
        try:
            closes = pd.Series([closes])
            closes = pd.to_numeric(closes, errors='coerce').dropna()
        except Exception:
            closes = pd.Series(dtype=float)

    def _to_float_safe(x):
        if x is None:
            return None
        try:
            # numpy scalar or pandas scalar
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)
        except Exception:
            try:
                return float(pd.Series([x]).iloc[0])
            except Exception:
                return None

    if len(closes) == 0:
        last = None
        prev = None
    else:
        last = _to_float_safe(closes.iloc[-1])
        prev = _to_float_safe(closes.iloc[-2]) if len(closes) > 1 else last
    pct = ((last - prev) / prev * 100.0) if (prev is not None and prev != 0) else None

    # Attempt to enrich using ticker.info (may be empty for some tickers)
    bid = ask = volume = open_p = day_high = day_low = None
    try:
        tk = yf.Ticker(ticker)
        info = getattr(tk, 'info', {}) or {}
        # try common keys
        bid = info.get('bid') or info.get('bidPrice')
        ask = info.get('ask') or info.get('askPrice')
        volume = info.get('volume') or info.get('regularMarketVolume')
        open_p = info.get('open') or info.get('regularMarketOpen')
        day_high = info.get('dayHigh') or info.get('regularMarketDayHigh')
        day_low = info.get('dayLow') or info.get('regularMarketDayLow')
    except Exception:
        # ignore enrichment errors
        pass

    return {
        'ticker': ticker,
        'current_price': last,
        'previous_close': prev,
        'daily_change_percent': pct,
        'bid': bid,
        'ask': ask,
        'volume': volume,
        'open': open_p,
        'day_high': day_high,
        'day_low': day_low,
    }


def raw_to_dataframe(raw: Any) -> pd.DataFrame:
    """Convert raw data into a pandas DataFrame.

    Accepts:
    - pandas DataFrame (returns a copy)
    - dict (creates DataFrame from dict)
    - list of dicts
    - single record (wrapped in list)
    """
    if isinstance(raw, pd.DataFrame):
        return raw.copy()

    # list of records
    if isinstance(raw, list):
        try:
            return pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame([raw])

    # dict-like
    if isinstance(raw, dict):
        try:
            return pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame([raw])

    # fallback: wrap single value
    return pd.DataFrame([raw])


if __name__ == '__main__':
    # Quick demo when run directly
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    args = parser.parse_args()

    print('Fetching current price...')
    info = get_current_price_and_change(args.ticker)
    print(info)

    print('\nFetching historical (1mo, daily)...')
    hist = get_historical_prices(args.ticker, period='1mo')
    print(hist.tail())


def clean_missing_data(df: pd.DataFrame, method: str = 'ffill', fill_value: Any = None) -> pd.DataFrame:
    """Handle missing data in-place semantics:

    - method: 'ffill', 'bfill', 'interpolate', 'drop', or 'fill'
    - fill_value: used when method == 'fill'
    Returns a new DataFrame.
    """
    df = df.copy()
    if method == 'ffill':
        return df.ffill()
    if method == 'bfill':
        return df.bfill()
    if method == 'interpolate':
        return df.interpolate()
    if method == 'drop':
        return df.dropna()
    if method == 'fill':
        return df.fillna(fill_value)
    # default
    return df


def calculate_daily_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.Series:
    """Calculate daily percentage returns from price column.

    Returns a pandas Series indexed as the input DataFrame.
    """
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found")
    prices = df[price_col].astype(float)
    returns = prices.pct_change()
    return returns


def calculate_cumulative_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.Series:
    """Calculate cumulative returns from price column.

    Uses (1 + daily_return).cumprod() - 1
    """
    daily = calculate_daily_returns(df, price_col=price_col).fillna(0.0)
    cum = (1.0 + daily).cumprod() - 1.0
    return cum


def add_moving_averages(df: pd.DataFrame, windows=(20, 50), price_col: str = 'Close') -> pd.DataFrame:
    """Add moving average columns for the given windows.

    Columns are named `MA{window}` (e.g., `MA20`). Returns a new DataFrame.
    """
    df = df.copy()
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found")
    for w in windows:
        col = f'MA{int(w)}'
        df[col] = df[price_col].rolling(window=int(w), min_periods=1).mean()
    return df


def format_for_visualization(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """Return a cleaned, visualization-friendly DataFrame.

    - Ensures a `Date` column (reset index if index is datetime)
    - Keeps numeric columns (`price_col`, moving averages), and fills small gaps
    - Adds `daily_return` and `cumulative_return` columns
    """
    df2 = raw_to_dataframe(df) if not isinstance(df, pd.DataFrame) else df.copy()

    # If datetime index, move to column
    if isinstance(df2.index, pd.DatetimeIndex):
        df2 = df2.reset_index().rename(columns={'index': 'Date'})
    # ensure Date column exists
    if 'Date' not in df2.columns and 'date' in df2.columns:
        df2 = df2.rename(columns={'date': 'Date'})

    # Normalize Date column
    if 'Date' in df2.columns:
        df2['Date'] = pd.to_datetime(df2['Date'])

    # Handle missing numeric values with forward-fill then back-fill
    numeric_cols = df2.select_dtypes(include=['number']).columns.tolist()
    df2[numeric_cols] = df2[numeric_cols].ffill().bfill()

    # Add returns and MAs
    if price_col in df2.columns:
        df2 = df2.set_index('Date') if 'Date' in df2.columns else df2
        df2 = add_moving_averages(df2, windows=(20, 50), price_col=price_col)
        df2['daily_return'] = calculate_daily_returns(df2, price_col=price_col)
        df2['cumulative_return'] = calculate_cumulative_returns(df2, price_col=price_col)
        if 'Date' in df2.columns:
            df2 = df2.reset_index()
    return df2

