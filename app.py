import os
import importlib.util
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Financial Data Explorer", layout="wide")
st.title("Financial Data Explorer")

# Load data_fetcher module from the same folder as this app (Projects)
base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "data_fetcher", str(base.joinpath("data_fetcher.py"))
)
data_fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_fetcher)

ticker = st.text_input("Ticker symbol", "AAPL")
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y", "10y"], index=3)
missing_method = st.selectbox("Missing data handling", ["ffill", "bfill", "interpolate", "drop", "fill"], index=0)

if st.button("Fetch"):
    with st.spinner("Downloading and processing data..."):
        df = data_fetcher.get_historical_prices(ticker, period=period)
    if df.empty:
        st.warning("No data found for the given ticker/period.")
    else:
        # Handle missing data
        df = data_fetcher.clean_missing_data(df, method=missing_method)

        # Add moving averages and returns
        df = data_fetcher.add_moving_averages(df, windows=(20, 50), price_col='Close')
        df['daily_return'] = data_fetcher.calculate_daily_returns(df, price_col='Close')
        df['cumulative_return'] = data_fetcher.calculate_cumulative_returns(df, price_col='Close')

        # Current price and change
        info = data_fetcher.get_current_price_and_change(ticker)

        def fmt_num(x, dp=2):
            try:
                if x is None:
                    return 'N/A'
                if isinstance(x, (int, float)) and abs(x) >= 1:
                    return f"{x:,.{dp}f}"
                if isinstance(x, (int, float)):
                    return f"{x:.{dp}f}"
                return str(x)
            except Exception:
                return str(x)

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Current Price", value=fmt_num(info.get('current_price')))
        col2.metric(label="Previous Close", value=fmt_num(info.get('previous_close')))
        change = info.get('daily_change_percent')
        change_str = f"{change:.2f}%" if change is not None else 'N/A'
        col3.metric(label="Daily % Change", value=change_str)

        # Additional fields
        col4, col5, col6, col7 = st.columns(4)
        col4.metric(label="Open", value=fmt_num(info.get('open')))
        col5.metric(label="Day High", value=fmt_num(info.get('day_high')))
        col6.metric(label="Day Low", value=fmt_num(info.get('day_low')))
        vol = info.get('volume')
        col7.metric(label="Volume", value=f"{int(vol):,}" if isinstance(vol, (int, float)) else 'N/A')

        col8, col9 = st.columns(2)
        col8.metric(label="Bid", value=fmt_num(info.get('bid')))
        col9.metric(label="Ask", value=fmt_num(info.get('ask')))

        st.subheader(f"{ticker} — last rows")
        st.dataframe(df.tail())

        # Price and moving averages chart
        st.subheader("Close Price and Moving Averages")
        ma_plot = df[['Close']].copy()
        if 'MA20' in df.columns:
            ma_plot['MA20'] = df['MA20']
        if 'MA50' in df.columns:
            ma_plot['MA50'] = df['MA50']
        st.line_chart(ma_plot)

        # Volume chart (bars)
        if 'Volume' in df.columns:
            st.subheader("Volume")
            st.bar_chart(df['Volume'])

        # Returns
        st.subheader("Daily Returns")
        st.line_chart(df['daily_return'].fillna(0))

        st.subheader("Cumulative Returns")
        st.line_chart(df['cumulative_return'].fillna(0))

        # CSV download
        csv = df.to_csv().encode('utf-8')
        st.download_button("Download CSV", csv, file_name=f"{ticker}.csv", mime="text/csv")
