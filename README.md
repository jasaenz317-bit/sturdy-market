# Financial Data Project

Short description
- Utility scaffold for collecting, processing, and visualizing financial time-series data.
- Includes a Streamlit app, data helpers, and simple tests.

What this contains
- `Projects/data_fetcher.py`: helpers to download and normalize ticker data (historical prices, current price, returns, moving averages, missing-data handling).
- `Projects/app.py`: Streamlit UI that fetches a ticker, displays current price/bid/ask/volume, plots Close with MA20/MA50, daily & cumulative returns, and a Volume bar chart.
- `data/`: place CSV datasets here (ignored by default).
- `tests/`: simple tests (e.g. `tests/test_sample.py`).

Quick start

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# bash / macOS
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app (development mode):

```bash
streamlit run Projects/app.py
```

Notes
- The Streamlit app uses `yfinance` for data and includes simple handling for missing values, daily/cumulative returns, and moving averages.
- Large datasets should be placed in `data/` and are excluded from git by default.
- Update `Projects/data_fetcher.py` if you need additional fields or smoothing options.

Screenshots

- `Financial_Project_Screenshot2` — shown below:

![Financial Project Screenshot 2](Financial_Project_Screenshot2.jpeg)

License
- MIT — see `LICENSE`.

