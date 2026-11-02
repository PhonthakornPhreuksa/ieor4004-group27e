"""Step 1: download data.

Downloads adjusted daily closes for the NASDAQ-100 constituents and QQQ,
plus the 3-month Treasury yield (FRED DGS3MO). Saves CSVs to ../data.
Run from anywhere:  python src/get_data.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

START = "2023-09-29"   # one day before the first return date (Oct 2, 2023)
END = "2026-10-01"     # yfinance end date is exclusive -> last date Sep 30, 2026


def main():
    tickers = pd.read_csv(DATA / "constituents.csv")["ticker"].tolist()
    px = yf.download(tickers + ["QQQ"], start=START, end=END,
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px[tickers].to_csv(DATA / "prices.csv")
    px[["QQQ"]].to_csv(DATA / "qqq.csv")

    # Risk-free rate (annualized %, used for evaluation only)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    try:
        rf = pd.read_csv(url, parse_dates=[0], index_col=0, na_values=".")
        rf.loc[START:END].to_csv(DATA / "dgs3mo.csv")
    except Exception as e:  # FRED blocked or offline
        print("Could not download DGS3MO:", e)
    print("Saved", px.shape, "prices to", DATA)


if __name__ == "__main__":
    main()
