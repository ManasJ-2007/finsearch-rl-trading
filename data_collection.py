"""
data_collection.py
-------------------
Downloads OHLCV data for Nifty100 constituents from Yahoo Finance and
builds the technical-indicator feature set used by the RL environment
and the ARIMA/LSTM benchmarks.


Output: a single long-format parquet/csv with columns
    date, ticker, open, high, low, close, volume,
    return, ma_10, ma_50, rsi_14, macd, macd_signal, bb_upper, bb_lower, volatility_20
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


NIFTY100_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "WIPRO.NS",
    "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATASTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "GRASIM.NS", "DRREDDY.NS",
    "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS", "BRITANNIA.NS", "HEROMOTOCO.NS",
    "BPCL.NS", "INDUSINDBK.NS", "TECHM.NS", "APOLLOHOSP.NS", "UPL.NS",
    "BAJAJ-AUTO.NS", "HINDALCO.NS", "SHRIRAMFIN.NS", "PIDILITIND.NS", "DABUR.NS",
]  # 50 large-cap names from the Nifty 100 universe (49 had usable data for the study period)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + num_std * std, ma - num_std * std


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["return"] = df["close"].pct_change()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = rsi(df["close"])
    macd_line, signal_line = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    bb_up, bb_low = bollinger_bands(df["close"])
    df["bb_upper"] = bb_up
    df["bb_lower"] = bb_low
    df["volatility_20"] = df["return"].rolling(20).std()
    return df


def download_all(tickers, start, end, pause=0.4) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i, t in enumerate(tickers):
        try:
            raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                print(f"[warn] no data for {t}")
                continue
            raw = raw.reset_index()
            raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
            raw["ticker"] = t
            frames.append(raw[["date", "ticker", "open", "high", "low", "close", "volume"]])
            print(f"[{i+1}/{len(tickers)}] downloaded {t} ({len(raw)} rows)")
        except Exception as e:
            print(f"[error] {t}: {e}")
        time.sleep(pause)  # be polite to Yahoo's rate limits
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-07-27")
    ap.add_argument("--out", default="data/nifty100_features.parquet")
    ap.add_argument("--tickers", nargs="*", default=NIFTY100_TICKERS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = download_all(args.tickers, args.start, args.end)
    featured = raw.groupby("ticker", group_keys=False).apply(build_features)
    featured = featured.dropna().reset_index(drop=True)

    if args.out.endswith(".parquet"):
        featured.to_parquet(args.out, index=False)
    else:
        featured.to_csv(args.out, index=False)
    print(f"Saved {len(featured)} rows -> {args.out}")


if __name__ == "__main__":
    main()

