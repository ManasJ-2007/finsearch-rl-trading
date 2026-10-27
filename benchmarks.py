"""
benchmarks.py
-------------
Two classical/deep-learning benchmark strategies to compare against the
RL agent, evaluated on the same test window and same starting capital:

1. ARIMA benchmark: per-stock ARIMA(p,d,q) on log-price, forecast next-day
   price, go long if forecast > current price else flat. Equal-weight
   across all stocks with a positive signal on a given day.

2. LSTM benchmark: a small sequence model predicts next-day return from
   the last `lookback` days of the same feature set the RL agent sees.
   Trading rule: go long the top-k stocks by predicted return each day,
   equal-weighted, rebalanced daily.

Both benchmarks intentionally use the *same* trading-rule simplicity
(equal weight, long-only, no leverage) as the RL agent's constraints so
the returns-vs-risk comparison in evaluate_compare.py is apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

from trading_env import FEATURE_COLS


# ---------------------------------------------------------------- ARIMA ---
def arima_signals(train_df: pd.DataFrame, test_df: pd.DataFrame, tickers: list[str],
                   order=(1, 1, 1)) -> pd.DataFrame:
    """Returns a DataFrame indexed by test date, columns=tickers, values in
    {1 (long), 0 (flat)} based on a rolling one-step-ahead ARIMA forecast."""
    signals = {}
    for tkr in tickers:
        hist = train_df[train_df["ticker"] == tkr].sort_values("date")
        fut = test_df[test_df["ticker"] == tkr].sort_values("date")
        prices = list(hist["close"].values)
        sig = []
        for _, row in fut.iterrows():
            try:
                model = ARIMA(np.log(prices[-250:]), order=order)  # rolling window keeps fit fast
                fit = model.fit()
                forecast = np.exp(fit.forecast(steps=1)[0])
                sig.append(1 if forecast > prices[-1] else 0)
            except Exception:
                sig.append(0)
            prices.append(row["close"])
        signals[tkr] = pd.Series(sig, index=fut["date"].values)
    return pd.DataFrame(signals)


# ----------------------------------------------------------------- LSTM ---
class ReturnLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(df: pd.DataFrame, ticker: str, lookback: int = 20):
    sub = df[df["ticker"] == ticker].sort_values("date")
    feats = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
    target = sub["return"].shift(-1).to_numpy(dtype=np.float32)  # next-day return
    X, y, dates = [], [], []
    for i in range(lookback, len(sub) - 1):
        X.append(feats[i - lookback:i])
        y.append(target[i])
        dates.append(sub["date"].values[i])
    return np.array(X), np.array(y), np.array(dates)


def train_lstm(train_df: pd.DataFrame, tickers: list[str], lookback: int = 20,
                epochs: int = 15, device: str = "cpu") -> ReturnLSTM:
    Xs, ys = [], []
    for tkr in tickers:
        X, y, _ = make_sequences(train_df, tkr, lookback)
        if len(X):
            Xs.append(X)
            ys.append(y)
    X = torch.tensor(np.concatenate(Xs), dtype=torch.float32, device=device)
    y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=device)

    model = ReturnLSTM(n_features=len(FEATURE_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n = len(X)
    batch_size = 256
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"[LSTM] epoch {epoch+1}/{epochs} loss={total_loss/n:.6f}")
    return model


def lstm_signals(model: ReturnLSTM, test_df: pd.DataFrame, full_df: pd.DataFrame,
                  tickers: list[str], lookback: int = 20, top_k: int = 10,
                  device: str = "cpu") -> pd.DataFrame:
    """Predict next-day return per stock per test date, go long the top_k."""
    preds = {}
    for tkr in tickers:
        X, y, dates = make_sequences(full_df[full_df["ticker"] == tkr], tkr, lookback)
        test_dates = set(test_df["date"].values)
        mask = np.isin(dates, list(test_dates))
        if mask.sum() == 0:
            continue
        with torch.no_grad():
            p = model(torch.tensor(X[mask], dtype=torch.float32, device=device)).cpu().numpy()
        preds[tkr] = pd.Series(p, index=dates[mask])
    pred_df = pd.DataFrame(preds)
    signal_df = pd.DataFrame(0, index=pred_df.index, columns=pred_df.columns)
    for date, row in pred_df.iterrows():
        top = row.nlargest(top_k).index
        signal_df.loc[date, top] = 1
    return signal_df

