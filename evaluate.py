"""
evaluate.py
-----------
End-to-end evaluation on the held-out test window: runs the trained PPO agent,
the ARIMA and LSTM benchmarks and an equal-weight buy & hold portfolio from the
same starting capital, then writes the metrics table and comparison plot.

Usage:
    python evaluate.py --data data/nifty100_features.parquet --model models/ppo_trader.zip \
        --train-end 2026-06-15 --out results/reproduced
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from benchmarks import arima_signals, lstm_signals, train_lstm
from evaluate_compare import compare_strategies, plot_comparison
from trading_env import PortfolioTradingEnv

INITIAL_CASH = 1_000_000.0


def rl_values(model, train_df, test_df, tickers, window=20, train_norm=False):
    # warm-up: the env starts at index `window`, so prepend exactly `window` training dates
    warm_dates = np.sort(train_df["date"].unique())[-window:]
    env_df = pd.concat([train_df[train_df["date"].isin(warm_dates)], test_df])
    # train_norm=True scales test features with training-period statistics (no look-ahead)
    stats = PortfolioTradingEnv(train_df, tickers, window=window).norm_stats if train_norm else None
    env = PortfolioTradingEnv(env_df, tickers, initial_cash=INITIAL_CASH, window=window, norm_stats=stats)
    obs, _ = env.reset()
    values, done = [env.portfolio_value_history[0]], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        values.append(info["portfolio_value"])
        done = terminated or truncated
    dates = env.dates[window:window + len(values)]
    return pd.Series(values[:len(dates)], index=pd.to_datetime(dates))


def signal_values(signals, test_df, cost=0.001):
    """Daily-rebalanced equal-weight long portfolio over stocks with signal 1 (cash otherwise)."""
    rets = test_df.pivot(index="date", columns="ticker", values="close").pct_change().shift(-1)
    signals = signals.reindex(index=rets.index, columns=rets.columns).fillna(0)
    w = signals.div(signals.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    daily = (w * rets).sum(axis=1) - cost * turnover
    values = INITIAL_CASH * (1 + daily).cumprod().shift(1).fillna(1.0)
    return values.iloc[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nifty100_features.parquet")
    ap.add_argument("--model", default="models/ppo_trader.zip")
    ap.add_argument("--train-end", default="2026-06-15")
    ap.add_argument("--out", default="results/reproduced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-models", nargs="*", default=[], help="more PPO checkpoints to evaluate alongside")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique())
    train_df = df[df["date"] < args.train_end]
    test_df = df[df["date"] >= args.train_end]
    print(f"{len(tickers)} tickers | train {train_df['date'].min().date()}..{train_df['date'].max().date()} "
          f"| test {test_df['date'].min().date()}..{test_df['date'].max().date()} ({test_df['date'].nunique()} days)")

    model = PPO.load(args.model, device="cpu")
    series = {"RL (PPO)": rl_values(model, train_df, test_df, tickers),
              "RL (PPO, train-stat scaling)": rl_values(model, train_df, test_df, tickers, train_norm=True)}
    for path in args.extra_models:
        name = os.path.splitext(os.path.basename(path))[0]
        series[f"RL ({name})"] = rl_values(PPO.load(path, device="cpu"), train_df, test_df, tickers, train_norm=True)

    print("ARIMA: fitting rolling one-step forecasts ...")
    series["ARIMA"] = signal_values(arima_signals(train_df, test_df, tickers), test_df)

    print("LSTM: training ...")
    lstm = train_lstm(train_df, tickers)
    series["LSTM"] = signal_values(lstm_signals(lstm, test_df, df, tickers), test_df)

    closes = test_df.pivot(index="date", columns="ticker", values="close")
    series["Buy & Hold (EW)"] = INITIAL_CASH * (closes / closes.iloc[0]).mean(axis=1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table = compare_strategies(series)
    table.to_csv(f"{args.out}_comparison.csv")
    plot_comparison(series, f"{args.out}_comparison_plot.png")
    print(table.round(4))


if __name__ == "__main__":
    main()
