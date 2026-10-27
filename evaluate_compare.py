"""
evaluate_compare.py
--------------------
Runs the trained RL agent and both benchmark strategies on the same
held-out test window (the "last 6 weeks" required by the assignment),
computes return + risk metrics for each, and produces the comparison
table/plot for the report.

Metrics: cumulative return, annualized return, annualized volatility,
Sharpe ratio, Sortino ratio, max drawdown, 95% historical VaR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252


def compute_metrics(portfolio_values: pd.Series, rf: float = 0.06) -> dict:
    """portfolio_values: daily portfolio value series (not returns)."""
    returns = portfolio_values.pct_change().dropna()
    n_days = len(returns)
    cum_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    ann_return = (1 + cum_return) ** (TRADING_DAYS / max(n_days, 1)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    sharpe = (excess.mean() / (returns.std() + 1e-9)) * np.sqrt(TRADING_DAYS)

    downside = returns[returns < daily_rf]
    downside_std = downside.std() if len(downside) else 1e-9
    sortino = (excess.mean() / (downside_std + 1e-9)) * np.sqrt(TRADING_DAYS)

    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1
    max_dd = drawdown.min()

    var_95 = np.percentile(returns, 5)

    return {
        "cumulative_return": cum_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "var_95_daily": var_95,
    }


def compare_strategies(value_series: dict[str, pd.Series]) -> pd.DataFrame:
    """value_series: {"RL (PPO)": series, "ARIMA": series, "LSTM": series, "Buy&Hold": series}"""
    rows = {name: compute_metrics(s) for name, s in value_series.items()}
    return pd.DataFrame(rows).T


def plot_comparison(value_series: dict[str, pd.Series], out_path="comparison_plot.png"):
    plt.figure(figsize=(10, 5))
    for name, s in value_series.items():
        normalized = s / s.iloc[0]
        plt.plot(normalized.index, normalized.values, label=name)
    plt.title("Portfolio value over the test window (normalized to 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Growth of ₹1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    print(
        "Import compute_metrics / compare_strategies / plot_comparison into your "
        "own evaluation notebook once you have the RL, ARIMA, and LSTM portfolio "
        "value series from a real run — see README.md for the end-to-end flow."
    )

