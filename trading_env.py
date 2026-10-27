"""
trading_env.py
---------------
A Gymnasium-compatible multi-stock portfolio trading environment.

State:  for each stock -> [return, ma_10, ma_50, rsi_14, macd, macd_signal,
                            bb_upper, bb_lower, volatility_20]  (z-scored)
        plus portfolio-level: [holdings value ratio per stock, cash_ratio]

Action: continuous vector in [-1, 1]^n_stocks, one value per stock.
        A softmax over the actions gives relative target weights, scaled by
        the mean action ("confidence") so the agent can also hold cash; trades
        then move the portfolio to those target weights (long-only, no
        leverage, no shorting: a deliberate simplification).

Reward: change in portfolio value (log return) minus a volatility penalty,
        i.e. a differential-Sharpe-style reward so the agent isn't purely
        return-chasing and picks up some risk awareness during training.
        reward_t = log(V_t / V_{t-1}) - risk_lambda * rolling_std(returns)

The environment is kept simple enough to train in minutes on a CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

FEATURE_COLS = [
    "return", "ma_10", "ma_50", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "volatility_20",
]


class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        risk_lambda: float = 0.1,
        window: int = 20,
    ):
        super().__init__()
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.transaction_cost = transaction_cost
        self.risk_lambda = risk_lambda
        self.window = window
        self.initial_cash = initial_cash

        # Pivot into wide format: one row per date, MultiIndex columns (ticker, feature)
        pv = df.pivot_table(index="date", columns="ticker", values=["close"] + FEATURE_COLS)
        pv = pv.dropna()
        self.dates = pv.index.to_numpy()
        self.close = pv["close"][tickers].to_numpy(dtype=np.float32)
        self.features = np.stack(
            [pv[col][tickers].to_numpy(dtype=np.float32) for col in FEATURE_COLS], axis=-1
        )  # shape (T, n_assets, n_features)

        # Normalize features cross-sectionally to keep the network well-conditioned.
        mu = self.features.mean(axis=(0, 1), keepdims=True)
        sigma = self.features.std(axis=(0, 1), keepdims=True) + 1e-8
        self.features = (self.features - mu) / sigma

        self.n_features = self.features.shape[-1]
        obs_dim = self.n_assets * self.n_features + self.n_assets + 1  # + holdings ratio + cash ratio
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        feat = self.features[self.t].flatten()
        holdings_value = self.shares * self.close[self.t]
        total_value = self.cash + holdings_value.sum()
        holdings_ratio = holdings_value / (total_value + 1e-8)
        cash_ratio = np.array([self.cash / (total_value + 1e-8)], dtype=np.float32)
        return np.concatenate([feat, holdings_ratio, cash_ratio]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value_history = [self.initial_cash]
        self.returns_history = []
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        # softmax over positive part -> long-only target weights that sum to <=1
        exp_a = np.exp(action - action.max())
        target_weights = exp_a / exp_a.sum()
        # scale down target weights by how "confident" (positive) the action was on average
        confidence = np.clip(action.mean() * 0.5 + 0.5, 0.0, 1.0)
        target_weights = target_weights * confidence

        prices = self.close[self.t]
        holdings_value = self.shares * prices
        total_value = self.cash + holdings_value.sum()

        target_value = target_weights * total_value
        current_value = holdings_value
        trade_value = target_value - current_value

        cost = np.abs(trade_value).sum() * self.transaction_cost
        self.cash -= trade_value.sum() + cost
        self.shares = self.shares + trade_value / (prices + 1e-8)
        self.shares = np.clip(self.shares, 0, None)  # long-only

        self.t += 1
        terminated = self.t >= len(self.dates) - 1
        truncated = False

        new_prices = self.close[self.t]
        new_value = self.cash + (self.shares * new_prices).sum()
        prev_value = self.portfolio_value_history[-1]
        log_ret = np.log(max(new_value, 1e-8) / max(prev_value, 1e-8))
        self.portfolio_value_history.append(new_value)
        self.returns_history.append(log_ret)

        recent_vol = np.std(self.returns_history[-self.window:]) if len(self.returns_history) > 1 else 0.0
        reward = log_ret - self.risk_lambda * recent_vol

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"portfolio_value": new_value, "log_return": log_ret}
        return obs, float(reward), terminated, truncated, info

