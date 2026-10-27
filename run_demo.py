"""
run_demo.py
-----------
SYNTHETIC-DATA SMOKE TEST. Generates random-walk prices for 8 fake stocks
and runs the whole env -> PPO training -> evaluation pipeline in about a
minute, without downloading market data. Use it to check the code is wired
correctly; the real results come from data_collection.py, train_rl_agent.py
and evaluate.py.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import PortfolioTradingEnv, FEATURE_COLS
from evaluate_compare import compute_metrics, compare_strategies, plot_comparison

rng = np.random.default_rng(42)
tickers = [f"STOCK{i}" for i in range(8)]
n_days = 500
dates = pd.bdate_range("2024-01-01", periods=n_days)

rows = []
for tkr in tickers:
    price = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
    df = pd.DataFrame({"date": dates, "ticker": tkr, "close": price})
    df["open"] = df["close"] * (1 + rng.normal(0, 0.002, n_days))
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = rng.integers(1e5, 1e6, n_days)
    df["return"] = df["close"].pct_change()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema_f = df["close"].ewm(span=12).mean()
    ema_s = df["close"].ewm(span=26).mean()
    df["macd"] = ema_f - ema_s
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["volatility_20"] = df["return"].rolling(20).std()
    rows.append(df)

full_df = pd.concat(rows).dropna().reset_index(drop=True)
split_date = dates[-30]  # last ~6 weeks as "test"
train_df = full_df[full_df["date"] < split_date]
test_df = full_df[full_df["date"] >= split_date]

# --- Train a small PPO agent quickly (demo scale, not production scale) ---
env = DummyVecEnv([lambda: PortfolioTradingEnv(train_df, tickers)])
model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
model.learn(total_timesteps=8000)

# --- Evaluate RL agent on test window ---
test_env = PortfolioTradingEnv(pd.concat([train_df.tail(60), test_df]), tickers)
obs, _ = test_env.reset()
values = [test_env.portfolio_value_history[0]]
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    values.append(info["portfolio_value"])
    done = terminated or truncated
rl_series = pd.Series(values[:len(test_df)], index=test_df["date"].values[:len(values)])

# --- Simple equal-weight buy & hold baseline for comparison ---
bh_prices = test_df.pivot(index="date", columns="ticker", values="close")
bh_norm = bh_prices / bh_prices.iloc[0]
bh_series = bh_norm.mean(axis=1) * rl_series.iloc[0]

comparison = compare_strategies({"RL (PPO, demo/synthetic)": rl_series, "Buy & Hold (demo/synthetic)": bh_series})
print("\n=== DEMO comparison table (SYNTHETIC DATA, illustrative only) ===")
print(comparison.round(4))
comparison.to_csv("demo_comparison.csv")
plot_comparison({"RL (PPO, demo/synthetic)": rl_series, "Buy & Hold (demo/synthetic)": bh_series}, "demo_comparison_plot.png")

