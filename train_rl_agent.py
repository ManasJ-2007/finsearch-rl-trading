"""
train_rl_agent.py
------------------
Trains a PPO agent on the PortfolioTradingEnv using historical Nifty100
data, then saves the model for later evaluation on the held-out 6-week
test window.

Usage:
    python train_rl_agent.py --data data/nifty100_features.parquet \
        --train-end 2026-06-01 --timesteps 200000 --out models/ppo_trader.zip

PPO is used (rather than DDPG) because the action space here is a
softmax-normalized continuous vector over many assets, where PPO's
clipped-objective tends to be more stable than DDPG's deterministic
policy + replay buffer during early training. DDPG is left in as an
easy swap (see commented alternative below) if you want to compare
on-policy vs off-policy results in the write-up.
"""

import argparse

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import PortfolioTradingEnv


def load_split(path, train_end):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, parse_dates=["date"])
    tickers = sorted(df["ticker"].unique())
    train_df = df[df["date"] < train_end]
    test_df = df[df["date"] >= train_end]
    return train_df, test_df, tickers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nifty100_features.parquet")
    ap.add_argument("--train-end", default="2026-06-15")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", default="models/ppo_trader.zip")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-tensorboard", action="store_true")
    args = ap.parse_args()

    train_df, _, tickers = load_split(args.data, args.train_end)

    env = DummyVecEnv([lambda: PortfolioTradingEnv(train_df, tickers)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log=None if args.no_tensorboard else "./tb_logs/",
        seed=args.seed,
    )

    # --- DDPG alternative (swap in if you want an off-policy comparison) ---
    # from stable_baselines3 import DDPG
    # from stable_baselines3.common.noise import NormalActionNoise
    # import numpy as np
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
    # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved trained model -> {args.out}")


if __name__ == "__main__":
    main()
