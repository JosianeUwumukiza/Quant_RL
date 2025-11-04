import os
import numpy as np
from envs.gym_alpha_env import GymAlphaEnv
from stable_baselines3 import PPO

# --- utility functions ---
def sharpe(returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio for a series of returns."""
    r = np.array(returns)
    if len(r) < 2 or np.std(r) == 0:
        return 0.0
    mean_r = np.mean(r) - risk_free_rate
    std_r = np.std(r)
    sharpe_ratio = np.sqrt(252) * (mean_r / std_r)  # 252 trading days
    return float(sharpe_ratio)

def max_drawdown(equity_curve):
    """Compute maximum drawdown from an equity curve."""
    eq = np.array(equity_curve)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / peak
    return float(np.max(drawdowns))

# --- main evaluation ---
if __name__ == "__main__":
    # Use SPY or AAPL data if you’ve replaced sample_btc_minute.csv
    csv = os.path.join(os.path.dirname(__file__), "data", "sample_btc_minute.csv")
    model_path = "ppo_alpha.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Data not found: {csv}")

    env = GymAlphaEnv(csv, window=32, fee_bps=1.0)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    pnl, eq = [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        pnl.append(reward)
        eq.append(info.get("eq", 0.0))

    sharpe_val = sharpe(pnl)
    mdd_val = max_drawdown(eq)

    print("Sharpe:", round(sharpe_val, 6))
    print("Max Drawdown:", round(mdd_val, 6))
