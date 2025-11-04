import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Utility Functions ---
def rolling_sharpe(returns, window=390):
    r = pd.Series(returns)
    if len(r) < 2:
        return pd.Series(dtype=float)
    m = r.rolling(window).mean()
    s = r.rolling(window).std(ddof=1).replace(0, np.nan)
    return m / s

def max_drawdown(equity):
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = peak - eq
    return dd

def compute_features(df, vol_window=64, z_window=64):
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)
    vol = ret.rolling(vol_window).std().fillna(method="bfill")
    mean = close.rolling(z_window).mean().fillna(method="bfill")
    std = close.rolling(z_window).std(ddof=1).replace(0, np.nan).fillna(method="bfill")
    z = (close - mean) / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    regime = (vol > 0.015).astype(float)
    return close, ret, vol, z, regime

def plot_series(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# --- Model Loader ---
def maybe_load_model(model_path):
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        return model
    except Exception as e:
        print(f"[warn] Could not load model at {model_path}: {e}")
        return None

# --- Environment Loader ---
def make_env(csv_path, window=32, fee_bps=1.0):
    import glob, importlib.util
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cands = [
        os.path.join(ROOT, "build_new", "Release"),
        os.path.join(ROOT, "build", "Release"),
        os.path.join(ROOT),
    ]
    for d in cands:
        for pyd in glob.glob(os.path.join(d, "qrl_bindings*.pyd")):
            spec = importlib.util.spec_from_file_location("qrl_bindings", pyd)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            sys.modules["qrl_bindings"] = mod
            from envs.gym_alpha_env import GymAlphaEnv
            return GymAlphaEnv(csv_path, window=window, fee_bps=fee_bps)
    print("[warn] qrl_bindings not found; evaluation rollout will be skipped.")
    return None

# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Plot data, rollout metrics, and PPO training report.")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "sample_btc_minute.csv"))
    ap.add_argument("--model", default="ppo_alpha.zip")
    ap.add_argument("--outdir", default="report")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--fee_bps", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load Data ---
    if not os.path.exists(args.csv):
        print(f"[error] CSV not found: {args.csv}")
        sys.exit(1)
    df = pd.read_csv(args.csv)
    if "ts" in df.columns:
        x = pd.to_datetime(df["ts"], unit="s")
    else:
        x = pd.RangeIndex(len(df))

    close, ret, vol, z, regime = compute_features(df)
    plot_series(x, close, "Close Price", "Time", "Price", os.path.join(args.outdir, "price.png"))
    plot_series(x, vol, "Rolling Volatility", "Time", "Volatility", os.path.join(args.outdir, "volatility.png"))
    plot_series(x, z, "Z-score of Close", "Time", "Z", os.path.join(args.outdir, "zscore.png"))
    plot_series(x, regime, "High-Vol Regime", "Time", "Regime", os.path.join(args.outdir, "regime.png"))

    # --- Evaluation Rollout ---
    model = maybe_load_model(args.model)
    env = make_env(args.csv, window=args.window, fee_bps=args.fee_bps) if model else None
    pnl_series, eq_series = [], []

    if model and env:
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            pnl_series.append(float(reward))
            eq_series.append(float(info.get("eq", 0.0)))

    # --- Plot Results ---
    if len(eq_series) > 2:
        eq = pd.Series(eq_series)
        dd = max_drawdown(eq)
        plot_series(range(len(eq)), eq, "Equity Curve", "Step", "Equity", os.path.join(args.outdir, "equity.png"))
        plot_series(range(len(dd)), dd, "Drawdown", "Step", "Drawdown", os.path.join(args.outdir, "drawdown.png"))

    if len(pnl_series) > 30:
        rs = rolling_sharpe(pnl_series, window=200)
        plot_series(range(len(rs)), rs, "Rolling Sharpe", "Step", "Sharpe", os.path.join(args.outdir, "rolling_sharpe.png"))

    summary = {
        "csv": os.path.abspath(args.csv),
        "model": os.path.abspath(args.model) if os.path.exists(args.model) else None,
        "num_steps_eval": len(pnl_series),
        "report_dir": os.path.abspath(args.outdir),
        "artifacts": [f for f in os.listdir(args.outdir) if f.endswith(".png")],
    }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print("✅ Report written to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
