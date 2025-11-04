import os
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# your env wrapper
from envs.gym_alpha_env import GymAlphaEnv


# -------------------- helpers --------------------
TRADING_DAYS = 252

def slice_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dt = pd.to_datetime(df["ts"], unit="s")
    mask = (dt >= pd.Timestamp(start)) & (dt < pd.Timestamp(end))
    return df.loc[mask].copy()

def make_env(csv_path: str, log_dir: str, window: int, fee_bps: float):
    os.makedirs(log_dir, exist_ok=True)
    def _ctor():
        return GymAlphaEnv(csv_path, window=window, fee_bps=fee_bps)
    return _ctor

def apply_transaction_costs(position: pd.Series, ret: pd.Series, fee_bps: float) -> pd.Series:
    position = position.fillna(0).astype(float)
    turnover = position.diff().abs().fillna(position.abs())   # cost on changes; first bar if not flat
    fee = turnover * (fee_bps / 1e4)                          # bps -> decimal
    pnl = position * ret - fee                                # same-bar return model
    return pnl

def compute_metrics(equity: pd.Series, daily_ret: pd.Series):
    equity = equity.dropna()
    daily_ret = daily_ret.dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / max(1, len(equity))) - 1 if len(equity) > 0 else np.nan
    vol = daily_ret.std()
    sharpe = (daily_ret.mean() / vol * sqrt(TRADING_DAYS)) if (vol is not None and vol > 0) else np.nan
    roll_max = equity.cummax()
    maxdd = (equity / roll_max - 1.0).min()
    return dict(TotalReturn=total_return, CAGR=cagr, Sharpe=sharpe, MaxDD=maxdd)

def ensure_close(df_slice: pd.DataFrame) -> pd.Series:
    if "close" in df_slice.columns:
        return df_slice["close"]
    raise ValueError("Input CSV must contain a 'close' column.")

def to_timeseries_index(csv_path: str) -> pd.DatetimeIndex:
    ts = pd.read_csv(csv_path)["ts"]
    return pd.to_datetime(ts, unit="s")


# -------------------- main simulation --------------------
def simulate(args):
    # 1) Slice the master CSV to desired 2025 window
    df_full = pd.read_csv(args.csv).sort_values("ts")
    for col in ["ts", "open", "high", "low", "close", "volume"]:
        if col not in df_full.columns:
            raise ValueError("CSV must contain ts,open,high,low,close,volume")
    df_slice = slice_by_dates(df_full, args.start, args.end)
    if len(df_slice) < 10:
        raise ValueError(f"No rows found in range {args.start}..{args.end}")

    os.makedirs(args.outdir, exist_ok=True)
    test_csv = os.path.join(args.outdir, "test_2025.csv")
    df_slice.to_csv(test_csv, index=False)

    # 2) Build eval env and load normalization/model
    raw_env = DummyVecEnv([make_env(test_csv, os.path.join(args.outdir, "logs"), args.window, args.fee_bps)])
    if args.vecnorm and os.path.exists(args.vecnorm):
        env = VecNormalize.load(args.vecnorm, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        env = raw_env

    model = PPO.load(args.model, env=env)

    # 3) Rollout RL policy, gather per-step pnl/equity/pos
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    done = False

    rl_rets, rl_equity, rl_pos = [], [1.0], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        # sb3 compatibility (gym vs gymnasium)
        if len(step_out) == 4:
            next_obs, reward, done_arr, info = step_out
        else:
            next_obs, reward, terminated, truncated, info = step_out
            done_arr = np.array([bool(terminated[0] or truncated[0])])

        info0 = info[0] if isinstance(info, (list, tuple)) and len(info) else info
        r = info0.get("pnl", float(reward[0]))  # prefer true fractional pnl
        rl_rets.append(float(r))
        rl_equity.append(rl_equity[-1] * (1.0 + float(r)))

        # unwrap action -> scalar {-1,0,1}
        if isinstance(action, (np.ndarray, list)):
            a = int(action[0]) if len(np.atleast_1d(action)) else int(action)
        else:
            a = int(action)
        rl_pos.append(a)

        obs = next_obs
        done = bool(done_arr[0])

    # 4) Benchmarks on same window: Buy&Hold, MA crossover
    idx = to_timeseries_index(test_csv)
    close = ensure_close(df_slice).reset_index(drop=True)
    returns = close.pct_change().fillna(0.0)

    # Buy & Hold
    pos_bh = pd.Series(1.0, index=returns.index)
    ret_bh = apply_transaction_costs(pos_bh, returns, args.fee_bps)
    eq_bh = (1 + ret_bh).cumprod()

    # MA crossover
    fast_ma = close.rolling(args.fast).mean()
    slow_ma = close.rolling(args.slow).mean()
    if args.short_when_fast_below:
        sig = np.sign(fast_ma - slow_ma).replace(np.nan, 0.0)  # +1 / -1 / 0
    else:
        sig = (fast_ma > slow_ma).astype(float).fillna(0.0)    # 1 / 0
    pos_ma = sig
    ret_ma = apply_transaction_costs(pos_ma, returns, args.fee_bps)
    eq_ma = (1 + ret_ma).cumprod()

    # RL series
    rl_ret_s = pd.Series(rl_rets, index=returns.index[:len(rl_rets)])
    rl_eq_s  = pd.Series(rl_equity[1:], index=returns.index[:len(rl_rets)])
    rl_pos_s = pd.Series(rl_pos, index=returns.index[:len(rl_pos)])

    # 5) Metrics
    metrics = pd.DataFrame([
        {"Strategy":"RL",        **compute_metrics(rl_eq_s, rl_ret_s)},
        {"Strategy":"Buy&Hold",  **compute_metrics(eq_bh,  ret_bh)},
        {"Strategy":f"MA{args.fast}/{args.slow}", **compute_metrics(eq_ma,  ret_ma)},
    ]).set_index("Strategy").round(4)

    # 6) Save timeseries + metrics
    out_ts = pd.DataFrame({
        "Date": idx[:len(returns)],
        "Close": close.values,
        "ret_rl": rl_ret_s.reindex(returns.index),
        "ret_bh": ret_bh,
        "ret_ma": ret_ma,
        "equity_rl": rl_eq_s.reindex(returns.index),
        "equity_bh": eq_bh,
        "equity_ma": eq_ma,
        "pos_rl": rl_pos_s.reindex(returns.index),
        "pos_bh": pos_bh,
        "pos_ma": pos_ma,
    })
    ts_path = os.path.join(args.outdir, "strategy_comparison_timeseries.csv")
    m_path  = os.path.join(args.outdir, "strategy_comparison_metrics.csv")
    out_ts.to_csv(ts_path, index=False)
    metrics.to_csv(m_path)

    print("\n=== Metrics (", args.start, "→", args.end, ") ===")
    print(metrics)
    print(f"\nSaved: {ts_path}\nSaved: {m_path}")

    # 7) Optional animation
    if args.animate:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            print("Rendering animation…")

            dates = out_ts["Date"]
            price = out_ts["Close"]
            eq_rl = out_ts["equity_rl"]
            eq_bh = out_ts["equity_bh"]
            eq_ma = out_ts["equity_ma"]
            pos_rl = out_ts["pos_rl"].fillna(0)

            fig = plt.figure(figsize=(10,6))
            ax_price = plt.subplot2grid((3,1),(0,0), rowspan=2)
            ax_eq = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=ax_price)

            l_price, = ax_price.plot([], [], lw=1.2, label="Close")
            l_rl,    = ax_eq.plot([], [], lw=1.2, label="Equity RL")
            l_bh,    = ax_eq.plot([], [], lw=1.0, label="Equity B&H")
            l_ma,    = ax_eq.plot([], [], lw=1.0, label=f"Equity MA{args.fast}/{args.slow}")

            ax_price.set_title("Trading Animation — RL vs Buy&Hold vs MA")
            ax_price.set_ylabel("Price")
            ax_eq.set_ylabel("Equity")
            ax_eq.set_xlabel("Date")
            ax_price.legend(loc="upper left")
            ax_eq.legend(loc="upper left")

            N = len(out_ts)
            def _pad(y):
                y = pd.Series(y).dropna()
                if len(y)==0: return (0.9,1.1)
                lo, hi = float(y.min()), float(y.max())
                pad = max(1e-6, (hi-lo)*0.1)
                return lo-pad, hi+pad

            def init():
                ax_price.set_xlim(dates.iloc[0], dates.iloc[min(N-1, 100)])
                ylo, yhi = _pad(price.iloc[:100])
                ax_price.set_ylim(ylo,yhi)
                ax_eq.set_xlim(dates.iloc[0], dates.iloc[min(N-1, 100)])
                ylo2, yhi2 = _pad(eq_rl.iloc[:100])
                ax_eq.set_ylim(ylo2,yhi2)
                return l_price, l_rl, l_bh, l_ma

            def update(i):
                win = 250
                s = max(0, i - win)
                x = dates.iloc[s:i+1]
                l_price.set_data(x, price.iloc[s:i+1])
                l_rl.set_data(x, eq_rl.iloc[s:i+1])
                l_bh.set_data(x, eq_bh.iloc[s:i+1])
                l_ma.set_data(x, eq_ma.iloc[s:i+1])

                p_lo, p_hi = _pad(price.iloc[s:i+1])
                ax_price.set_xlim(dates.iloc[s], dates.iloc[max(s+10, i)])
                ax_price.set_ylim(p_lo, p_hi)

                e_lo, e_hi = _pad(pd.concat([eq_rl.iloc[s:i+1], eq_bh.iloc[s:i+1], eq_ma.iloc[s:i+1]]))
                ax_eq.set_xlim(dates.iloc[s], dates.iloc[max(s+10, i)])
                ax_eq.set_ylim(e_lo, e_hi)

                # position shading for RL
                for coll in list(ax_price.collections):
                    coll.remove()
                seg_pos = pos_rl.iloc[s:i+1]
                ax_price.fill_between(x, price.iloc[s:i+1].min(), price.iloc[s:i+1],
                                      where=(seg_pos>0), alpha=0.08, step="pre")
                ax_price.fill_between(x, price.iloc[s:i+1].min(), price.iloc[s:i+1],
                                      where=(seg_pos<0), alpha=0.08, step="pre")

                return l_price, l_rl, l_bh, l_ma

            ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, interval=40, blit=False)
            mp4_path = os.path.join(args.outdir, "trading_2025_animation.mp4")
            try:
                ani.save(mp4_path, writer="ffmpeg", fps=30, dpi=160, bitrate=2400)
                print(f"Saved animation: {mp4_path}")
            except Exception as e:
                print(f"FFmpeg not available ({e}). Showing window instead.")
                plt.show()
        except Exception as e:
            print(f"Animation failed: {e}")
        os.makedirs("report", exist_ok=True)
        ani.save("report/benchmark.gif", writer="pillow", fps=20)


# -------------------- cli --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simulate 2025 window: RL vs Buy&Hold vs MA crossover (with optional animation).")
    p.add_argument("--csv", required=True, help="Master CSV (ts,open,high,low,close,volume).")
    p.add_argument("--fold_dir", required=True, help="Folder containing ppo_agent_walk.zip and optionally vecnorm.pkl.")
    p.add_argument("--start", default="2025-01-01", help="Start date (inclusive).")
    p.add_argument("--end",   default="2025-04-01", help="End date (exclusive).")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--fee_bps", type=float, default=1.0)
    p.add_argument("--fast", type=int, default=10, help="Fast MA window.")
    p.add_argument("--slow", type=int, default=50, help="Slow MA window.")
    p.add_argument("--short_when_fast_below", action="store_true", help="If set, go short when fast<slow; otherwise flat.")
    p.add_argument("--outdir", default="sim_2025_out")
    p.add_argument("--animate", action="store_true", help="Export MP4 animation.")
    args = p.parse_args()

    # resolve model path robustly (avoid .zip.zip)
    base_model = os.path.join(args.fold_dir, "ppo_agent_walk")
    model_path = base_model if os.path.exists(base_model + ".zip") else base_model
    args.model = model_path
    args.vecnorm = os.path.join(args.fold_dir, "vecnorm.pkl")  # optional

    simulate(args)

