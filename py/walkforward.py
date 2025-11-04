import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# ---- local import: your Gym wrapper ----
from envs.gym_alpha_env import GymAlphaEnv


# ======================== Utilities & Metrics ========================

def parse_custom_folds(spec: str):
    """
    Parse "2006-2014:2015,2015-2023:2024,2008-2016:2017-2018"
    -> list of (train_start, train_end, test_start, test_end)
    """
    folds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lhs, rhs = part.split(":")
        t0, t1 = [int(x) for x in lhs.split("-")]
        if "-" in rhs:
            s0, s1 = [int(x) for x in rhs.split("-")]
        else:
            s0 = s1 = int(rhs)
        folds.append((t0, t1, s0, s1))
    return folds


def infer_ann_factor_from_ts(ts_seconds: np.ndarray) -> float:
    """
    Infer annualization factor from median bar spacing.
    - Daily: 252
    - Intraday: 252 * bars_per_day (estimate using 6.5hr US session by default)
    """
    if ts_seconds.size < 3:
        return 252.0
    diffs = np.diff(np.sort(ts_seconds))
    med = float(np.median(diffs))
    # < 1 hour gap => intraday
    if med < 3600:
        bars_per_day = max(1, int(round((6.5 * 3600) / med)))
        return 252.0 * bars_per_day
    return 252.0


def ann_sharpe_from_returns(returns: np.ndarray, ann_factor: float, eps=1e-12) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    mu = r.mean()
    sd = r.std()
    if sd < eps:
        return 0.0
    return float(np.sqrt(ann_factor) * mu / (sd + eps))


def max_drawdown_from_equity(eq: List[float], eps=1e-12) -> float:
    e = np.asarray(eq, dtype=float)
    if e.size < 2:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / np.maximum(peak, 1.0 + eps)
    return float(dd.max())


def make_env(csv_path: str, log_dir: str, window: int = 32, fee_bps: float = 1.0):
    """
    Factory returning a function that builds a Monitor-wrapped GymAlphaEnv.
    """
    def _ctor():
        env = GymAlphaEnv(csv_path, window=window, fee_bps=fee_bps)
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return _ctor


# ======================== Train & Eval per fold ========================

def train_and_eval(
    train_csv: str,
    test_csv: str,
    total_timesteps: int,
    ppo_kwargs: Dict,
    log_root: str,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    window: int = 32,
    fee_bps: float = 1.0,
    warm_start_model: Optional[PPO] = None,
) -> Tuple[float, float, PPO]:
    """
    Train PPO on train_csv, evaluate on test_csv.
    Returns: (Sharpe, MaxDrawdown, trained_model) for warm-start chaining.
    Uses true per-step returns if env exposes info["pnl"]; otherwise falls back to reward.
    """
    fold_dir = log_root
    os.makedirs(fold_dir, exist_ok=True)

    # --- Train env ---
    train_env = DummyVecEnv([make_env(train_csv, log_dir=os.path.join(fold_dir, "train_logs"),
                                      window=window, fee_bps=fee_bps)])
    if normalize_obs or normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=normalize_obs, norm_reward=normalize_reward, clip_obs=10.0)

    if warm_start_model is None:
        model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    else:
        # continue training from prior fold weights
        model = warm_start_model
        model.set_env(train_env)

    logger = configure(fold_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=(warm_start_model is None))

    # ---- SAVE MODEL (as ppo_agent_walk.zip) ----
    model_path = os.path.join(fold_dir, "ppo_agent_walk")
    model.save(model_path)  # SB3 appends .zip

    # Save normalization stats if used
    if isinstance(train_env, VecNormalize):
        vec_path = os.path.join(fold_dir, "vecnorm.pkl")
        train_env.save(vec_path)

    # --- Test env ---
    test_env_raw = DummyVecEnv([make_env(test_csv, log_dir=os.path.join(fold_dir, "test_logs"),
                                         window=window, fee_bps=fee_bps)])
    if isinstance(train_env, VecNormalize):
        test_env = VecNormalize.load(os.path.join(fold_dir, "vecnorm.pkl"), test_env_raw)
        test_env.training = False
        test_env.norm_reward = False
    else:
        test_env = test_env_raw

    # --- Roll out deterministically (VecEnv-safe) ---
    reset_out = test_env.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    done = False

    rets = []          # per-step returns (prefer true pnl)
    equity = [1.0]     # compounded equity if pnl is provided, else additive proxy

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.int64)
        else:
            action = action.reshape(1, *action.shape) if action.ndim == 0 else \
                     action.reshape(1, -1) if action.ndim == 1 else action

        step_out = test_env.step(action)
        # SB3 gym vs gymnasium API compatibility
        if len(step_out) == 4:
            next_obs, reward, done_arr, info = step_out
        else:
            next_obs, reward, terminated, truncated, info = step_out
            done_arr = np.array([bool(terminated[0] or truncated[0])])

        info0 = info[0] if isinstance(info, (list, tuple)) and len(info) else info

        pnl_step = None
        if isinstance(info0, dict):
            # Expose this in your env step for precise Sharpe:
            # info["pnl"] = pos * ret_next - fee   (fractional return per step)
            pnl_step = info0.get("pnl", None)

        if pnl_step is not None:
            r = float(pnl_step)
            equity.append(equity[-1] * (1.0 + r))
            rets.append(r)
        else:
            # Fallback: proxy using reward (not ideal but works)
            r = float(reward[0])
            equity.append(equity[-1] + r)
            rets.append(r)

        obs = next_obs
        done = bool(done_arr[0])

    # Annualization factor from test timestamps
    test_ts = pd.read_csv(test_csv)["ts"].to_numpy()
    ann_factor = infer_ann_factor_from_ts(test_ts)
    sharpe = ann_sharpe_from_returns(np.array(rets), ann_factor)
    mdd = max_drawdown_from_equity(equity)

    return sharpe, mdd, model


# ======================== Slicing & I/O helpers ========================

def slice_by_years(df: pd.DataFrame, start_year: int, end_year_inclusive: int) -> pd.DataFrame:
    dt = pd.to_datetime(df["ts"], unit="s")
    mask = (dt.dt.year >= start_year) & (dt.dt.year <= end_year_inclusive)
    return df.loc[mask].copy()


def write_temp_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ======================== Main: Walk-Forward Driver ========================

def main():
    ap = argparse.ArgumentParser(description="Walk-forward training & evaluation for PPO RL.")
    ap.add_argument("--csv", required=True, help="Full dataset CSV (columns: ts,open,high,low,close,volume)")

    # Rolling-fold params (ignored if --custom_folds is provided)
    ap.add_argument("--start_year", type=int, help="First year included in any fold (e.g., 2010)")
    ap.add_argument("--end_year", type=int, help="Last year included in any fold (e.g., 2024)")
    ap.add_argument("--train_years", type=int, default=5, help="Years in each training window")
    ap.add_argument("--test_years", type=int, default=1, help="Years in each testing window")

    # Custom folds: "2006-2014:2015,2015-2023:2024,2008-2016:2017-2018"
    ap.add_argument("--custom_folds", type=str, default=None,
                    help='Explicit folds, e.g. "2006-2014:2015,2015-2023:2024,2008-2016:2017"')

    # PPO / training knobs
    ap.add_argument("--timesteps", type=int, default=800_000, help="Total PPO timesteps per fold")
    ap.add_argument("--window", type=int, default=32, help="Observation window for env")
    ap.add_argument("--fee_bps", type=float, default=1.0, help="Transaction fee in basis points")
    ap.add_argument("--logdir", default="walk_logs", help="Directory to store fold logs and stats")
    ap.add_argument("--norm_obs", action="store_true", help="Use VecNormalize for observations")
    ap.add_argument("--norm_reward", action="store_true", help="Use VecNormalize for rewards")
    ap.add_argument("--warm_start", action="store_true", help="Warm-start each fold from previous weights")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_steps", type=int, default=6144)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--ent_coef", type=float, default=0.007)
    ap.add_argument("--clip_range", type=float, default=0.2)
    args = ap.parse_args()

    df_full = pd.read_csv(args.csv).sort_values("ts")
    if not {"ts", "open", "high", "low", "close", "volume"}.issubset(df_full.columns):
        raise ValueError("CSV must contain ts, open, high, low, close, volume columns.")

    # PPO defaults
    ppo_kwargs = dict(
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        tensorboard_log=None,  # per-fold logger set via configure()
    )

    # Build folds
    if args.custom_folds:
        folds_to_run = parse_custom_folds(args.custom_folds)
    else:
        if args.start_year is None or args.end_year is None:
            raise ValueError("--start_year and --end_year are required when --custom_folds is not provided.")
        first_train_start = args.start_year
        last_possible_train_end = args.end_year - args.test_years
        folds_to_run = []
        for train_start in range(first_train_start, last_possible_train_end - args.train_years + 2):
            train_end = train_start + args.train_years - 1
            test_start = train_end + 1
            test_end = test_start + args.test_years - 1
            if test_end > args.end_year:
                break
            folds_to_run.append((train_start, train_end, test_start, test_end))

    results = []
    model: Optional[PPO] = None

    for (train_start, train_end, test_start, test_end) in folds_to_run:
        train_df = slice_by_years(df_full, train_start, train_end)
        test_df  = slice_by_years(df_full, test_start,  test_end)

        if len(train_df) < 500 or len(test_df) < 200:
            print(f"[skip] Too few rows: train={len(train_df)}, test={len(test_df)} "
                  f"for {train_start}-{train_end}->{test_start}-{test_end}")
            continue

        fold_name = f"{train_start}-{train_end}_to_{test_start}-{test_end}"
        fold_dir = os.path.join(args.logdir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        train_csv = write_temp_csv(train_df, os.path.join(fold_dir, "train.csv"))
        test_csv  = write_temp_csv(test_df,  os.path.join(fold_dir, "test.csv"))

        print(f"\n=== Fold: Train {train_start}-{train_end} → Test {test_start}-{test_end} ===")
        sharpe, mdd, model = train_and_eval(
            train_csv=train_csv,
            test_csv=test_csv,
            total_timesteps=args.timesteps,
            ppo_kwargs=ppo_kwargs,
            log_root=fold_dir,
            normalize_obs=args.norm_obs,
            normalize_reward=args.norm_reward,
            window=args.window,
            fee_bps=args.fee_bps,
            warm_start_model=(model if args.warm_start else None),
        )

        print(f"Fold Sharpe: {sharpe:.4f} | MaxDD: {mdd:.4f}")
        print(f"Saved model: {os.path.join(fold_dir, 'ppo_agent_walk.zip')}")
        print(f"Saved VecNormalize (if used): {os.path.join(fold_dir, 'vecnorm.pkl') if args.norm_obs or args.norm_reward else 'N/A'}")

        results.append({
            "train_start": train_start, "train_end": train_end,
            "test_start":  test_start,  "test_end":  test_end,
            "sharpe": sharpe, "max_drawdown": mdd,
            "train_rows": len(train_df), "test_rows": len(test_df),
        })

    if not results:
        print("No folds produced results. Check your year range and dataset coverage.")
        return

    res_df = pd.DataFrame(results)
    os.makedirs(args.logdir, exist_ok=True)
    res_csv = os.path.join(args.logdir, "walkforward_results.csv")
    res_df.to_csv(res_csv, index=False)

    mean_sharpe = res_df["sharpe"].mean()
    std_sharpe = res_df["sharpe"].std(ddof=1)
    mean_mdd = res_df["max_drawdown"].mean()

    print("\n=== Walk-Forward Summary ===")
    print(res_df[["train_start","train_end","test_start","test_end","sharpe","max_drawdown","train_rows","test_rows"]].to_string(index=False))
    print(f"\nAverage Test Sharpe: {mean_sharpe:.4f} ± {std_sharpe:.4f}  (across {len(res_df)} folds)")
    print(f"Average Test MaxDD : {mean_mdd:.4f}")
    print(f"\nSaved per-fold results to: {res_csv}")


if __name__ == "__main__":
    main()
