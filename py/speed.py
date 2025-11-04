import os
import time
import argparse
import numpy as np

# Use the Python wrapper that loads CSV and calls the C++ core
from envs.gym_alpha_env import GymAlphaEnv

def do_step(env, action):
    """Step once and return (obs, done). Handles gym/gymnasium return shapes."""
    out = env.step(action)
    if len(out) == 4:
        # gym: obs, reward, done, info
        obs, _, done, _ = out
        return obs, bool(done)
    elif len(out) == 5:
        # gymnasium: obs, reward, terminated, truncated, info
        obs, _, terminated, truncated, _ = out
        return obs, bool(terminated or truncated)
    else:
        raise RuntimeError(f"Unexpected env.step() return length: {len(out)}")

def do_reset(env):
    """Reset and return obs, handling gym/gymnasium shapes."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        # gymnasium: (obs, info)
        return out[0]
    return out  # gym: obs

def main():
    parser = argparse.ArgumentParser(description="Benchmark GymAlphaEnv/C++ simulator speed.")
    parser.add_argument("--csv", default=None,
                        help="Path to CSV (ts,open,high,low,close,volume). Default: data/SPY_1d_2005toNow.csv")
    parser.add_argument("--steps", type=int, default=10_000, help="Number of steps to time")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--action", type=int, default=0, help="Action to step with (-1,0,+1)")
    args = parser.parse_args()

    # Resolve default CSV relative to this file
    if args.csv is None:
        here = os.path.dirname(__file__)
        args.csv = os.path.join(here, "data", "SPY_1d_2005toNow.csv")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    env = GymAlphaEnv(args.csv, window=args.window, fee_bps=args.fee_bps)
    _ = do_reset(env)

    N = int(args.steps)
    print(f"Running {N:,} steps with data: {os.path.basename(args.csv)} (window={args.window}, fee_bps={args.fee_bps})")

    # Warmup a tiny bit to JIT/cache any paths
    for _ in range(128):
        obs, done = do_step(env, args.action)
        if done:
            _ = do_reset(env)

    t0 = time.perf_counter()
    steps_done = 0
    while steps_done < N:
        obs, done = do_step(env, args.action)
        steps_done += 1
        if done:
            _ = do_reset(env)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_us = (elapsed / steps_done) * 1e6
    tput = steps_done / elapsed

    print("\n=== Speed Results ===")
    print(f"Total time       : {elapsed:.3f} s")
    print(f"Avg step latency : {avg_us:.2f} µs")
    print(f"Throughput       : {tput:,.0f} steps/s")
    print(f"Steps completed  : {steps_done:,}")

if __name__ == "__main__":
    main()
