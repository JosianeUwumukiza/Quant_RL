import numpy as np
import pandas as pd

def momentum_signal(close: pd.Series, short=10, long=40):
    s = close.rolling(short).mean()
    l = close.rolling(long).mean()
    sig = np.where(s > l, 1, -1)
    return pd.Series(sig, index=close.index).fillna(0)

def mean_revert_signal(close: pd.Series, lookback=20, thresh=1.0):
    r = (close - close.rolling(lookback).mean()) / (close.rolling(lookback).std()+1e-12)
    sig = np.where(r >  thresh, -1, np.where(r < -thresh, 1, 0))
    return pd.Series(sig, index=close.index).fillna(0)

def backtest(close: pd.Series, signal: pd.Series, fee_bps=1.0):
    ret = close.pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0)
    gross = pos * ret
    churn = (pos.diff().abs().fillna(0)) * (fee_bps*1e-4)
    pnl = gross - churn
    eq = pnl.cumsum()
    return pnl, eq
