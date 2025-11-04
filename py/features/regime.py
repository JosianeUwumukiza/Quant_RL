import numpy as np
import pandas as pd

def rolling_vol_regime(close: pd.Series, window=64, vol_hi=0.02, vol_lo=0.0075):
    ret = close.pct_change().fillna(0.0)
    vol = ret.rolling(window).std().fillna(method="bfill")
    state = np.where(vol > vol_hi, 2, np.where(vol < vol_lo, 0, 1))
    return pd.Series(state, index=close.index), vol
