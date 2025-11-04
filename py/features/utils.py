import numpy as np

def sharpe(returns, eps=1e-12):
    r = np.asarray(returns)
    if r.size < 2:
        return 0.0
    m = r.mean()
    s = r.std(ddof=1)
    return float(m / max(s, eps))

def max_drawdown(equity):
    eq = np.asarray(equity)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max() if len(dd) else 0.0)
