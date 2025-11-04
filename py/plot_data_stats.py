# py/plot_data_stats.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_xy(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_hist(y, bins, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.hist(y, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def empirical_cdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1) / len(xs)
    return xs, ys

def acf(x, nlags=40):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    autocorr = [1.0]
    denom = (x**2).sum()
    for k in range(1, nlags+1):
        num = (x[:-k] * x[k:]).sum()
        autocorr.append(num / denom if denom > 0 else 0.0)
    return np.arange(nlags+1), np.array(autocorr)

def main():
    here = os.path.dirname(__file__)
    data_csv = os.path.join(here, "data", "sample_btc_minute.csv")  # change to SPY_5m.csv if you prefer
    out_dir = os.path.join(here, "report")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_csv):
        raise FileNotFoundError(data_csv)

    df = pd.read_csv(data_csv)
    if "ts" in df.columns:
        t = pd.to_datetime(df["ts"], unit="s")
    else:
        t = pd.RangeIndex(len(df))
    close = df["close"].astype(float)
    ret = close.pct_change().dropna().values

    # 1) Price and returns
    plot_xy(t, close, "Close Price", "Time", "Price", os.path.join(out_dir, "data_price.png"))
    plot_hist(ret, bins=100, title="Returns Histogram", xlabel="Return", ylabel="Count",
              outpath=os.path.join(out_dir, "data_returns_hist.png"))

    # 2) Rolling volatility (realized)
    vol = pd.Series(ret).rolling(64).std().bfill().values
    plot_xy(range(len(vol)), vol, "Rolling Volatility (window=64)", "Step", "Volatility",
            os.path.join(out_dir, "data_rolling_vol.png"))

    # 3) Empirical CDF vs Normal CDF (fat tails check)
    xs, ys = empirical_cdf(ret)
    mu, sigma = float(np.mean(ret)), float(np.std(ret) + 1e-12)
    norm_cdf = 0.5 * (1 + (xs - mu) / (np.sqrt(2) * sigma))
    plt.figure()
    plt.plot(xs, ys, label="Empirical CDF")
    plt.plot(xs, norm_cdf, label="Normal (same μ, σ)")
    plt.title("Empirical vs Normal CDF (returns)")
    plt.xlabel("Return")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "data_cdf_vs_normal.png"), dpi=160)
    plt.close()

    # 4) Autocorrelation (are short-lag patterns tradable?)
    lags, vals = acf(ret, nlags=40)
    plt.figure()
    plt.stem(lags, vals)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "data_acf.png"), dpi=160)
    plt.close()

    print("✅ Data stats saved to:", out_dir)
    print("Hints:")
    print("- If tails >> normal in 'data_cdf_vs_normal.png', consider GARCH or heavy-tail models (t-distribution).")
    print("- If ACF shows negative lag-1, short-term mean-reversion features can help.")
    print("- If rolling volatility clusters, regime features and volatility targeting help a lot.")

if __name__ == "__main__":
    main()
