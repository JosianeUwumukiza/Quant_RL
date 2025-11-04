import os, sys
import pandas as pd
import yfinance as yf

def pick(df, keywords):
    """
    Return a 1D float Series for the first column whose *lowercased name*
    contains ANY of the given keyword substrings, robust to MultiIndex and (n,1) frames.
    """
    # Flatten MultiIndex names to strings
    cols = [(" ".join([str(x) for x in c]) if isinstance(c, tuple) else str(c)) for c in df.columns]
    for col, raw in zip(cols, df.columns):
        name = col.lower().replace("_", " ").strip()
        if any(k in name for k in keywords):
            s = df[raw]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return pd.to_numeric(s, errors="coerce")
    return None

if __name__ == "__main__":
    symbol   = sys.argv[1] if len(sys.argv)>1 else "BTC-USD"
    interval = sys.argv[2] if len(sys.argv)>2 else "1m"
    period   = sys.argv[3] if len(sys.argv)>3 else "5d"

    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise SystemExit(f"No data returned for {symbol} {interval} {period}")

    # If MultiIndex, leave as-is; 'pick' handles both
    print("Downloaded columns:", list(df.columns))

    df = df.reset_index()

    # Timestamp column
    if "Datetime" in df.columns:
        ts = pd.to_datetime(df["Datetime"], utc=True).astype("int64") // 10**9
    elif "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], utc=True).astype("int64") // 10**9
    else:
        ts = pd.to_datetime(df.index, utc=True).astype("int64") // 10**9

    # Robust OHLCV extraction by substring (handles "Close BTC-USD", etc.)
    open_ser  = pick(df, [" open"])
    high_ser  = pick(df, [" high"])
    low_ser   = pick(df, [" low"])
    # Prefer adj close if present, otherwise close
    close_ser = pick(df, ["adj close", " close"])
    vol_ser   = pick(df, [" volume"])

    if close_ser is None:
        raise SystemExit("Could not find any close/adj close column in downloaded data.")

    # Fallbacks: use close as proxy for missing OHLC; zeros for missing volume
    if open_ser is None:  open_ser  = close_ser
    if high_ser is None:  high_ser  = close_ser
    if low_ser  is None:  low_ser   = close_ser
    if vol_ser is None:   vol_ser   = pd.Series(0.0, index=close_ser.index)

    out = pd.DataFrame({
        "ts":     ts.values,
        "open":   open_ser.values.astype(float),
        "high":   high_ser.values.astype(float),
        "low":    low_ser.values.astype(float),
        "close":  close_ser.values.astype(float),
        "volume": vol_ser.values.astype(float),
    }).dropna()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{symbol.replace('-','_')}_{interval}.csv")
    out.to_csv(out_path, index=False)

    # Also write the standard filename used by train/evaluate
    sample = os.path.join(data_dir, "sample_btc_minute.csv")
    out.to_csv(sample, index=False)

    print("Saved:", out_path)
    print("Also wrote:", sample)
