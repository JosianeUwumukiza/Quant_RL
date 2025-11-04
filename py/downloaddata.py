import os
import argparse
import pandas as pd
import yfinance as yf

def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten possible MultiIndex columns to simple strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in tup if str(c) != ""]) for tup in df.columns]
    return df

def col(df: pd.DataFrame, name: str):
    """Return a 1-D numpy array for the given column name (handles variants like 'Open SPY')."""
    # Try exact
    if name in df.columns:
        return df[name].to_numpy().ravel()
    # Try variant with symbol suffix (e.g., 'Open SPY')
    cand = [c for c in df.columns if c.lower().startswith(name.lower())]
    if cand:
        return df[cand[0]].to_numpy().ravel()
    raise KeyError(f"Column '{name}' not found in: {list(df.columns)}")

def main():
    ap = argparse.ArgumentParser(description="Download OHLCV to normalized CSV")
    ap.add_argument("--symbol", required=True, help="Ticker, e.g. SPY")
    ap.add_argument("--interval", default="1d", help="1d, 5m, 1m, etc.")
    ap.add_argument("--start", default="2005-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD or omit for now")
    ap.add_argument("--out", default=None, help="Output path (default: py/data/<SYMBOL>_<INTERVAL>.csv)")
    ap.add_argument("--append", action="store_true", help="Append to existing file (de-dup by ts)")
    ap.add_argument("--auto_adjust", action="store_true", help="Use adjusted prices (splits/dividends)")
    args = ap.parse_args()

    df = yf.download(
        tickers=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=args.auto_adjust,
        progress=True,
    )
    if df is None or df.empty:
        raise SystemExit("No data returned from yfinance.")

    df = flatten_cols(df).reset_index()

    # Identify datetime column name
    dt_col = None
    for cand in ("Datetime", "Date"):
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise RuntimeError(f"Could not find a datetime column in: {list(df.columns)}")

    # Normalize
    ts = pd.to_datetime(df[dt_col], utc=False).astype("int64") // 10**9
    out = pd.DataFrame({
        "ts":     ts.astype("int64"),
        "open":   col(df, "Open").astype("float64"),
        "high":   col(df, "High").astype("float64"),
        "low":    col(df, "Low").astype("float64"),
        "close":  col(df, "Close").astype("float64"),
        "volume": col(df, "Volume").astype("float64"),
    }).dropna()

    # Output path
    if args.out:
        save_path = args.out
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, f"{args.symbol}_{args.interval}.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Append or overwrite
    if args.append and os.path.exists(save_path):
        old = pd.read_csv(save_path)
        combined = pd.concat([old, out], ignore_index=True)
        combined = combined.drop_duplicates(subset="ts").sort_values("ts")
        combined.to_csv(save_path, index=False)
        print(f"✅ Appended. Rows now: {len(combined)} → {save_path}")
    else:
        out = out.drop_duplicates(subset="ts").sort_values("ts")
        out.to_csv(save_path, index=False)
        print(f"✅ Wrote {len(out)} rows → {save_path}")

if __name__ == "__main__":
    main()
