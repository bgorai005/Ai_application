# data_pipeline/transform.py
import pandas as pd
import os


def add_features(df):
    # Ensure numeric columns
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Moving averages
    df["MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

    # RSI (Wilder's smoothing via ewm approximation)
    period = 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use exponential smoothing with alpha=1/period to approximate Wilder's RSI
    roll_up = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100.0 / (1.0 + rs))

    # MACD (12, 26) and signal line (9)
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = signal
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Drop rows with any NaNs introduced by feature windows
    df.dropna(inplace=True)
    return df


def save_transformed_data(df, path="data/processed/nifty50_5min_features.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Transformed data saved to {path}")

    # remove non-numeric rows after saving
    df_clean = pd.read_csv(path)

    # Select only numeric rows (ignore rows with strings etc.)
    numeric_cols = ["Close", "High", "Low", "Open", "Volume", "MA_20", "MA_50", "RSI", "MACD", "MACD_signal", "MACD_hist"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean = df_clean[pd.to_numeric(df_clean[col], errors="coerce").notna()]

    # Save cleaned version
    df_clean.to_csv(path, index=False)
    print(f"[INFO] Cleaned non-numeric rows and re-saved to {path}")

