# data_pipeline/ingest.py
import yfinance as yf
import pandas as pd
import os
import ast


CANONICAL_COLUMNS = ["Datetime", "Close", "High", "Low", "Open", "Volume"]


def _normalize_column_name(column):
    if isinstance(column, tuple):
        return str(column[0])

    if isinstance(column, str) and column.startswith("("):
        try:
            parsed = ast.literal_eval(column)
            if isinstance(parsed, tuple) and parsed:
                return str(parsed[0])
        except (SyntaxError, ValueError):
            pass

    return str(column)


def normalize_market_data(df):
    df = df.copy()
    normalized = {}

    for column in df.columns:
        key = _normalize_column_name(column)
        if key not in CANONICAL_COLUMNS:
            continue

        series = df[column]
        if key == "Datetime":
            series = pd.to_datetime(series, errors="coerce")
        else:
            series = pd.to_numeric(series, errors="coerce")

        if key in normalized:
            normalized[key] = normalized[key].combine_first(series)
        else:
            normalized[key] = series

    missing_columns = [col for col in CANONICAL_COLUMNS if col not in normalized]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = pd.DataFrame({col: normalized[col] for col in CANONICAL_COLUMNS})
    cleaned.dropna(subset=CANONICAL_COLUMNS, inplace=True)
    cleaned.drop_duplicates(subset=["Datetime"], keep="last", inplace=True)
    cleaned.sort_values(by="Datetime", inplace=True)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def download_intraday_data(interval="5m", period="60d"):
    ticker = "^NSEI"
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df.reset_index(inplace=True)
    return df


def save_raw_data(df, path="data/raw/nifty50_5min.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df_new = normalize_market_data(df)

    # If a raw file exists, append + deduplicate by Datetime; otherwise create new file
    if os.path.exists(path):
        existing = normalize_market_data(pd.read_csv(path))
        combined = pd.concat([existing, df_new], ignore_index=True, sort=False)
        combined = normalize_market_data(combined)
        combined.drop_duplicates(subset=['Datetime'], keep='last', inplace=True)
        combined.sort_values(by='Datetime', inplace=True)
        combined.to_csv(path, index=False)
        print(f"[INFO] Appended data to {path} (now {len(combined)} rows).")
    else:
        df_new.to_csv(path, index=False)
        print(f"[INFO] Raw NIFTY 50 index data saved to {path}")

    df_clean = normalize_market_data(pd.read_csv(path))
    df_clean.to_csv(path, index=False)
    print(f"[INFO] Cleaned non-numeric rows and re-saved to {path}")
