# data_pipeline/pipeline.py

import pandas as pd
from ingest import download_intraday_data, normalize_market_data, save_raw_data
from validate import validate_data
from transform import add_features, save_transformed_data


def run_pipeline():
    print("[PIPELINE] Starting data pipeline...")

    # Download latest intraday chunk and append to raw store
    df_new = download_intraday_data()
    save_raw_data(df_new)

    # Load the combined raw dataset (appended) and run validation/transform on full data
    raw_path = "data/raw/nifty50_5min.csv"
    df_combined = normalize_market_data(pd.read_csv(raw_path))
    df_combined.to_csv(raw_path, index=False)

    validate_data(df_combined)

    df_transformed = add_features(df_combined)
    save_transformed_data(df_transformed)

    print("[PIPELINE] Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
