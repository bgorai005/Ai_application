# data_pipeline/validate.py

REQUIRED_COLUMNS = ["Datetime", "Close", "High", "Low", "Open", "Volume"]


def validate_data(df):
    assert not df.empty, "DataFrame is empty!"
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing_columns, f"Missing required columns: {missing_columns}"
    missing_values = df[REQUIRED_COLUMNS].isnull().sum()
    missing_values = missing_values[missing_values > 0]
    assert missing_values.empty, f"Required columns contain missing values: {missing_values.to_dict()}"
    print("[INFO] Data validation passed")
