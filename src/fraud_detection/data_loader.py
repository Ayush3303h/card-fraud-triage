
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def load_dataset(data_path=DEFAULT_DATA_PATH):
    """Load the original credit card transaction dataset."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    required_columns = {"Time", "Class"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_dataset(df):
    """Remove exact duplicate rows and sort transactions chronologically."""
    cleaned_df = df.drop_duplicates().copy()
    cleaned_df = cleaned_df.sort_values("Time").reset_index(drop=True)

    return cleaned_df


def chronological_split(df, train_ratio=0.70, validation_ratio=0.15):
    """Split transactions chronologically into train, validation and test."""
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")

    n = len(df)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df
