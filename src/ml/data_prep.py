"""
Load the model-ready feature table from PostgreSQL and prepare it for
training: encode categoricals, split by the pre-computed chronological
split column, separate features from the target.
"""

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

CATEGORICAL_COLS = [
    "weekday_name", "road_type_label", "weather_conditions_label",
    "temperature_band", "precipitation_level", "urban_or_rural_label",
]
NUMERIC_COLS = [
    "year", "month", "hour", "is_weekend", "speed_limit",
    "latitude", "longitude", "number_of_vehicles",
    "temperature_2m", "precipitation", "wind_speed_10m",
]
TARGET_COL = "target_severity"


def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def load_features() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM ml_accident_features", engine)


def prepare_splits(df: pd.DataFrame):
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test)."""
    df = df.copy()
    df["is_weekend"] = df["is_weekend"].astype(int)

    # One-hot encode categoricals, fit on full data so train/val/test share columns
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    feature_cols = [c for c in df_encoded.columns if c not in
                     ["accident_key", "collision_index", TARGET_COL, "split"]]

    splits = {}
    for split_name in ["train", "validation", "test"]:
        subset = df_encoded[df["split"] == split_name]
        X = subset[feature_cols]
        y = subset[TARGET_COL]
        splits[split_name] = (X, y)

    return splits, feature_cols


if __name__ == "__main__":
    df = load_features()
    print(f"Loaded {len(df)} rows")
    print(df["split"].value_counts())
    splits, feature_cols = prepare_splits(df)
    for name, (X, y) in splits.items():
        print(f"{name}: X={X.shape}, y distribution:\n{y.value_counts()}")
    print(f"\nTotal feature columns after encoding: {len(feature_cols)}")