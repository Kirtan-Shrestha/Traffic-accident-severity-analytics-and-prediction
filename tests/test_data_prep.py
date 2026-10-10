import pandas as pd

from src.ml.data_prep import (
    CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL, prepare_splits, get_raw_splits,
)


def _toy_df():
    rows = []
    for split, n in [("train", 4), ("validation", 2), ("test", 2)]:
        for i in range(n):
            rows.append({
                "accident_key": f"{split}-{i}",
                "collision_index": f"{split}-{i}",
                "year": 2023, "month": 1, "hour": 10, "is_weekend": i % 2 == 0,
                "speed_limit": 30, "latitude": 51.5, "longitude": -0.1,
                "number_of_vehicles": 2, "temperature_2m": 10.0, "precipitation": 0.0,
                "wind_speed_10m": 5.0, "weekday_name": "Monday",
                "road_type_label": "Single carriageway",
                "weather_conditions_label": "Fine no high winds",
                "temperature_band": "0-10", "precipitation_level": "None",
                "urban_or_rural_label": "Urban",
                TARGET_COL: "Slight", "split": split,
            })
    return pd.DataFrame(rows)


def test_prepare_splits_sizes_match_split_column():
    df = _toy_df()
    splits, feature_cols = prepare_splits(df)
    assert set(splits.keys()) == {"train", "validation", "test"}
    assert splits["train"][0].shape[0] == 4
    assert splits["validation"][0].shape[0] == 2
    assert splits["test"][0].shape[0] == 2


def test_prepare_splits_excludes_identifier_and_target_columns():
    df = _toy_df()
    _, feature_cols = prepare_splits(df)
    for excluded in ["accident_key", "collision_index", TARGET_COL, "split"]:
        assert excluded not in feature_cols


def test_prepare_splits_one_hot_encodes_categoricals():
    df = _toy_df()
    _, feature_cols = prepare_splits(df)
    assert "road_type_label_Single carriageway" in feature_cols
    for col in CATEGORICAL_COLS:
        assert col not in feature_cols


def test_get_raw_splits_keeps_categoricals_as_strings():
    df = _toy_df()
    splits = get_raw_splits(df)
    X_train, y_train = splits["train"]
    assert X_train.shape[0] == 4
    assert set(X_train.columns) == set(CATEGORICAL_COLS + NUMERIC_COLS)
    assert X_train["road_type_label"].dtype == object
    assert y_train.iloc[0] == "Slight"


def test_no_leakage_between_splits():
    df = _toy_df()
    splits, _ = prepare_splits(df)
    train_idx = set(splits["train"][0].index)
    val_idx = set(splits["validation"][0].index)
    test_idx = set(splits["test"][0].index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
