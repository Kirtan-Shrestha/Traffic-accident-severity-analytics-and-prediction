import numpy as np
import pandas as pd

from src.ml.inference_pipeline import AccidentSeverityPipeline, INVERSE_LABEL_MAP


class _FakeTreeModel:
    """Stand-in for a trained LightGBM model; returns a fixed numeric class
    code per row so the encoding step can be tested in isolation from
    actual model training."""

    def __init__(self, numeric_pred):
        self.numeric_pred = numeric_pred

    def predict(self, X):
        return np.full(len(X), self.numeric_pred)


def _sample_feature_cols():
    return [
        "year", "month", "hour", "is_weekend", "speed_limit",
        "latitude", "longitude", "number_of_vehicles",
        "temperature_2m", "precipitation", "wind_speed_10m",
        "weekday_name_Wednesday", "road_type_label_Single carriageway",
        "weather_conditions_label_Fine no high winds",
        "temperature_band_10-20", "precipitation_level_None",
        "urban_or_rural_label_Urban",
    ]


def _sample_raw_row():
    return pd.DataFrame([{
        "year": 2023, "month": 10, "hour": 17, "is_weekend": False,
        "speed_limit": 30, "latitude": 51.5, "longitude": -0.1,
        "number_of_vehicles": 2, "temperature_2m": 12.5, "precipitation": 0.0,
        "wind_speed_10m": 15.0, "weekday_name": "Wednesday",
        "road_type_label": "Single carriageway",
        "weather_conditions_label": "Fine no high winds",
        "temperature_band": "10-20", "precipitation_level": "None",
        "urban_or_rural_label": "Urban",
    }])


def test_encode_produces_exact_training_columns():
    feature_cols = _sample_feature_cols()
    pipeline = AccidentSeverityPipeline(model=_FakeTreeModel(0), feature_cols=feature_cols)
    encoded = pipeline._encode(_sample_raw_row())
    assert list(encoded.columns) == feature_cols
    assert encoded.shape[0] == 1


def test_encode_sets_matching_onehot_to_one():
    feature_cols = _sample_feature_cols()
    pipeline = AccidentSeverityPipeline(model=_FakeTreeModel(0), feature_cols=feature_cols)
    encoded = pipeline._encode(_sample_raw_row())
    assert encoded.loc[0, "weekday_name_Wednesday"] == 1.0
    assert encoded.loc[0, "road_type_label_Single carriageway"] == 1.0


def test_encode_unseen_category_drops_to_zero():
    feature_cols = _sample_feature_cols()
    pipeline = AccidentSeverityPipeline(model=_FakeTreeModel(0), feature_cols=feature_cols)
    row = _sample_raw_row()
    row["road_type_label"] = "Roundabout"
    encoded = pipeline._encode(row)
    assert encoded.loc[0, "road_type_label_Single carriageway"] == 0.0
    assert "road_type_label_Roundabout" not in encoded.columns


def test_predict_maps_numeric_labels_back_to_strings():
    feature_cols = _sample_feature_cols()
    for numeric_code, label in INVERSE_LABEL_MAP.items():
        pipeline = AccidentSeverityPipeline(model=_FakeTreeModel(numeric_code), feature_cols=feature_cols)
        result = pipeline.predict(context=None, model_input=_sample_raw_row())
        assert result[0] == label
