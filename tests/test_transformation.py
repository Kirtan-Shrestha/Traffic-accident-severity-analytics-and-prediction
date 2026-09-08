"""Tests for transformation and lookup decoding logic."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transformation.lookup_utils import decode_field


def test_decode_field_maps_known_codes():
    df = pd.DataFrame({"collision_severity": [1, 2, 3]})
    lookups = {"collision_severity": {"1": "Fatal", "2": "Serious", "3": "Slight"}}
    result = decode_field(df, "collision_severity", lookups)
    assert list(result["collision_severity_label"]) == ["Fatal", "Serious", "Slight"]


def test_decode_field_handles_unmapped_code():
    df = pd.DataFrame({"collision_severity": [1, 99]})
    lookups = {"collision_severity": {"1": "Fatal", "2": "Serious", "3": "Slight"}}
    result = decode_field(df, "collision_severity", lookups)
    assert result["collision_severity_label"].iloc[1] == "Unmapped code"


def test_decode_field_missing_lookup_key_is_noop():
    df = pd.DataFrame({"some_field": [1, 2]})
    lookups = {}
    result = decode_field(df, "some_field", lookups)
    # No label column added when the field has no lookup defined
    assert "some_field_label" not in result.columns


def test_temperature_band_bucketing():
    """Mirrors the banding logic used in load_to_postgres.build_dim_weather."""
    temps = pd.Series([-5, 3, 15, 25])
    bands = pd.cut(
        temps, bins=[-100, 0, 10, 20, 100],
        labels=["Below 0", "0-10", "10-20", "Above 20"],
    ).astype(str)
    assert list(bands) == ["Below 0", "0-10", "10-20", "Above 20"]


def test_precipitation_level_bucketing():
    precip = pd.Series([0, 1, 5, 20])
    levels = pd.cut(
        precip, bins=[-0.01, 0, 2, 10, 1000],
        labels=["None", "Light", "Moderate", "Heavy"],
    ).astype(str)
    assert list(levels) == ["None", "Light", "Moderate", "Heavy"]