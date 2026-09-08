"""Tests for the data quality validation logic."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.validate_collisions import run_checks


def make_sample_df():
    return pd.DataFrame({
        "collision_index": ["A1", "A2", "A3", "A4", "A2"],  # A2 duplicated
        "latitude": [51.5, None, 55.0, 90.0, 51.5],  # A2 missing, A4 out of range
        "longitude": [-0.1, -1.0, -3.0, -0.1, -1.0],
        "collision_datetime": pd.to_datetime(
            ["2023-01-01 10:00", "2023-01-02 11:00", None, "2023-01-04 12:00", "2023-01-02 11:00"]
        ),  # A3 has no datetime
        "collision_severity": [1, 2, 3, 9, 2],  # A4 has invalid severity 9
        "number_of_vehicles": [1, 2, 1, 0, 2],  # A4 has 0 vehicles
        "number_of_casualties": [1, 1, 1, 1, 1],
    })


def test_valid_record_passes():
    df = make_sample_df()
    cleaned, rejected = run_checks(df, source="test")
    assert "A1" in cleaned["collision_index"].values


def test_missing_coordinate_rejected():
    df = make_sample_df()
    cleaned, rejected = run_checks(df, source="test")
    assert "A2" in rejected["record_id"].values
    assert "invalid_latitude" in rejected[rejected["record_id"] == "A2"]["error_type"].values


def test_duplicate_rejected():
    df = make_sample_df()
    cleaned, rejected = run_checks(df, source="test")
    assert "duplicate_record" in rejected["error_type"].values


def test_invalid_datetime_rejected():
    df = make_sample_df()
    cleaned, rejected = run_checks(df, source="test")
    assert "A3" in rejected["record_id"].values
    assert "invalid_datetime" in rejected[rejected["record_id"] == "A3"]["error_type"].values


def test_invalid_severity_and_counts_rejected():
    df = make_sample_df()
    cleaned, rejected = run_checks(df, source="test")
    a4_errors = rejected[rejected["record_id"] == "A4"]["error_type"].values
    assert "invalid_severity" in a4_errors
    assert "invalid_counts" in a4_errors


def test_no_record_silently_dropped():
    """Every input record must appear in either cleaned or rejected - never disappear."""
    df = make_sample_df()
    cleaned, rejected = run_checks(df, source="test")
    all_ids = set(df["collision_index"])
    accounted_for = set(cleaned["collision_index"]) | set(rejected["record_id"])
    assert all_ids.issubset(accounted_for)