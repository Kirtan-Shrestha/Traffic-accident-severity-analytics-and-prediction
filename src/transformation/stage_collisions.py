"""
Stage the raw collisions dataset: standardize date/time into a combined
datetime, decode key coded fields into human-readable labels (keeping the
original code alongside), and write to the staging layer.

This does NOT remove duplicates or invalid records - that happens in
Phase 5 (data quality). Staging preserves every row, just with clean types.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
STAGING_DIR = Path("data/staging")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")
LOOKUPS_PATH = Path("docs/code_lookups.json")

STAGING_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Fields we decode into human-readable labels for analysis and the dashboard
FIELDS_TO_DECODE = [
    "collision_severity",
    "road_type",
    "weather_conditions",
    "light_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "staging.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_lookups() -> dict:
    return json.loads(LOOKUPS_PATH.read_text())


def decode_field(df: pd.DataFrame, field: str, lookups: dict) -> pd.DataFrame:
    """Add a `{field}_label` column decoded from the code lookups, if available."""
    if field not in lookups:
        logger.warning(f"No lookup found for field '{field}', skipping decode")
        return df

    field_lookup = lookups[field]
    df[f"{field}_label"] = df[field].astype(str).map(field_lookup)
    df[f"{field}_label"] = df[f"{field}_label"].fillna("Unmapped code")
    return df


def stage_collisions() -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / f"collisions_{DATASET_YEAR}.csv"
    logger.info(f"Loading raw collisions from {raw_path}")

    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    logger.info(f"Loaded {initial_rows} raw collision rows")

    # Standardize column names (lowercase, already snake_case in this dataset)
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse combined datetime from date (DD/MM/YYYY) + time (HH:MM)
    df["collision_datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"].fillna("00:00"),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    n_bad_datetime = df["collision_datetime"].isna().sum()
    logger.info(f"Parsed collision_datetime; {n_bad_datetime} rows failed to parse")

    # Extract date parts useful for later analytical dimensions
    df["collision_year"] = df["collision_datetime"].dt.year
    df["collision_month"] = df["collision_datetime"].dt.month
    df["collision_hour"] = df["collision_datetime"].dt.hour
    df["collision_weekday"] = df["collision_datetime"].dt.day_name()

    # Decode coded fields into human-readable labels
    lookups = load_lookups()
    for field in FIELDS_TO_DECODE:
        if field in df.columns:
            df = decode_field(df, field, lookups)
        else:
            logger.warning(f"Field '{field}' not found in collisions data")

    # Round grid coordinates to match the weather cache join key (Phase 6)
    grid_precision = float(os.getenv("WEATHER_GRID_PRECISION", "0.3"))
    df["lat_grid"] = (df["latitude"] / grid_precision).round() * grid_precision
    df["lon_grid"] = (df["longitude"] / grid_precision).round() * grid_precision

    logger.info(f"Staging complete: {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    logger.info("=== Starting collision staging ===")
    staged_df = stage_collisions()

    output_path = STAGING_DIR / f"collisions_staged_{DATASET_YEAR}.csv"
    staged_df.to_csv(output_path, index=False)

    logger.info(f"Saved staged collisions to {output_path}")
    logger.info(f"=== Staging complete: {len(staged_df)} rows ===")

    print(f"\nStaged {len(staged_df)} rows to {output_path}")
    print(f"Columns: {staged_df.columns.tolist()}")