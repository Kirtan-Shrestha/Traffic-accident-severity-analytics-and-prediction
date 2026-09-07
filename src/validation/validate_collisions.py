"""
Apply data-quality checks to staged collisions. Rows that pass all checks
go to the cleaned layer; rows that fail any check go to the rejected layer
along with a detailed reason. No row is silently dropped.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

STAGING_DIR = Path("data/staging")
CLEANED_DIR = Path("data/cleaned")
REJECTED_DIR = Path("data/rejected")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rough UK bounding box (generous, includes Shetland and outlying islands)
UK_LAT_MIN, UK_LAT_MAX = 49.5, 61.0
UK_LON_MIN, UK_LON_MAX = -8.5, 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_quality.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_checks(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all validation rules. Returns (cleaned_df, rejected_records_df).
    rejected_records_df has one row PER ERROR (a record can appear multiple
    times if it fails multiple checks).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    rejected_rows = []
    reject_ids = set()

    def flag(mask: pd.Series, error_type: str, message: str):
        bad_ids = df.loc[mask, "collision_index"]
        for record_id in bad_ids:
            rejected_rows.append({
                "record_id": record_id,
                "error_type": error_type,
                "error_message": message,
                "pipeline_timestamp": timestamp,
                "source": source,
            })
        reject_ids.update(bad_ids)

    # 1. Missing collision_index
    flag(df["collision_index"].isna(), "missing_id", "collision_index is missing")

    # 2. Duplicate collision_index (flag all but the first occurrence)
    dup_mask = df["collision_index"].duplicated(keep="first")
    flag(dup_mask, "duplicate_record", "duplicate collision_index")

    # 3. Invalid latitude
    lat_invalid = df["latitude"].isna() | (df["latitude"] < UK_LAT_MIN) | (df["latitude"] > UK_LAT_MAX)
    flag(lat_invalid, "invalid_latitude", f"latitude outside expected UK range [{UK_LAT_MIN}, {UK_LAT_MAX}]")

    # 4. Invalid longitude
    lon_invalid = df["longitude"].isna() | (df["longitude"] < UK_LON_MIN) | (df["longitude"] > UK_LON_MAX)
    flag(lon_invalid, "invalid_longitude", f"longitude outside expected UK range [{UK_LON_MIN}, {UK_LON_MAX}]")

    # 5. Invalid/unparseable datetime
    dt_invalid = df["collision_datetime"].isna()
    flag(dt_invalid, "invalid_datetime", "collision_datetime could not be parsed")

    # 6. Invalid severity (must be 1, 2, or 3)
    severity_invalid = ~df["collision_severity"].isin([1, 2, 3])
    flag(severity_invalid, "invalid_severity", "collision_severity not in {1, 2, 3}")

    # 7. Invalid vehicle/casualty counts (must be positive)
    count_invalid = (df["number_of_vehicles"] <= 0) | (df["number_of_casualties"] <= 0)
    flag(count_invalid, "invalid_counts", "number_of_vehicles or number_of_casualties is not positive")

    rejected_df = pd.DataFrame(rejected_rows)
    cleaned_df = df[~df["collision_index"].isin(reject_ids)].copy()

    return cleaned_df, rejected_df


def main():
    input_path = STAGING_DIR / f"collisions_staged_{DATASET_YEAR}.csv"
    logger.info(f"Loading staged collisions from {input_path}")

    df = pd.read_csv(input_path, parse_dates=["collision_datetime"])
    initial_rows = len(df)
    logger.info(f"Loaded {initial_rows} staged rows")

    cleaned_df, rejected_df = run_checks(df, source=str(input_path))

    cleaned_path = CLEANED_DIR / f"collisions_cleaned_{DATASET_YEAR}.csv"
    rejected_path = REJECTED_DIR / f"collisions_rejected_{DATASET_YEAR}.csv"

    cleaned_df.to_csv(cleaned_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)

    n_unique_rejected_ids = rejected_df["record_id"].nunique() if len(rejected_df) else 0

    logger.info(f"Cleaned: {len(cleaned_df)} rows -> {cleaned_path}")
    logger.info(f"Rejected: {len(rejected_df)} error entries covering {n_unique_rejected_ids} unique records -> {rejected_path}")
    logger.info(f"Check: {len(cleaned_df)} + {n_unique_rejected_ids} = {len(cleaned_df) + n_unique_rejected_ids} (should equal {initial_rows})")

    print(f"\nCleaned: {len(cleaned_df)} rows")
    print(f"Rejected: {n_unique_rejected_ids} unique records ({len(rejected_df)} total error entries)")
    if len(rejected_df):
        print("\nError type breakdown:")
        print(rejected_df["error_type"].value_counts())


if __name__ == "__main__":
    main()