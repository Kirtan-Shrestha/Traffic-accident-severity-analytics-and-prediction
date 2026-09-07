"""
Stage the raw vehicles dataset: decode key coded fields into human-readable
labels. Does NOT remove duplicates or invalid records (Phase 5 handles that).
"""

import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

from src.transformation.lookup_utils import load_lookups, decode_field

load_dotenv()

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
STAGING_DIR = Path("data/staging")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")

STAGING_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FIELDS_TO_DECODE = [
    "vehicle_type",
    "sex_of_driver",
    "journey_purpose_of_driver",
    "propulsion_code",
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


def stage_vehicles() -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / f"vehicles_{DATASET_YEAR}.csv"
    logger.info(f"Loading raw vehicles from {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} raw vehicle rows")

    df.columns = [c.strip().lower() for c in df.columns]

    lookups = load_lookups()
    for field in FIELDS_TO_DECODE:
        if field in df.columns:
            df = decode_field(df, field, lookups)
        else:
            logger.warning(f"Field '{field}' not found in vehicles data")

    logger.info(f"Vehicle staging complete: {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    logger.info("=== Starting vehicle staging ===")
    staged_df = stage_vehicles()

    output_path = STAGING_DIR / f"vehicles_staged_{DATASET_YEAR}.csv"
    staged_df.to_csv(output_path, index=False)

    logger.info(f"Saved staged vehicles to {output_path}")
    print(f"\nStaged {len(staged_df)} rows to {output_path}")