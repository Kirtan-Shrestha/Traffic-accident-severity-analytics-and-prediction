"""
Join cleaned collision records with historical weather data.
Weather is matched by rounded grid location (lat_grid, lon_grid) and the
nearest hour to the collision's datetime.
"""

import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

CLEANED_DIR = Path("data/cleaned")
WEATHER_CACHE_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw")) / "weather"
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weather_join.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_all_weather() -> pd.DataFrame:
    """Combine all cached per-location weather CSVs into one DataFrame."""
    weather_files = list(WEATHER_CACHE_DIR.glob("weather_*.csv"))
    logger.info(f"Loading {len(weather_files)} weather cache files")

    dfs = [pd.read_csv(f) for f in weather_files]
    combined = pd.concat(dfs, ignore_index=True)

    combined["time"] = pd.to_datetime(combined["time"])
    logger.info(f"Combined weather data: {len(combined)} hourly records")
    return combined


def join_weather() -> pd.DataFrame:
    collisions_path = CLEANED_DIR / f"collisions_cleaned_{DATASET_YEAR}.csv"
    logger.info(f"Loading cleaned collisions from {collisions_path}")

    collisions = pd.read_csv(collisions_path, parse_dates=["collision_datetime"])
    initial_rows = len(collisions)
    logger.info(f"Loaded {initial_rows} cleaned collision rows")

    weather = load_all_weather()

    # Round collision datetime down to the nearest hour for the join key
    collisions["weather_join_hour"] = collisions["collision_datetime"].dt.floor("h")

    # Round grid coords to 1 decimal to avoid floating point mismatch on merge
    collisions["lat_grid"] = collisions["lat_grid"].round(1)
    collisions["lon_grid"] = collisions["lon_grid"].round(1)
    weather["lat_grid"] = weather["lat_grid"].round(1)
    weather["lon_grid"] = weather["lon_grid"].round(1)

    weather = weather.rename(columns={"time": "weather_join_hour"})

    merged = collisions.merge(
        weather,
        on=["lat_grid", "lon_grid", "weather_join_hour"],
        how="left",
        suffixes=("", "_weather"),
    )

    n_matched = merged["temperature_2m"].notna().sum()
    n_unmatched = merged["temperature_2m"].isna().sum()

    logger.info(f"Weather join complete: {n_matched} matched, {n_unmatched} unmatched out of {len(merged)}")

    if len(merged) != initial_rows:
        logger.warning(
            f"Row count changed after join! {initial_rows} -> {len(merged)} "
            f"(possible duplicate weather rows for same grid+hour)"
        )

    return merged


if __name__ == "__main__":
    logger.info("=== Starting weather join ===")
    enriched = join_weather()

    output_path = CLEANED_DIR / f"collisions_weather_enriched_{DATASET_YEAR}.csv"
    enriched.to_csv(output_path, index=False)

    logger.info(f"Saved enriched collisions to {output_path}")
    print(f"\nEnriched {len(enriched)} rows saved to {output_path}")
    print(f"Matched weather: {enriched['temperature_2m'].notna().sum()}")
    print(f"Missing weather: {enriched['temperature_2m'].isna().sum()}")