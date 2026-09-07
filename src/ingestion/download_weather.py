"""
Fetch historical weather data from the Open-Meteo Archive API for the unique
set of accident locations (rounded to a grid to minimize API calls).

Each unique rounded location gets ONE API call covering the full year, rather
than one call per accident. Results are cached to data/raw/weather/ so re-runs
don't re-fetch existing locations.
"""

import logging
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv()

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")
GRID_PRECISION = float(os.getenv("WEATHER_GRID_PRECISION", "0.3"))

WEATHER_CACHE_DIR = RAW_DATA_DIR / "weather"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "relative_humidity_2m",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weather_ingestion.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_unique_locations(collisions_path: Path, precision: float) -> pd.DataFrame:
    """Round accident coordinates to a grid and return unique locations."""
    df = pd.read_csv(collisions_path, usecols=["latitude", "longitude"])
    df = df.dropna(subset=["latitude", "longitude"])

    df["lat_grid"] = (df["latitude"] / precision).round() * precision
    df["lon_grid"] = (df["longitude"] / precision).round() * precision

    unique = df[["lat_grid", "lon_grid"]].drop_duplicates().reset_index(drop=True)
    return unique


def cache_filename(lat_grid: float, lon_grid: float) -> Path:
    return WEATHER_CACHE_DIR / f"weather_{lat_grid:.1f}_{lon_grid:.1f}.csv"


def fetch_weather_for_location(lat: float, lon: float, year: str, max_retries: int = 5) -> dict:
    """Call the Open-Meteo archive API for one location, full year. Retries on rate limiting."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    timestamp = datetime.now(timezone.utc).isoformat()

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=30)

            if response.status_code == 429:
                wait = min(60, 5 * attempt)
                logger.warning(f"Rate limited for ({lat}, {lon}), attempt {attempt}/{max_retries}, waiting {wait}s")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                raise ValueError(f"No 'hourly' key in response: {data}")

            hourly_df = pd.DataFrame(data["hourly"])
            hourly_df["lat_grid"] = lat
            hourly_df["lon_grid"] = lon

            dest = cache_filename(lat, lon)
            hourly_df.to_csv(dest, index=False)

            return {
                "lat_grid": lat,
                "lon_grid": lon,
                "status": "success",
                "timestamp": timestamp,
                "rows": len(hourly_df),
            }

        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Weather fetch failed for ({lat}, {lon}) attempt {attempt}: {e}")
            if attempt == max_retries:
                return {
                    "lat_grid": lat,
                    "lon_grid": lon,
                    "status": "failed",
                    "timestamp": timestamp,
                    "error": str(e),
                }
            time.sleep(5)

    return {
        "lat_grid": lat,
        "lon_grid": lon,
        "status": "failed",
        "timestamp": timestamp,
        "error": "max retries exceeded (rate limited)",
    }


def run_weather_ingestion() -> list[dict]:
    collisions_path = RAW_DATA_DIR / f"collisions_{DATASET_YEAR}.csv"
    unique_locations = get_unique_locations(collisions_path, GRID_PRECISION)
    logger.info(f"Found {len(unique_locations)} unique grid locations to fetch weather for")

    results = []
    for i, row in unique_locations.iterrows():
        lat, lon = row["lat_grid"], row["lon_grid"]
        dest = cache_filename(lat, lon)

        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f"[{i+1}/{len(unique_locations)}] Cached, skipping: ({lat}, {lon})")
            results.append({
                "lat_grid": lat, "lon_grid": lon,
                "status": "skipped_existing",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            continue

        logger.info(f"[{i+1}/{len(unique_locations)}] Fetching weather for ({lat}, {lon})")
        result = fetch_weather_for_location(lat, lon, DATASET_YEAR)
        results.append(result)

        # Be polite to the free API - small delay between calls
        time.sleep(1.5)

    return results


if __name__ == "__main__":
    logger.info("=== Starting weather ingestion run ===")
    results = run_weather_ingestion()

    success = sum(1 for r in results if r["status"] in ("success", "skipped_existing"))
    failed = sum(1 for r in results if r["status"] == "failed")
    logger.info(f"=== Weather ingestion complete: {success} succeeded/cached, {failed} failed ===")

    if failed > 0:
        logger.warning("Some locations failed - check weather_ingestion.log for details")