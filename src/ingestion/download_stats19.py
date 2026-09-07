"""
Download and verify UK STATS19 road safety data (collisions, vehicles, casualties)
for a given year. Downloads are idempotent: if a file already exists with a
non-zero size, it is not re-downloaded, but its presence is still logged.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
import os

load_dotenv()

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")

BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data"

FILES = {
    "collisions": f"dft-road-casualty-statistics-collision-{DATASET_YEAR}.csv",
    "vehicles": f"dft-road-casualty-statistics-vehicle-{DATASET_YEAR}.csv",
    "casualties": f"dft-road-casualty-statistics-casualty-{DATASET_YEAR}.csv",
}

LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingestion.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: Path, dataset_name: str) -> dict:
    """Download a single file and return a result record for logging."""
    timestamp = datetime.now(timezone.utc).isoformat()

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"{dataset_name}: already exists at {dest_path}, skipping download")
        return {
            "dataset": dataset_name,
            "source": url,
            "status": "skipped_existing",
            "timestamp": timestamp,
            "file_path": str(dest_path),
            "size_bytes": dest_path.stat().st_size,
        }

    try:
        logger.info(f"{dataset_name}: downloading from {url}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        dest_path.write_bytes(response.content)
        size = dest_path.stat().st_size

        logger.info(f"{dataset_name}: downloaded successfully ({size} bytes)")
        return {
            "dataset": dataset_name,
            "source": url,
            "status": "success",
            "timestamp": timestamp,
            "file_path": str(dest_path),
            "size_bytes": size,
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"{dataset_name}: download failed - {e}")
        return {
            "dataset": dataset_name,
            "source": url,
            "status": "failed",
            "timestamp": timestamp,
            "file_path": str(dest_path),
            "error": str(e),
        }


def count_rows(file_path: Path) -> int:
    """Count data rows in a CSV (excluding header)."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f) - 1


def run_ingestion() -> list[dict]:
    results = []
    for name, filename in FILES.items():
        url = f"{BASE_URL}/{filename}"
        dest = RAW_DATA_DIR / f"{name}_{DATASET_YEAR}.csv"
        result = download_file(url, dest, name)

        if result["status"] in ("success", "skipped_existing"):
            try:
                row_count = count_rows(dest)
                result["row_count"] = row_count
                logger.info(f"{name}: {row_count} rows")
            except Exception as e:
                logger.error(f"{name}: failed to count rows - {e}")
                result["row_count"] = None

        results.append(result)

    return results


if __name__ == "__main__":
    logger.info("=== Starting STATS19 ingestion run ===")
    results = run_ingestion()

    success_count = sum(1 for r in results if r["status"] in ("success", "skipped_existing"))
    logger.info(f"=== Ingestion complete: {success_count}/{len(results)} files ready ===")

    for r in results:
        print(r)