"""Refresh all analytical mart materialized views after a fresh data load."""

import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MART_VIEWS = [
    "mart_accidents_by_hour",
    "mart_accidents_by_weekday",
    "mart_accidents_by_road_type",
    "mart_accidents_by_weather",
    "mart_severity_distribution",
    "mart_location_hotspots",
]


def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def main():
    engine = get_engine()
    with engine.begin() as conn:
        for view in MART_VIEWS:
            logger.info(f"Refreshing {view}")
            conn.execute(text(f"REFRESH MATERIALIZED VIEW {view};"))
    logger.info("All materialized views refreshed successfully")


if __name__ == "__main__":
    main()