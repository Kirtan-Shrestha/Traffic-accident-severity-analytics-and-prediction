"""
Load the weather-enriched, cleaned collision data into the PostgreSQL star
schema: populate dimensions first (date, road, weather, location, severity),
then the fact table referencing their surrogate keys.

dim_weather is kept categorical/low-cardinality (condition + temperature band
+ precipitation level) rather than exact continuous values, which instead
live as measures on fact_accident - this keeps the dimension genuinely
reusable across many accidents, as a dimension table should be.

Idempotent: re-running truncates and reloads all tables cleanly.
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

CLEANED_DIR = Path("data/cleaned")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DATASET_YEAR = os.getenv("DATASET_YEAR", "2023")

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "db_load.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def load_data() -> pd.DataFrame:
    path = CLEANED_DIR / f"collisions_weather_enriched_{DATASET_YEAR}.csv"
    logger.info(f"Loading enriched data from {path}")
    df = pd.read_csv(path, parse_dates=["collision_datetime"])
    logger.info(f"Loaded {len(df)} rows")
    return df


def truncate_all(engine):
    logger.info("Truncating existing data for a clean reload")
    with engine.begin() as conn:
        conn.execute(text(
            "TRUNCATE TABLE fact_accident, dim_date, dim_road, dim_weather, "
            "dim_location, dim_severity RESTART IDENTITY CASCADE;"
        ))


def add_weather_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical temperature_band and precipitation_level columns."""
    df = df.copy()
    df["temperature_band"] = pd.cut(
        df["temperature_2m"],
        bins=[-100, 0, 10, 20, 100],
        labels=["Below 0", "0-10", "10-20", "Above 20"],
    ).astype(str)
    df["precipitation_level"] = pd.cut(
        df["precipitation"],
        bins=[-0.01, 0, 2, 10, 1000],
        labels=["None", "Light", "Moderate", "Heavy"],
    ).astype(str)
    return df


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["collision_datetime"]
    dim = pd.DataFrame({
        "date_key": dt.dt.strftime("%Y%m%d%H").astype(int),
        "full_date": dt.dt.date,
        "year": dt.dt.year,
        "month": dt.dt.month,
        "day": dt.dt.day,
        "hour": dt.dt.hour,
        "weekday_name": dt.dt.day_name(),
        "is_weekend": dt.dt.dayofweek >= 5,
    }).drop_duplicates(subset=["date_key"])
    return dim


def build_dim_road(df: pd.DataFrame) -> pd.DataFrame:
    dim = df[["road_type", "road_type_label", "speed_limit", "first_road_class"]].drop_duplicates().reset_index(drop=True)
    dim.columns = ["road_type_code", "road_type_label", "speed_limit", "first_road_class"]
    dim.insert(0, "road_key", range(1, len(dim) + 1))
    return dim


def build_dim_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical weather dimension: condition + temperature band + precipitation level."""
    dim = df[["weather_conditions", "weather_conditions_label", "temperature_band", "precipitation_level"]].drop_duplicates().reset_index(drop=True)
    dim.columns = ["weather_conditions_code", "weather_conditions_label", "temperature_band", "precipitation_level"]
    dim.insert(0, "weather_key", range(1, len(dim) + 1))
    return dim


def build_dim_location(df: pd.DataFrame) -> pd.DataFrame:
    dim = df[["latitude", "longitude", "lat_grid", "lon_grid",
              "local_authority_district", "police_force", "urban_or_rural_area_label"]].drop_duplicates(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)
    dim.columns = ["latitude", "longitude", "lat_grid", "lon_grid",
                   "local_authority_district", "police_force", "urban_or_rural_label"]
    dim.insert(0, "location_key", range(1, len(dim) + 1))
    return dim


def build_dim_severity(df: pd.DataFrame) -> pd.DataFrame:
    dim = df[["collision_severity", "collision_severity_label"]].drop_duplicates().reset_index(drop=True)
    dim.columns = ["severity_code", "severity_label"]
    dim.insert(0, "severity_key", range(1, len(dim) + 1))
    return dim


def build_fact(df: pd.DataFrame, dim_road: pd.DataFrame, dim_weather: pd.DataFrame,
               dim_location: pd.DataFrame, dim_severity: pd.DataFrame) -> pd.DataFrame:
    fact = df.copy()
    fact["date_key"] = fact["collision_datetime"].dt.strftime("%Y%m%d%H").astype(int)

    fact = fact.merge(
        dim_road, left_on=["road_type", "road_type_label", "speed_limit", "first_road_class"],
        right_on=["road_type_code", "road_type_label", "speed_limit", "first_road_class"], how="left"
    )
    fact = fact.merge(
        dim_weather, left_on=["weather_conditions", "weather_conditions_label", "temperature_band", "precipitation_level"],
        right_on=["weather_conditions_code", "weather_conditions_label", "temperature_band", "precipitation_level"], how="left"
    )
    fact = fact.merge(
        dim_location, on=["latitude", "longitude"], how="left", suffixes=("", "_loc")
    )
    fact = fact.merge(
        dim_severity, left_on=["collision_severity", "collision_severity_label"],
        right_on=["severity_code", "severity_label"], how="left"
    )

    result = fact[[
        "collision_index", "date_key", "road_key", "weather_key", "location_key", "severity_key",
        "number_of_vehicles", "number_of_casualties",
        "temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"
    ]].drop_duplicates(subset=["collision_index"])

    return result


def main():
    engine = get_engine()
    df = load_data()
    df = add_weather_bands(df)

    dim_date = build_dim_date(df)
    dim_road = build_dim_road(df)
    dim_weather = build_dim_weather(df)
    dim_location = build_dim_location(df)
    dim_severity = build_dim_severity(df)
    fact = build_fact(df, dim_road, dim_weather, dim_location, dim_severity)

    truncate_all(engine)

    logger.info(f"Loading dim_date: {len(dim_date)} rows")
    dim_date.to_sql("dim_date", engine, if_exists="append", index=False)

    logger.info(f"Loading dim_road: {len(dim_road)} rows")
    dim_road.to_sql("dim_road", engine, if_exists="append", index=False)

    logger.info(f"Loading dim_weather: {len(dim_weather)} rows")
    dim_weather.to_sql("dim_weather", engine, if_exists="append", index=False)

    logger.info(f"Loading dim_location: {len(dim_location)} rows")
    dim_location.to_sql("dim_location", engine, if_exists="append", index=False)

    logger.info(f"Loading dim_severity: {len(dim_severity)} rows")
    dim_severity.to_sql("dim_severity", engine, if_exists="append", index=False)

    logger.info(f"Loading fact_accident: {len(fact)} rows")
    fact.to_sql("fact_accident", engine, if_exists="append", index=False)

    # Populate PostGIS geometry column from lat/lon
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE dim_location SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) WHERE geom IS NULL;"
        ))

    logger.info("=== Database load complete ===")
    print(f"\nLoaded: dim_date={len(dim_date)}, dim_road={len(dim_road)}, dim_weather={len(dim_weather)}, "
          f"dim_location={len(dim_location)}, dim_severity={len(dim_severity)}, fact_accident={len(fact)}")


if __name__ == "__main__":
    main()