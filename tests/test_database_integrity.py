"""
Integration tests that verify the loaded database's referential integrity
and analytical mart correctness. Requires the PostgreSQL container to be
running with data already loaded (Phase 8).
"""

import pytest
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()


@pytest.fixture(scope="module")
def engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def test_fact_table_has_data(engine):
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM fact_accident")).scalar()
    assert count > 100000


def test_no_orphaned_fact_rows(engine):
    """Every fact row must resolve to a real row in each dimension (no broken FKs)."""
    checks = {
        "date_key": "dim_date",
        "road_key": "dim_road",
        "weather_key": "dim_weather",
        "location_key": "dim_location",
        "severity_key": "dim_severity",
    }
    with engine.connect() as conn:
        for fk_col, dim_table in checks.items():
            orphans = conn.execute(text(
                f"SELECT COUNT(*) FROM fact_accident f "
                f"LEFT JOIN {dim_table} d ON f.{fk_col} = d.{fk_col} "
                f"WHERE f.{fk_col} IS NOT NULL AND d.{fk_col} IS NULL"
            )).scalar()
            assert orphans == 0, f"Found {orphans} orphaned rows for {fk_col} -> {dim_table}"


def test_no_duplicate_collision_index(engine):
    with engine.connect() as conn:
        dupes = conn.execute(text(
            "SELECT COUNT(*) FROM ("
            "  SELECT collision_index FROM fact_accident "
            "  GROUP BY collision_index HAVING COUNT(*) > 1"
            ") sub"
        )).scalar()
    assert dupes == 0


def test_severity_distribution_sums_to_total(engine):
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM fact_accident")).scalar()
        mart_total = conn.execute(text("SELECT SUM(total_accidents) FROM mart_severity_distribution")).scalar()
    assert total == mart_total


def test_all_mart_views_exist_and_have_data(engine):
    views = [
        "mart_accidents_by_hour", "mart_accidents_by_weekday", "mart_accidents_by_road_type",
        "mart_accidents_by_weather", "mart_severity_distribution", "mart_location_hotspots",
    ]
    with engine.connect() as conn:
        for view in views:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {view}")).scalar()
            assert count > 0, f"{view} has no rows"