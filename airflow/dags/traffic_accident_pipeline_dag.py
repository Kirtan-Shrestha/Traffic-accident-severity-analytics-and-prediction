"""
Traffic Accident Severity Analytics - Data Pipeline DAG

Orchestrates the full Part 1 pipeline: extraction, validation, staging,
weather enrichment, and database/mart loading. Each task invokes an
already-tested script from the project's src/ directory.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

# Environment overrides so scripts running inside the Airflow container
# connect to the data warehouse via Docker's host bridge instead of localhost
DB_ENV = (
    "export POSTGRES_HOST=host.docker.internal && "
)

default_args = {
    "owner": "student",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="traffic_accident_pipeline",
    description="End-to-end traffic accident data pipeline: ingest, validate, transform, load, mart",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["traffic-accidents", "data-engineering"],
) as dag:

    extract_accident_data = BashOperator(
        task_id="extract_accident_data",
        bash_command=f"cd {PROJECT_DIR} && python -m src.ingestion.download_stats19",
    )

    extract_weather_data = BashOperator(
        task_id="extract_weather_data",
        bash_command=f"cd {PROJECT_DIR} && python -m src.ingestion.download_weather",
    )

    stage_collisions = BashOperator(
        task_id="stage_collisions",
        bash_command=f"cd {PROJECT_DIR} && python -m src.transformation.stage_collisions",
    )

    stage_vehicles = BashOperator(
        task_id="stage_vehicles",
        bash_command=f"cd {PROJECT_DIR} && python -m src.transformation.stage_vehicles",
    )

    stage_casualties = BashOperator(
        task_id="stage_casualties",
        bash_command=f"cd {PROJECT_DIR} && python -m src.transformation.stage_casualties",
    )

    validate_raw_data = BashOperator(
        task_id="validate_raw_data",
        bash_command=f"cd {PROJECT_DIR} && python -m src.validation.validate_collisions",
    )

    join_accident_weather = BashOperator(
        task_id="join_accident_weather",
        bash_command=f"cd {PROJECT_DIR} && python -m src.transformation.join_weather",
    )

    load_postgresql = BashOperator(
        task_id="load_postgresql",
        bash_command=f"cd {PROJECT_DIR} && {DB_ENV} python -m src.database.load_to_postgres",
    )

    build_analytical_mart = BashOperator(
        task_id="build_analytical_mart",
        bash_command=f"cd {PROJECT_DIR} && {DB_ENV} python -m src.database.refresh_mart",
    )

    # Task dependency graph
    extract_accident_data >> [stage_collisions, stage_vehicles, stage_casualties]
    extract_weather_data >> join_accident_weather
    stage_collisions >> validate_raw_data >> join_accident_weather
    join_accident_weather >> load_postgresql >> build_analytical_mart