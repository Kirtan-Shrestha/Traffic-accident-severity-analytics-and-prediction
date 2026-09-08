# Traffic Accident Severity Analytics — Data Pipeline (Part 1)
Individual project for Data Engineering and MLOps — Project 4: Traffic Accident
Severity Analytics and Prediction. This repository covers **Part 1: the data
pipeline** — acquisition, orchestration, storage, and dashboard. Part 2
(MLOps model deployment) is a separate, later phase and is not implemented here.

## Project overview

The pipeline ingests UK road collision records (STATS19, 2023) and enriches
them with historical weather data (Open-Meteo), then validates, transforms,
and loads them into a PostgreSQL/PostGIS star schema. An Airflow DAG
orchestrates the full flow, and a Streamlit dashboard provides interactive
analysis with 7 views and 5 filters.

## Architecture

See `docs/architecture_diagram.svg` for the full diagram. In short:

STATS19 + Open-Meteo -> Ingestion -> Raw layer -> Staging + validation
-> (rejected records branch) -> Cleaned + weather-enriched
-> PostgreSQL/PostGIS (star schema) -> Analytical mart -> Streamlit dashboard


Apache Airflow orchestrates every stage from ingestion through mart-building
as a single DAG (`airflow/dags/traffic_accident_pipeline_dag.py`).

## Requirements

- Windows 10/11 (or any OS with Docker support)
- Python 3.11+
- Docker Desktop with Docker Compose v2+
- Git

## Project structure

traffic-accident-pipeline/
├── airflow/ # Airflow docker-compose setup and DAG
│ └── dags/
├── data/
│ ├── raw/ # Immutable downloaded data (gitignored)
│ ├── staging/ # Standardized, decoded (gitignored)
│ ├── cleaned/ # Validated + weather-enriched (gitignored)
│ └── rejected/ # Rejected-record logs (gitignored)
├── dashboard/ # Streamlit app
├── docs/ # Data dictionary, architecture diagram, dataset info
├── logs/ # Pipeline execution logs (gitignored)
├── sql/ # Star schema + analytical mart SQL
├── src/
│ ├── ingestion/ # Download scripts (STATS19, weather)
│ ├── transformation/ # Staging, weather join, code lookups
│ ├── validation/ # Data quality checks
│ └── database/ # PostgreSQL load + mart refresh
├── tests/ # Unit and integration tests
├── docker-compose.yml # PostgreSQL/PostGIS
├── requirements.txt
└── .env.example


## Environment variables

Copy `.env.example` to `.env` and fill in real values:

RAW_DATA_DIR=data/raw
LOG_DIR=logs
DATASET_YEAR=2023
WEATHER_GRID_PRECISION=0.3
POSTGRES_DB=traffic_accidents
POSTGRES_USER=traffic_admin
POSTGRES_PASSWORD=<your_password>
POSTGRES_HOST=localhost
POSTGRES_PORT=5432


Never commit `.env` — it's gitignored. `.env.example` documents required
variables without real credentials.

## Installation

```powershell
git clone <your-repo-url>
cd traffic-accident-pipeline
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env    # then edit .env with real values
```

## Dataset setup

The pipeline downloads data automatically — no manual download needed:

```powershell
python -m src.ingestion.download_stats19
python -m src.ingestion.download_weather
```

See `docs/dataset_info.md` for source details, license, and field descriptions.

## Database setup

Start PostgreSQL/PostGIS:

```powershell
docker compose up -d
```

Apply the schema and analytical mart:

```powershell
Get-Content sql\01_schema.sql | docker exec -i traffic_accident_db psql -U traffic_admin -d traffic_accidents
Get-Content sql\02_analytical_mart.sql | docker exec -i traffic_accident_db psql -U traffic_admin -d traffic_accidents
```

## Running the pipeline manually (without Airflow)

```powershell
python -m src.ingestion.download_stats19
python -m src.ingestion.download_weather
python -m src.transformation.stage_collisions
python -m src.transformation.stage_vehicles
python -m src.transformation.stage_casualties
python -m src.validation.validate_collisions
python -m src.transformation.join_weather
python -m src.database.load_to_postgres
python -m src.database.refresh_mart
```

## Airflow setup

```powershell
cd airflow
docker compose up airflow-init
docker compose up -d
```

Access the UI at http://localhost:8080 (login: `airflow` / `airflow`).
Unpause and trigger `traffic_accident_pipeline` to run the full pipeline
end-to-end through Airflow.

## Running the dashboard

```powershell
streamlit run dashboard\app.py
```

Opens at http://localhost:8501.

## Testing

```powershell
python -m pytest tests\ -v
```

16 tests covering validation logic, transformation/decoding logic, and
database referential integrity (requires the database to be running and
loaded for the integration tests in `test_database_integrity.py`).

## Troubleshooting

- **`ModuleNotFoundError`**: ensure your venv is activated (`venv\Scripts\activate`)
  and dependencies are installed (`pip install -r requirements.txt`).
- **Database connection errors**: confirm the container is healthy with
  `docker compose ps`, and that `.env` values match `docker-compose.yml`.
- **Airflow tasks failing on DB connection**: Airflow's containers reach the
  database via `host.docker.internal`, not `localhost` — this is handled in
  the DAG's environment overrides.
- **Weather API 429 errors**: the ingestion script includes automatic retry
  with backoff; re-running is safe and idempotent (cached locations are
  skipped).

## Data sources

- UK Road Safety Open Data (STATS19), Department for Transport — Open
  Government Licence v3.0. See `docs/dataset_info.md`.
- Open-Meteo Historical Weather API — free for non-commercial use.
