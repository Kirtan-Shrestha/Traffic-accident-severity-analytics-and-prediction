# Traffic Accident Severity Analytics and Prediction

Individual project for Data Engineering and MLOps - Project 4: Traffic Accident
Severity Analytics and Prediction. This repository covers both parts of the
project: **Part 1 - the data pipeline** (acquisition, orchestration, storage,
and dashboard) and **Part 2 - the MLOps extension** (model training, tracking,
registry, serving, monitoring, and retraining strategy).

## Project overview

The pipeline ingests UK road collision records (STATS19, 2023) and enriches
them with historical weather data (Open-Meteo), then validates, transforms,
and loads them into a PostgreSQL/PostGIS star schema. An Airflow DAG
orchestrates the full flow, and a Streamlit dashboard provides interactive
analysis with 7 views and 5 filters, plus a live severity-prediction page.

On top of that, Part 2 trains and compares classifiers (Random Forest,
XGBoost, LightGBM) to predict accident severity, tracks every run in MLflow,
packages the winning model with its preprocessing into a single versioned
pipeline, serves it through a containerized FastAPI service, and monitors it
for data drift and performance degradation over time.

## Architecture

See `docs/architecture_diagram.svg` for the Part 1 pipeline diagram. In short:

Part 1 (data pipeline):
STATS19 + Open-Meteo -> Ingestion -> Raw layer -> Staging + validation
-> (rejected records branch) -> Cleaned + weather-enriched
-> PostgreSQL/PostGIS (star schema) -> Analytical mart -> Streamlit dashboard

Apache Airflow orchestrates every stage from ingestion through mart-building
as a single DAG (`airflow/dags/traffic_accident_pipeline_dag.py`).

Part 2 (MLOps):
Feature table (PostgreSQL) -> chronological train/val/test split
-> train + compare models (MLflow tracking) -> package preprocessing + model
as one pyfunc pipeline -> register in MLflow Model Registry (production alias)
-> serve via FastAPI (containerized with Docker) -> Streamlit prediction page
-> drift + performance monitoring (PSI, Fatal-class recall) -> retraining

## Requirements

- Windows 10/11 (or any OS with Docker support)
- Python 3.11+
- Docker Desktop with Docker Compose v2+
- Git

## Project structure

traffic-accident-pipeline/
|-- airflow/                        # Airflow docker-compose setup and DAG
|   `-- dags/
|-- data/
|   |-- raw/                        # Immutable downloaded data (gitignored)
|   |-- staging/                    # Standardized, decoded (gitignored)
|   |-- cleaned/                    # Validated + weather-enriched (gitignored)
|   `-- rejected/                   # Rejected-record logs (gitignored)
|-- dashboard/                      # Streamlit app (analytics + prediction page)
|-- docs/                           # Data dictionary, architecture diagram,
|                                    # dataset info, monitoring/retraining doc
|-- logs/                           # Pipeline execution logs (gitignored)
|-- sql/                            # Star schema + analytical mart SQL
|-- src/
|   |-- ingestion/                  # Download scripts (STATS19, weather)
|   |-- transformation/             # Staging, weather join, code lookups
|   |-- validation/                 # Data quality checks
|   |-- database/                   # PostgreSQL load + mart refresh
|   |-- ml/                         # Part 2: data_prep, model training,
|   |                               # inference pipeline, model registration
|   |-- api/                        # Part 2: FastAPI service (schemas, main)
|   `-- monitoring/                 # Part 2: drift + performance monitoring
|-- tests/                          # Unit and integration tests (Part 1 + 2)
|-- mlruns/                         # Local MLflow tracking data (gitignored)
|-- docker-compose.yml              # PostgreSQL/PostGIS, MLflow, FastAPI
|-- Dockerfile.api                  # FastAPI service image
|-- requirements.txt                # Full project / Part 1 dependencies
|-- requirements-api.txt            # Minimal dependencies for the API container
|-- pytest.ini
`-- .env.example

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

Never commit `.env` - it's gitignored. `.env.example` documents required
variables without real credentials. The FastAPI container reads
`MLFLOW_TRACKING_URI` from `docker-compose.yml` (defaults to
`http://mlflow:5000` inside the Docker network).

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

The pipeline downloads data automatically - no manual download needed:

```powershell
python -m src.ingestion.download_stats19
python -m src.ingestion.download_weather
```

See `docs/dataset_info.md` for source details, license, and field descriptions.

## Database setup

Start PostgreSQL/PostGIS (and, for Part 2, MLflow and the API - see below):

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

Opens at http://localhost:8501. Use the sidebar to switch between the
**Analytics Dashboard** and the **Predict Severity** page (the latter calls
the FastAPI service - see Part 2 below).

## Testing

```powershell
pytest -v
```

30 tests covering: Part 1 validation logic, transformation/decoding logic,
and database referential integrity (requires the database to be running and
loaded for the integration tests in `test_database_integrity.py`); and Part 2
chronological data-split correctness, inference-pipeline encoding
correctness (including unseen-category handling), and the FastAPI service's
`/health` and `/predict` endpoints (model is stubbed in these tests, so they
run without a live MLflow server).

## Troubleshooting

- **`ModuleNotFoundError`**: ensure your venv is activated (`venv\Scripts\activate`)
  and dependencies are installed (`pip install -r requirements.txt`).
- **Database connection errors**: confirm the container is healthy with
  `docker compose ps`, and that `.env` values match `docker-compose.yml`.
- **Airflow tasks failing on DB connection**: Airflow's containers reach the
  database via `host.docker.internal`, not `localhost` - this is handled in
  the DAG's environment overrides.
- **Weather API 429 errors**: the ingestion script includes automatic retry
  with backoff; re-running is safe and idempotent (cached locations are
  skipped).
- **LightGBM crashes on Windows with an access-violation OSError**: caused by
  import order - always `import lightgbm as lgb` before importing pandas or
  any module that imports pandas (e.g. `src.ml.data_prep`).

## Data sources

- UK Road Safety Open Data (STATS19), Department for Transport - Open
  Government Licence v3.0. See `docs/dataset_info.md`.
- Open-Meteo Historical Weather API - free for non-commercial use.

---

# Part 2: MLOps Extension

Part 2 turns the Part 1 feature table into a served, monitored machine
learning system: training and comparing classifiers, tracking experiments,
packaging and registering the best model, serving it over an API, and
monitoring it for drift and performance degradation, with a documented
retraining strategy.

## Part 2 architecture

1. **Experiment tracking** - MLflow server (Dockerized), tracking URI
   `http://localhost:5000` from the host / `http://mlflow:5000` from
   other containers. Artifacts are served via MLflow's HTTP-proxied
   artifact store (`--artifacts-destination` + `--serve-artifacts`), not a
   plain filesystem path, so artifact access works correctly across both
   host-run training scripts and containerized services.
2. **Data split** - `src/ml/data_prep.py` loads the feature table from
   `ml_accident_features` and splits it chronologically (train = Jan-Aug,
   validation = Sep-Oct, test = Nov-Dec 2023) to avoid temporal leakage.
3. **Class imbalance** - handled via `class_weight="balanced"`
   (scikit-learn / LightGBM) or `sample_weight` from
   `compute_class_weight` (XGBoost), since Fatal-severity accidents are
   rare relative to Slight/Serious.
4. **Model comparison** - `src/ml/train_tree_models.py` trains Random
   Forest, XGBoost, and LightGBM, logging recall, macro-F1, and confusion
   matrices per class to MLflow for each.
5. **Packaging** - `src/ml/inference_pipeline.py` wraps the chosen model
   (LightGBM) together with its one-hot encoding step in a single
   `mlflow.pyfunc.PythonModel`, so a raw feature record can be scored
   directly.
6. **Registration** - `src/ml/register_model.py` logs and registers the
   packaged pipeline as `accident_severity_classifier` in the MLflow Model
   Registry, promoted via the `production` alias.
7. **Serving** - `src/api/main.py` (FastAPI) loads
   `models:/accident_severity_classifier@production` at startup and
   exposes `/health` and `/predict`, containerized via `Dockerfile.api`
   and orchestrated as the `api` service in `docker-compose.yml`.
8. **Dashboard integration** - `dashboard/app.py`'s "Predict Severity" page
   calls the FastAPI `/predict` endpoint and displays the result, with a
   warning banner for Fatal predictions.
9. **Monitoring** - `src/monitoring/drift_report.py` computes Population
   Stability Index (PSI) for target/categorical/geographic drift between
   the training and test windows, and tracks Fatal-class recall and
   false-negative rate degradation; results are logged to MLflow. The
   FastAPI service also logs path/status/latency for every request via
   middleware, for serving-time performance and failure monitoring.
10. **Retraining strategy** - documented in full in
    `docs/monitoring_and_retraining.md`, including concrete PSI and
    recall-degradation thresholds that trigger retraining.

## Running Part 2 end to end

Start all services (PostgreSQL, MLflow, and the API):

```powershell
docker compose up -d
docker compose ps
```

Train and compare models (from the host, with the venv activated):

```powershell
venv\Scripts\activate
python -m src.ml.train_tree_models
```

Package and register the production model:

```powershell
python -m src.ml.register_model
```

Verify the API is serving the newly registered model:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

Run drift and performance monitoring:

```powershell
python -m src.monitoring.drift_report
```

View experiment tracking and the model registry at
http://localhost:5000. View live predictions at http://localhost:8501
(Predict Severity page) or directly via `POST http://localhost:8000/predict`
(see `src/api/schemas.py` for the exact request fields).

## Model lifecycle, monitoring, and retraining criteria

See `docs/monitoring_and_retraining.md` for the full write-up: the
train -> validate -> package -> register -> serve -> monitor -> retrain
lifecycle, what is monitored and why, the observed drift/performance
findings from this project's data, and the concrete thresholds that
trigger retraining.
