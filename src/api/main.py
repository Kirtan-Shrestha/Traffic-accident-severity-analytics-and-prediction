"""
FastAPI service exposing the registered accident severity model for
real-time inference. Loads the "production"-aliased MLflow pyfunc
pipeline once at startup (preprocessing + model bundled together), so
each request only needs to supply raw feature values.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import lightgbm as lgb  # noqa: F401 - import before pandas/mlflow, see train_tree_models.py
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient

from src.api.schemas import AccidentFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "accident_severity_classifier"
MODEL_ALIAS = "production"

state = {"model": None, "model_version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Loading model models:/{MODEL_NAME}@{MODEL_ALIAS} from {tracking_uri}")

    state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    state["model_version"] = mv.version
    logger.info(f"Model loaded, version {state['model_version']}")

    yield

    state["model"] = None


app = FastAPI(title="Accident Severity Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs every request's path, status code, and latency -- the basic
    signal needed to monitor API performance, latency and failure rate
    in production.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"request path={request.url.path} status={response.status_code} latency_ms={latency_ms:.1f}"
        )
        return response
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception(f"request path={request.url.path} FAILED latency_ms={latency_ms:.1f}")
        raise
    
@app.get("/health")
def health():
    return {
        "status": "ok" if state["model"] is not None else "model_not_loaded",
        "model_version": state["model_version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: AccidentFeatures):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        row = pd.DataFrame([features.model_dump()])
        pred = state["model"].predict(row)[0]
        return PredictionResponse(
            predicted_severity=pred, model_version=str(state["model_version"])
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))