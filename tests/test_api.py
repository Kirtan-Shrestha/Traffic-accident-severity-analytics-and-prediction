import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


class _FakeModel:
    """Stand-in for the loaded MLflow pyfunc pipeline; avoids needing a
    live MLflow server during tests."""

    def predict(self, row: pd.DataFrame):
        return ["Slight"] * len(row)


@pytest.fixture
def client_with_model():
    api_main.state["model"] = _FakeModel()
    api_main.state["model_version"] = "99"
    client = TestClient(api_main.app)
    yield client
    api_main.state["model"] = None
    api_main.state["model_version"] = None


SAMPLE_PAYLOAD = {
    "year": 2023, "month": 10, "hour": 17, "is_weekend": False,
    "speed_limit": 30, "latitude": 51.5, "longitude": -0.1,
    "number_of_vehicles": 2, "temperature_2m": 12.5, "precipitation": 0.0,
    "wind_speed_10m": 15.0, "weekday_name": "Wednesday",
    "road_type_label": "Single carriageway",
    "weather_conditions_label": "Fine no high winds",
    "temperature_band": "10-20", "precipitation_level": "None",
    "urban_or_rural_label": "Urban",
}


def test_health_model_loaded(client_with_model):
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "99"


def test_health_model_not_loaded():
    api_main.state["model"] = None
    api_main.state["model_version"] = None
    client = TestClient(api_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "model_not_loaded"


def test_predict_success(client_with_model):
    resp = client_with_model.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_severity"] == "Slight"
    assert body["model_version"] == "99"


def test_predict_missing_field_returns_422(client_with_model):
    bad_payload = dict(SAMPLE_PAYLOAD)
    del bad_payload["year"]
    resp = client_with_model.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_without_model_returns_503():
    api_main.state["model"] = None
    api_main.state["model_version"] = None
    client = TestClient(api_main.app)
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 503
