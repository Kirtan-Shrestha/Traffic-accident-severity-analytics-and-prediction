"""
Pydantic request/response models for the accident severity prediction
API. Field names match the raw columns the trained pipeline's _encode()
step expects (src/ml/inference_pipeline.py) -- clients send plain
feature values, not pre-encoded ones.
"""
from pydantic import BaseModel, Field


class AccidentFeatures(BaseModel):
    year: int = Field(..., examples=[2023])
    month: int = Field(..., ge=1, le=12)
    hour: int = Field(..., ge=0, le=23)
    is_weekend: bool
    speed_limit: int
    latitude: float
    longitude: float
    number_of_vehicles: int
    temperature_2m: float
    precipitation: float
    wind_speed_10m: float
    weekday_name: str = Field(..., examples=["Wednesday"])
    road_type_label: str = Field(..., examples=["Single carriageway"])
    weather_conditions_label: str = Field(..., examples=["Fine no high winds"])
    temperature_band: str = Field(..., examples=["10-20"])
    precipitation_level: str = Field(..., examples=["None"])
    urban_or_rural_label: str = Field(..., examples=["Urban"])


class PredictionResponse(BaseModel):
    predicted_severity: str
    model_version: str