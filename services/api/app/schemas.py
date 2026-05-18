"""Request and response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WeatherFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temp_c: float = Field(..., description="Daily mean temperature in Celsius")
    dewp_c: float
    wind_ms: float
    prcp_mm: float = Field(..., ge=0)

    temp_c_lag1: float
    temp_c_lag2: float
    temp_c_lag3: float
    temp_c_lag7: float
    dewp_c_lag1: float
    dewp_c_lag2: float
    dewp_c_lag3: float
    dewp_c_lag7: float
    wind_ms_lag1: float
    wind_ms_lag2: float
    wind_ms_lag3: float
    wind_ms_lag7: float
    prcp_mm_lag1: float
    prcp_mm_lag2: float
    prcp_mm_lag3: float
    prcp_mm_lag7: float

    day_of_year: float = Field(..., ge=1, le=366)
    month: float = Field(..., ge=1, le=12)
    doy_sin: float = Field(..., ge=-1, le=1)
    doy_cos: float = Field(..., ge=-1, le=1)


class PredictRequest(BaseModel):
    instances: list[WeatherFeatures]


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ReloadResponse(BaseModel):
    reloaded: bool
    previous_version: str | None
    current_version: str | None