"""Shared pytest fixtures for the weather-mlops test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "services" / "drift_monitor"))


@pytest.fixture(autouse=True)
def _set_reload_interval_high(monkeypatch):
    monkeypatch.setenv("MODEL_RELOAD_INTERVAL_SECONDS", "999999")


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "temp_c": rng.normal(10, 5, n),
        "dewp_c": rng.normal(5, 4, n),
        "wind_ms": rng.uniform(1, 8, n),
        "prcp_mm": rng.exponential(2, n),
    })
    for col in ["temp_c", "dewp_c", "wind_ms", "prcp_mm"]:
        for lag in (1, 2, 3, 7):
            df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()
    df["day_of_year"] = df["date"].dt.dayofyear.astype("float64")
    df["month"] = df["date"].dt.month.astype("float64")
    radians = 2 * np.pi * df["day_of_year"] / 365.25
    df["doy_sin"] = np.sin(radians)
    df["doy_cos"] = np.cos(radians)
    df["target_temp"] = df["temp_c"].shift(-1).bfill()
    return df