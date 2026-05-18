"""Test that /predict validates input strictly via Pydantic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _valid_instance() -> dict:
    feature_names = [
        "temp_c", "dewp_c", "wind_ms", "prcp_mm",
        "temp_c_lag1", "temp_c_lag2", "temp_c_lag3", "temp_c_lag7",
        "dewp_c_lag1", "dewp_c_lag2", "dewp_c_lag3", "dewp_c_lag7",
        "wind_ms_lag1", "wind_ms_lag2", "wind_ms_lag3", "wind_ms_lag7",
        "prcp_mm_lag1", "prcp_mm_lag2", "prcp_mm_lag3", "prcp_mm_lag7",
        "day_of_year", "month", "doy_sin", "doy_cos",
    ]
    instance = {name: 1.0 for name in feature_names}
    instance["day_of_year"] = 180.0
    instance["month"] = 6.0
    instance["doy_sin"] = 0.5
    instance["doy_cos"] = 0.5
    instance["prcp_mm"] = 0.0
    return instance


def test_predict_rejects_unknown_field():
    with patch("app.main.loader") as mock_loader:
        mock_loader.loaded = None
        from app.main import app

        instance = _valid_instance() | {"extra_field": 42.0}
        with TestClient(app) as client:
            response = client.post("/predict", json={"instances": [instance]})

    assert response.status_code == 422


def test_predict_rejects_out_of_range():
    with patch("app.main.loader") as mock_loader:
        mock_loader.loaded = None
        from app.main import app

        instance = _valid_instance() | {"month": 99.0}
        with TestClient(app) as client:
            response = client.post("/predict", json={"instances": [instance]})

    assert response.status_code == 422


def test_predict_returns_503_without_model():
    with patch("app.main.loader") as mock_loader:
        mock_loader.loaded = None
        from app.main import app

        with TestClient(app) as client:
            response = client.post("/predict", json={"instances": [_valid_instance()]})

    assert response.status_code == 503


def test_predict_returns_predictions_with_model():
    fake_loaded = MagicMock()
    fake_loaded.version = "42"

    with patch("app.main.loader") as mock_loader:
        mock_loader.loaded = fake_loaded
        mock_loader.predict.return_value = [10.5]

        from app.main import app

        with TestClient(app) as client:
            response = client.post("/predict", json={"instances": [_valid_instance()]})

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [10.5]
    assert body["model_version"] == "42"