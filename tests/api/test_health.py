"""Test that the /health endpoint responds correctly."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_endpoint_responds_ok():
    with patch("app.main.loader") as mock_loader:
        mock_loader.loaded = None
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["model_version"] is None