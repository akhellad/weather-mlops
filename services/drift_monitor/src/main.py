"""FastAPI server exposing the data drift report endpoint."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.drift_report import compute_drift
from src.simulator import inject_seasonal_drift

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("drift-monitor")


def _default_data_path(filename: str) -> str:
    try:
        root = Path(__file__).resolve().parents[3]
    except IndexError:
        return f"data/processed/{filename}"
    return str(root / "data" / "processed" / filename)


REFERENCE_PATH = Path(
    os.environ.get("DRIFT_REFERENCE_PATH", _default_data_path("reference.parquet"))
)
CURRENT_PATH = Path(
    os.environ.get("DRIFT_CURRENT_PATH", _default_data_path("current.parquet"))
)
THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.3"))
SIMULATE_DRIFT = os.environ.get("DRIFT_SIMULATE", "true").lower() == "true"

EXCLUDE_COLUMNS = ("date", "target_temp")

log.info("Reference path: %s (exists=%s)", REFERENCE_PATH, REFERENCE_PATH.exists())
log.info("Current path: %s (exists=%s)", CURRENT_PATH, CURRENT_PATH.exists())

app = FastAPI(title="Weather Drift Monitor")


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns])


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "reference_exists": REFERENCE_PATH.exists(),
        "current_exists": CURRENT_PATH.exists(),
        "simulate_drift": SIMULATE_DRIFT,
    }


@app.get("/drift-report")
def drift_report() -> dict:
    if not REFERENCE_PATH.exists() or not CURRENT_PATH.exists():
        raise HTTPException(status_code=503, detail="Reference or current data not available")

    reference = _load_features(REFERENCE_PATH)
    current = _load_features(CURRENT_PATH)

    if SIMULATE_DRIFT:
        current = inject_seasonal_drift(current)

    try:
        summary = compute_drift(reference, current, threshold=THRESHOLD)
    except Exception as exc:
        log.exception("Drift computation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "drift_detected": summary.drift_detected,
        "drift_share": summary.drift_share,
        "n_drifted": summary.n_drifted,
        "n_features": summary.n_features,
        "threshold": summary.threshold,
        "simulated": SIMULATE_DRIFT,
        "per_column": summary.per_column,
    }