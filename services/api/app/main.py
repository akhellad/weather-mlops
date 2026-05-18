"""FastAPI server for weather temperature predictions with hot model reload."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.model_loader import ModelLoader
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "weather-xgb-temperature")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
RELOAD_INTERVAL = int(os.environ.get("MODEL_RELOAD_INTERVAL_SECONDS", "60"))

loader = ModelLoader(model_name=MODEL_NAME, alias=MODEL_ALIAS, tracking_uri=TRACKING_URI)


async def _reload_loop() -> None:
    while True:
        await asyncio.sleep(RELOAD_INTERVAL)
        try:
            changed, prev, curr = await asyncio.to_thread(loader.reload_if_changed)
            if changed:
                log.info("Hot-reloaded model: %s -> %s", prev, curr)
        except Exception:
            log.exception("Reload loop iteration failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(loader.load)
    except Exception:
        log.exception("Initial model load failed; API will start but /predict will 503")
    task = asyncio.create_task(_reload_loop())
    yield
    task.cancel()


app = FastAPI(title="Weather Temperature API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = loader.loaded
    return HealthResponse(
        status="ok",
        model_loaded=loaded is not None,
        model_version=loaded.version if loaded else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    loaded = loader.loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    df = pd.DataFrame([instance.model_dump() for instance in request.instances])
    try:
        predictions = loader.predict(df)
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(predictions=predictions, model_version=loaded.version)


@app.post("/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    changed, prev, curr = loader.reload_if_changed()
    return ReloadResponse(reloaded=changed, previous_version=prev, current_version=curr)