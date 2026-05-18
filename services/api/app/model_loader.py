"""MLflow model loader with thread-safe hot reload."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import RestException
from mlflow.pyfunc import PyFuncModel

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: PyFuncModel
    version: str


class ModelLoader:
    def __init__(self, model_name: str, alias: str, tracking_uri: str) -> None:
        self._model_name = model_name
        self._alias = alias
        self._tracking_uri = tracking_uri
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    @property
    def loaded(self) -> LoadedModel | None:
        with self._lock:
            return self._loaded

    def _resolve_current_version(self) -> str | None:
        try:
            version = self._client.get_model_version_by_alias(self._model_name, self._alias)
        except RestException:
            return None
        return version.version

    def load(self) -> LoadedModel | None:
        version = self._resolve_current_version()
        if version is None:
            log.warning("No model registered under %s@%s", self._model_name, self._alias)
            return None
        uri = f"models:/{self._model_name}@{self._alias}"
        log.info("Loading model %s (version %s)", uri, version)
        model = mlflow.pyfunc.load_model(uri)
        loaded = LoadedModel(model=model, version=version)
        with self._lock:
            self._loaded = loaded
        return loaded

    def reload_if_changed(self) -> tuple[bool, str | None, str | None]:
        previous_version = self._loaded.version if self._loaded else None
        current_version = self._resolve_current_version()

        if current_version is None:
            return False, previous_version, None

        if current_version == previous_version:
            return False, previous_version, current_version

        log.info("Detected new model version %s (was %s) — reloading",
                 current_version, previous_version)
        self.load()
        return True, previous_version, current_version

    def predict(self, df: pd.DataFrame) -> list[float]:
        with self._lock:
            if self._loaded is None:
                raise RuntimeError("No model loaded")
            preds = self._loaded.model.predict(df)
        return [float(p) for p in preds]