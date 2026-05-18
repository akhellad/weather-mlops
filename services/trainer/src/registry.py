"""MLflow Model Registry helpers: promotion logic and Production metric lookup."""

from __future__ import annotations

import logging

from mlflow import MlflowClient
from mlflow.exceptions import RestException

log = logging.getLogger(__name__)

PRODUCTION_ALIAS = "production"


def get_production_metric(client: MlflowClient, model_name: str, metric: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except RestException:
        log.info("No model registered yet under '%s' alias", PRODUCTION_ALIAS)
        return None

    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric)
    if value is None:
        log.warning("Production model run %s has no '%s' metric", version.run_id, metric)
    return value


def _ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except RestException:
        log.info("Creating registered model '%s'", model_name)
        client.create_registered_model(model_name)


def register_and_promote(
    client: MlflowClient,
    model_uri: str,
    model_name: str,
    run_id: str,
) -> str:
    _ensure_registered_model(client, model_name)
    version = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version.version)
    log.info("Registered %s v%s and set alias '%s'", model_name, version.version, PRODUCTION_ALIAS)
    return version.version