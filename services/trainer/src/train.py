"""Train XGBoost regressor, log to MLflow, and promote if it beats the current Production model."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load
from src.registry import get_production_metric, register_and_promote

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("trainer")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "weather-temperature")
MODEL_NAME = os.environ.get("MODEL_NAME", "weather-xgb-temperature")
DECISION_METRIC = "rmse_test"

REFERENCE_PATH = Path(os.environ.get("TRAIN_REFERENCE_PATH", "data/processed/reference.parquet"))
CURRENT_PATH = Path(os.environ.get("TRAIN_CURRENT_PATH", "data/processed/current.parquet"))

PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
}
EARLY_STOPPING_ROUNDS = 30


def compute_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {f"rmse_{prefix}": rmse, f"mae_{prefix}": mae, f"r2_{prefix}": r2}


def main() -> int:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    data = load(REFERENCE_PATH, CURRENT_PATH)

    with mlflow.start_run() as run:
        mlflow.log_params(PARAMS)
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        mlflow.log_param("n_features", len(data.feature_names))
        mlflow.log_param("n_train", len(data.X_train))
        mlflow.log_param("n_val", len(data.X_val))
        mlflow.log_param("n_test", len(data.X_test))

        model = xgb.XGBRegressor(**PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            data.X_train,
            data.y_train,
            eval_set=[(data.X_val, data.y_val)],
            verbose=False,
        )
        log.info("Trained model, best iteration: %s", model.best_iteration)

        val_pred = model.predict(data.X_val)
        test_pred = model.predict(data.X_test)
        metrics = {
            **compute_metrics(data.y_val, val_pred, "val"),
            **compute_metrics(data.y_test, test_pred, "test"),
        }
        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        signature = infer_signature(data.X_train, model.predict(data.X_train.head(5)))
        logged = mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=data.X_train.head(5),
        )
        model_uri = logged.model_uri
        log.info("Model logged at URI: %s", model_uri)

        candidate_rmse = metrics[DECISION_METRIC]
        production_rmse = get_production_metric(client, MODEL_NAME, DECISION_METRIC)

        if production_rmse is None:
            log.info("No incumbent model — promoting candidate (RMSE=%.4f)", candidate_rmse)
            register_and_promote(client, model_uri, MODEL_NAME, run.info.run_id)
            mlflow.set_tag("decision", "promoted_initial")
            return 0

        log.info("Comparing candidate RMSE=%.4f vs production RMSE=%.4f",
                 candidate_rmse, production_rmse)
        mlflow.log_metric("rmse_production_incumbent", production_rmse)

        if candidate_rmse < production_rmse:
            log.info("Candidate beats production — promoting")
            register_and_promote(client, model_uri, MODEL_NAME, run.info.run_id)
            mlflow.set_tag("decision", "promoted")
            return 0

        log.warning("Candidate worse than production — keeping incumbent")
        mlflow.set_tag("decision", "rejected")
        return 1


if __name__ == "__main__":
    sys.exit(main())