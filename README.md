## Local setup

```bash
cp .env.example .env
# Edit .env to set GOOGLE_APPLICATION_CREDENTIALS to your GCP service account JSON
docker compose up -d mlflow-server api drift-monitor
```

## Manual retraining

```bash
docker compose run --rm trainer
```

## Data pipeline

```bash
dvc repro
```

## Services

| Service | Port | Purpose |
|---|---|---|
| mlflow-server | 5000 | Tracking + registry |
| api | 8000 | Model serving |
| drift-monitor | 8001 | Drift reports |
| trainer | — | Job-only container |

## Retraining logic

1. `drift-monitor` exposes `/drift-report` based on Evidently comparison between reference and current data
2. GitHub Actions polls the endpoint (scheduled or via `workflow_dispatch`)
3. If drift is detected, the `trainer` container is launched
4. The new model is logged to MLflow and compared against the current `production` alias
5. Better RMSE → promoted to `production` via alias swap; API reloads the model without restart
6. Worse RMSE → the workflow fails, the incumbent stays, an issue is opened automatically

## Drift simulation

Drift is simulated by injecting a deterministic seasonal shift (+6°C temperature, +4°C dew point, ×1.6 precipitation) into the current dataset, with a fixed seed for reproducibility. See `services/drift_monitor/src/simulator.py`.

## Testing

```bash
uv run pytest
```

The test suite is fully isolated: no external services, no MLflow server, no parquets required.