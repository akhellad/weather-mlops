# era5-mlops

End-to-end MLOps pipeline for next-day temperature prediction from ERA5 data.

## Stack

- **MLflow** — tracking + model registry
- **FastAPI** — serving with hot model reload
- **Evidently** — data drift detection
- **XGBoost** — regression model
- **DVC** — data versioning (local remote)
- **Docker Compose** — local orchestration
- **GitHub Actions** — CI + automated retraining

## Architecture

```
GitHub Actions ──────► drift-monitor (/drift-report)
       │                      │
       │ if drift             ▼
       └──────────────► trainer ──► MLflow Registry
                                         │
                                         ▼
                                       api (hot reload)
```

## Local setup

```bash
cp .env.example .env
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
2. GitHub Actions polls the endpoint (or is triggered via `workflow_dispatch`)
3. If drift is detected, the `trainer` container is launched
4. New model is logged to MLflow and compared against the current `Production` model
5. Better RMSE → promoted to `Production`, API reloads automatically
6. Worse RMSE → job fails, current model stays, an issue is opened

## Drift simulation

Drift is simulated by injecting out-of-season data into the current dataset.
See `services/drift_monitor/src/simulator.py`.
