# weather-mlops

[![CI](https://github.com/akhellad/weather-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/akhellad/weather-mlops/actions/workflows/ci.yml)
[![Retrain](https://github.com/akhellad/weather-mlops/actions/workflows/retrain.yml/badge.svg)](https://github.com/akhellad/weather-mlops/actions/workflows/retrain.yml)

End-to-end MLOps pipeline for next-day temperature prediction from NOAA GSOD weather data. Demonstrates data drift detection, automated retraining with rollback, model registry with hot reload, and full reproducibility via Docker Compose.

## Architecture

```
                  ┌─────────────────────────────────────┐
                  │       GitHub Actions (cron)         │
                  └────────────────┬────────────────────┘
                                   │ 1. fetch + prepare data
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │   ┌─────────────┐  2. drift?    ┌─────────────────┐          │
   │   │ drift-monitor├──────────────►   trainer       │          │
   │   │  (Evidently) │   if yes     │   (XGBoost)     │          │
   │   └──────▲──────┘               └────────┬────────┘          │
   │          │ reference + current           │ log + register    │
   │          │                               ▼                   │
   │   ┌──────┴──────────┐                ┌──────────────┐        │
   │   │  data/processed │                │ MLflow       │        │
   │   │  (parquet)      │                │ Registry     │        │
   │   └─────────────────┘                └──────┬───────┘        │
   │                                             │ @production    │
   │                                             ▼ alias          │
   │                                      ┌──────────────┐        │
   │                                      │     api      │        │
   │                                      │  (FastAPI,   │        │
   │                                      │  hot reload) │        │
   │                                      └──────────────┘        │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
cp .env.example .env
# Edit .env: set GOOGLE_APPLICATION_CREDENTIALS_HOST to your GCP service account JSON path
docker compose up -d mlflow-server api drift-monitor
```

Then:
- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Drift report: http://localhost:8001/drift-report

To train the first model (or trigger a manual retrain):

```bash
docker compose --profile jobs run --rm trainer
```

## Retraining logic

1. `drift-monitor` exposes `/drift-report` based on an Evidently comparison between the reference dataset (historical) and the current dataset (recent).
2. GitHub Actions polls the endpoint daily (or on `workflow_dispatch`).
3. If drift is detected (share of drifted features > threshold), the `trainer` container is launched.
4. The trainer logs the new model to MLflow and compares its test RMSE against the model currently bearing the `production` alias.
5. **Better RMSE** → the alias is moved to the new version. The API picks up the new model within the polling interval (default 60s) without restart, or instantly via `POST /reload`.
6. **Worse RMSE** → the workflow fails with exit code 1, the incumbent model stays active, and a GitHub issue is opened automatically tagged `retrain-rejected`.

## Highlights

- **Automatic rollback**: a candidate model that underperforms is rejected by the trainer process itself, not after deployment. Production state is never corrupted.
- **Hot model reload**: the API polls the MLflow registry and reloads the production model in-place. No container restart, no downtime.
- **Reproducible drift simulation**: drift is injected via a deterministic seasonal shift (+6°C temperature, ×1.6 precipitation, fixed random seed). Reproducible across machines and CI runs.
- **Strict input validation**: every prediction request is validated against a Pydantic schema matching the model signature.
- **Full Docker stack**: a single `docker compose up` brings the entire system online — tracking server, serving API, drift monitor, all networked together.
- **Isolated tests**: the test suite (10 tests, ~3s) requires no MLflow server, no parquet files, no external services. Runs as easily on a laptop as in CI.

## Stack

| Component | Tool |
|---|---|
| Model | XGBoost regressor |
| Tracking & registry | MLflow 3.11 |
| Serving | FastAPI + Pydantic |
| Drift detection | Evidently |
| Data source | NOAA GSOD via BigQuery |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions |
| Dependencies | uv |
| Python | 3.11 (containers) / 3.12 (local) |

## Services

| Service | Port | Lifecycle | Purpose |
|---|---|---|---|
| `mlflow-server` | 5000 | long-running | Tracking + model registry + artifact proxy |
| `api` | 8000 | long-running | Serves predictions, hot-reloads the production model |
| `drift-monitor` | 8001 | long-running | Computes drift reports on demand |
| `trainer` | — | one-shot (`profile: jobs`) | Trains, evaluates, decides promotion |

## Project layout

```
.
├── .github/workflows/    # CI (lint + tests) and retraining workflow
├── pipelines/            # ingest.py (BigQuery → parquet), prepare.py (features + split)
├── services/
│   ├── api/              # FastAPI serving with Pydantic schemas and hot reload
│   ├── drift_monitor/    # Evidently-based drift detection with seasonal simulation
│   └── trainer/          # XGBoost training, MLflow logging, promotion logic
├── tests/                # 10 isolated unit tests
├── docker-compose.yml
├── dvc.yaml              # Data pipeline definition
└── pyproject.toml        # Root project (pipelines + local dev)
```

## Testing

```bash
uv run pytest
```

The test suite is fully isolated: no external services, no MLflow server, no parquet files required. The MLflow client is mocked, the Evidently parser is tested against a fixture, and the drift simulator is checked for determinism.

## Configuration

All runtime config goes through environment variables. See `.env.example` for the full list. Notable ones:

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` (local) / `http://mlflow-server:5000` (Docker) | MLflow server endpoint |
| `MODEL_NAME` | `weather-xgb-temperature` | Name in the registry |
| `MODEL_ALIAS` | `production` | Alias the API tracks |
| `MODEL_RELOAD_INTERVAL_SECONDS` | `60` | Polling interval for hot reload |
| `DRIFT_THRESHOLD` | `0.3` | Min share of drifted features to flag drift |
| `DRIFT_SIMULATE` | `true` | Inject reproducible seasonal drift |
| `GOOGLE_APPLICATION_CREDENTIALS_HOST` | — | Path on host to your GCP service account JSON (bind-mounted into the trainer) |

## Limitations and what's next

This project is a demonstration of the MLOps mechanics, not a full production system. Concretely:

- **MLflow state is ephemeral in CI**: each GitHub Actions run starts from a fresh volume. For a real deployment, MLflow would point at a persistent backend store (PostgreSQL) and a cloud artifact store (S3, GCS).
- **Drift is simulated**: real production drift would come from incoming serving data being logged back to a current dataset. The pipeline scaffolding is there; only the data source would change.
- **No authentication**: the API and the drift monitor are open. In production they'd sit behind an API gateway with auth and rate limiting.
- **Single-region**: the entire stack is single-host. Scaling out the API would mean splitting MLflow tracking from the registry and putting the API behind a load balancer.

## License

MIT