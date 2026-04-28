# Architecture Overview

## High-Level Design

The application is split into four independently deployable concerns:

1. `streamlit_app.py` provides the user-facing dashboard.
2. `app/main.py` acts as the API gateway for prediction requests.
3. `app/model_server.py` hosts the inference runtime and exposes model prediction endpoints.
4. `data_pipeline/` and `ml_pipeline/` handle ingestion, feature engineering, training, evaluation, and artifact generation.

The services are connected through REST calls and shared filesystem artifacts mounted through Docker Compose.

```text
Streamlit UI -> FastAPI Gateway -> Model Server -> Model Artifacts
       |                |
       |                +-> signal history / feedback / model health
       |
       +-> data views from processed CSV files

Data Pipeline -> processed features -> ML Training -> MLflow + models/
Monitoring -> Prometheus -> Grafana
```

## Component Boundaries

- The frontend does not call the model directly.
- The gateway can proxy prediction requests to the model server and fall back to a local loader when needed.
- The data pipeline writes raw and processed datasets to disk.
- The training job reads processed data, logs to MLflow, and stores model artifacts in `models/`.
- Prometheus scrapes the backend services and the custom system exporter.

## API Endpoints

| Service | Endpoint | Method | Input | Output |
| --- | --- | --- | --- | --- |
| `app/main.py` | `/health` | GET | None | `{"status": "ok"}` |
| `app/main.py` | `/ready` | GET | None | readiness JSON |
| `app/main.py` | `/predict` | POST | `PredictRequest` | `PredictResponse` |
| `app/dashboard.py` | `/signal/current` | GET | None | latest signal entry |
| `app/dashboard.py` | `/signal/history?limit=N` | GET | query param `limit` | history list |
| `app/dashboard.py` | `/signal/context?length=N` | GET | query param `length` | context rows |
| `app/dashboard.py` | `/signal/accuracy` | GET | None | accuracy metrics |
| `app/dashboard.py` | `/model/info` | GET | None | model health info |
| `app/dashboard.py` | `/model/metrics` | GET | None | accuracy + sparkline |
| `app/dashboard.py` | `/feedback` | POST | `FeedbackRequest` | `{"status": "ok"}` |

## Low-Level Design Notes

- The request/response schemas are defined in `app/schemas.py`.
- Prediction requests contain a 2D list of floats with shape `(sequence_length, input_size)`.
- Signal entries include timestamp, signal label, confidence, predicted bands, and input context.
- Feedback requests store rating, optional comment, and optional timestamp metadata.

## Operational Notes

- MLflow tracks experiment runs, parameters, metrics, and artifacts from the training job.
- Prometheus collects application and system metrics for live dashboards.
- Grafana visualizes the monitored metrics in near real time.
