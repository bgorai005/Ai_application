# Ai_application# Nifty50 Stock Price Prediction & Monitoring App (MLOps Project)

This project is an end-to-end AI application that predicts Nifty50 stock prices using 5-minute interval data. It is designed with MLOps principles in mind — focusing on automation, reproducibility, monitoring, and modular pipelines.

---

## Features
- Fetches 5-minute interval stock data for `Nifty50 Index` via `yfinance`
- Validates and transforms data with rolling averages
- 🛠Modular pipeline structure (Ingest → Validate → Transform)
- Version-controlled data pipeline with DVC
- Local storage of raw and processed datasets
- Ready for integration with MLflow, Docker, Prometheus & Grafana


## Architecture 
```
+-------------------------+       +---------------------+      +-------------------+
|   Frontend Web App     | <---> | Backend (Model API) | <--> |  Model Inference   |
|     (Streamlit)        |       |     (FastAPI)       |      |(MLflow, .pth, .pkl)|
+-------------------------+       +---------------------+      +-------------------+
                                           |
                                           v
                                 +----------------------+
                                 | Prometheus Monitoring|
                                 +----------------------+
                                           |
                                           v
                                   +------------------+
                                   | Grafana Dashboard|
                                   +------------------+  
All Dockerized together:
- Frontend
- Backend
- Model server
- MLflow
- Prometheus
- Metrics Exporter
- Grafana

```


## Folder Structure
```
.
├── data/
│   ├── raw/
│   │   └── nifty50_5min.csv
│   └── processed/
│       └── nifty50_5min_features.csv

├── models/
│   ├── best_model.pth
│   └── scaler.pkl

├── data_pipeline/
│   ├── ingest.py
│   ├── validate.py
│   ├── transform.py
│   └── pipeline.py

ml_pipeline/
|    ├── model_architecture.py    # model class only
|    └── train.py                 # train script

├── app/                      # FastAPI Inference API
│   ├── main.py               # FastAPI app (entry point)
│   ├── model.py              # Load model and predict
│   ├── schemas.py            # Request/Response data models
│
├── system-metric-exporter
│   ├── Dockerfile
│   ├── metrics_exporter.py
│   └── requirements.txt
|
drift_detection/
├── detect_drift.py
├── drift_report.json
└── __init__.py
|
├── dvc.yaml
├── .dvc/
├── .git/
├── docker-compose.yml
├── dockerfile
└── requirements.txt
```

## Data pipeline behavior

- The data pipeline now appends newly downloaded raw data to `data/raw/nifty50_5min.csv` instead of overwriting it. Appended rows are deduplicated by the `Datetime` column (the latest row for a given timestamp is kept).
- After raw data is appended, the pipeline recomputes features across the full dataset and writes the result to `data/processed/nifty50_5min_features.csv`. The processed file is replaced with the recomputed full dataset so rolling features (e.g., moving averages) are computed correctly across the entire history.
- To run the pipeline:

```bash
python data_pipeline/pipeline.py
```

## Model Training and Serving

The project uses the `models/` directory as the single model artifact location.

- Training saves the fitted scaler to `models/scaler.pkl`.
- Training saves the best PyTorch model weights to `models/best_model.pth`.
- The FastAPI model server loads artifacts from `models/` during prediction.
- The old singular `model/` directory is not used by the prediction service.

The training workflow is:

```text
data_pipeline/pipeline.py
  -> downloads latest Nifty50 data
  -> validates raw data
  -> generates rolling features
  -> writes data/processed/nifty50_5min_features.csv

ml_pipeline/train.py
  -> reads processed data
  -> trains the LSTM model
  -> saves models/scaler.pkl and models/best_model.pth
  -> logs metrics and artifacts to MLflow
```

## Docker Compose Workflow

The Compose stack includes a one-shot `trainer` service. By default, it exits immediately so the app starts quickly using the existing model artifacts in `models/`.

Run the full app stack without retraining:

```bash
docker compose up --build
```

Run latest data download and model training first, then start the app stack:

```bash
RUN_TRAINING=true docker compose up --build
```

`RUN_TRAINING=true` makes the `trainer` service wait for MLflow, run the data pipeline, train the model, and write updated artifacts into `models/`. After that, `model_server`, `backend_app`, and `streamlit_app` start normally.

> Docker Compose does not support a custom `--train` flag directly, so this project uses the `RUN_TRAINING=true` environment variable as the training switch.

### Docker Services

- `trainer`: optional data refresh and model training service.
- `mlflow`: experiment tracking server at `http://localhost:5000`, built from the root Dockerfile so MLflow is installed at image build time.
- `model_server`: FastAPI inference service at `http://localhost:8500`.
- `backend_app`: API gateway at `http://localhost:8000`.
- `streamlit_app`: frontend at `http://localhost:8501`.
- `prometheus`: metrics collection at `http://localhost:9090`.
- `grafana`: dashboard UI at `http://localhost:3000`.


## Setup Instructions

### Clone the Repo and Set Up The Environment

```bash
git clone https://github.com/amar-at-iitm/End_to_end_AI_application
cd End_to_end_AI_application
```

```bash
pip install -r requirements.txt
```
