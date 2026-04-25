from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import numpy as np
import threading
import time
import json
import os
import logging

# Import prediction function and model object (loaded at module import in app.model)
from app.model import predict_stock_price, model as loaded_model


app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Gauge exposed to Prometheus to signal data drift
drift_gauge = Gauge("drift_detected", "1 if drift detected, else 0")


def _monitor_drift_file(path: str = "drift_detection/drift_report.json", poll_interval: int = 60):
    while True:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    report = json.load(f)
                drift = bool(report.get("drift_detected", False))
                drift_gauge.set(1 if drift else 0)
            else:
                drift_gauge.set(0)
        except Exception as e:
            logging.debug(f"Failed reading drift file: {e}")
        time.sleep(poll_interval)


@app.on_event("startup")
def startup_event():
    # Start a background thread to monitor drift report updates and expose metric
    thread = threading.Thread(target=_monitor_drift_file, daemon=True)
    thread.start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready" if loaded_model is not None else "not_ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    input_data = np.array(request.data)
    preds = predict_stock_price(input_data, steps=30)
    return PredictResponse(prediction=[float(x) for x in preds])
