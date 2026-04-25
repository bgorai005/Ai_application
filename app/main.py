# app/main.py
from fastapi import FastAPI, HTTPException
import numpy as np
import os
import requests
from app.schemas import PredictRequest, PredictResponse
from prometheus_fastapi_instrumentator import Instrumentator
import importlib


app = FastAPI()
Instrumentator().instrument(app).expose(app)



# Dashboard router (adds /signal, /model, /feedback endpoints)
try:
    from app.dashboard import router as dashboard_router

    app.include_router(dashboard_router)
except Exception:
    # Router missing or failing to import should not break the main API
    pass

# Model-server URL (when deployed via docker-compose the hostname is `model_server`)
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model_server:8500")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # Prefer checking remote model-server readiness
    try:
        r = requests.get(f"{MODEL_SERVER_URL}/ready", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # Fallback: attempt to import the local model module lazily (avoids requiring torch in gateway image)
    try:
        model_module = importlib.import_module("app.model")
        local_model_obj = getattr(model_module, "model", None)
        if local_model_obj is not None:
            return {"status": "ready", "source": "local_model"}
    except Exception:
        pass

    return {"status": "not_ready"}


@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    payload = {"data": request.data}

    # Try model-server first for separation of concerns
    try:
        r = requests.post(f"{MODEL_SERVER_URL}/predict", json=payload, timeout=3)
        if r.status_code == 200:
            json_resp = r.json()
            return PredictResponse(**json_resp)
    except Exception:
        pass

    # Local fallback
    input_data = np.array(request.data)
    try:
        # Lazy import to avoid importing torch unless fallback is needed
        model_module = importlib.import_module("app.model")
        local_predict = getattr(model_module, "predict_stock_price", None)
        if local_predict is not None:
            output = local_predict(input_data, steps=30)
            return PredictResponse(prediction=[float(val) for val in output])
    except Exception:
        pass

    # If both remote and local model are unavailable, return an HTTP 503
    raise HTTPException(status_code=503, detail="Model not available")
