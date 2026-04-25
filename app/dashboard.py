import os
import json
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException

from app.schemas import (
    SignalEntry,
    SignalContextItem,
    SignalHistoryResponse,
    ContextResponse,
    ModelInfoResponse,
    MetricsResponse,
    FeedbackRequest,
    FeedbackResponse,
)

router = APIRouter()

# Paths and config
PREDICTIONS_FILE = os.environ.get("PREDICTIONS_FILE", "data/predictions.json")
FEEDBACK_FILE = os.environ.get("FEEDBACK_FILE", "data/feedback.json")
PROCESSED_CSV = os.environ.get("PROCESSED_CSV", "data/processed/nifty50_5min_features.csv")
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model_server:8500")
SEQ_LENGTH = 30


def load_predictions() -> List[dict]:
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_predictions(entries: List[dict]):
    os.makedirs(os.path.dirname(PREDICTIONS_FILE) or ".", exist_ok=True)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def append_prediction(entry: dict):
    entries = load_predictions()
    entries.append(entry)
    save_predictions(entries)


def compute_bands(predictions: List[float], band_pct: float = 0.01):
    highs = [round(float(p) * (1 + band_pct), 3) for p in predictions]
    lows = [round(float(p) * (1 - band_pct), 3) for p in predictions]
    return highs, lows


def compute_confidence_from_context(context: List[dict]) -> float:
    try:
        closes = [float(r["Close"]) for r in context if r.get("Close") is not None]
        if not closes:
            return 50.0
        mean = np.mean(closes)
        std = np.std(closes)
        pct = std / (abs(mean) + 1e-6)
        conf = max(10.0, min(99.0, 100.0 - pct * 100.0))
        return round(float(conf), 2)
    except Exception:
        return 50.0


def compute_signal(next_pred: float, last_close: float, threshold_pct: float = 0.005) -> str:
    if next_pred > last_close * (1 + threshold_pct):
        return "BUY"
    if next_pred < last_close * (1 - threshold_pct):
        return "SELL"
    return "HOLD"


def read_context_from_csv(length: int = 20) -> List[dict]:
    if not os.path.exists(PROCESSED_CSV):
        return []
    try:
        df = pd.read_csv(PROCESSED_CSV)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
        last = df.tail(length)
        res = []
        for idx, row in last.iterrows():
            item = {
                "time_step": row["Datetime"].isoformat()
                if "Datetime" in row and not pd.isna(row["Datetime"])
                else str(idx),
                "Close": float(row["Close"]) if "Close" in row and not pd.isna(row["Close"]) else None,
                "Volume": float(row["Volume"]) if "Volume" in row and not pd.isna(row["Volume"]) else None,
                "RSI": float(row["RSI"]) if "RSI" in row and not pd.isna(row["RSI"]) else None,
                "MACD": float(row["MACD"]) if "MACD" in row and not pd.isna(row["MACD"]) else None,
            }
            res.append(item)
        return res
    except Exception:
        return []


@router.get("/signal/current", response_model=SignalEntry)
def get_current_signal():
    entries = load_predictions()
    if entries:
        return entries[-1]

    # compute prediction using last SEQ_LENGTH closes
    context = read_context_from_csv(SEQ_LENGTH)
    if not context or len(context) < SEQ_LENGTH:
        raise HTTPException(status_code=404, detail="Not enough data to build input context")

    input_data = [[float(r["Close"])] for r in context]
    payload = {"data": input_data}

    try:
        r = requests.post(f"{MODEL_SERVER_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        resp = r.json()
        predictions = [float(x) for x in resp.get("prediction", [])]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model server error: {e}")

    highs, lows = compute_bands(predictions)
    confidence = compute_confidence_from_context(context)
    last_close = float(context[-1]["Close"]) if context[-1].get("Close") is not None else 0.0
    signal = compute_signal(predictions[0], last_close)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "confidence": confidence,
        "predicted": [float(x) for x in predictions],
        "predicted_highs": highs,
        "predicted_lows": lows,
        "input_context": context,
        "last_close": last_close,
        "predicted_high": highs[0] if highs else None,
        "predicted_low": lows[0] if lows else None,
        "actual_close": None,
        "outcome": "Pending",
    }

    append_prediction(entry)
    return entry


@router.get("/signal/history", response_model=SignalHistoryResponse)
def get_signal_history(limit: int = 30):
    entries = load_predictions()
    return {"history": entries[-limit:]}


@router.get("/signal/context", response_model=ContextResponse)
def get_signal_context(length: int = 20):
    context = read_context_from_csv(length)
    return {"input_context": context}


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    api_status = "offline"
    try:
        r = requests.get(f"{MODEL_SERVER_URL}/ready", timeout=2)
        if r.status_code == 200:
            api_status = r.json().get("status", "ready")
    except Exception:
        api_status = "offline"

    entries = load_predictions()
    last_retrain = entries[-1]["timestamp"] if entries else None
    data_freshness = None
    if os.path.exists(PROCESSED_CSV):
        data_freshness = datetime.utcfromtimestamp(os.path.getmtime(PROCESSED_CSV)).isoformat()

    drift_status = "No drift"
    try:
        if os.path.exists("drift_detection/drift_report.json"):
            with open("drift_detection/drift_report.json", "r") as f:
                rep = json.load(f)
                drift_status = "Drift detected" if rep.get("drift_detected", False) else "No drift"
    except Exception:
        pass

    return {
        "api_status": api_status,
        "model_version": None,
        "last_retrain": last_retrain,
        "data_freshness": data_freshness,
        "drift_status": drift_status,
    }


def compute_accuracy(entries: List[dict], window: int):
    if not entries:
        return None
    filtered = [e for e in entries if e.get("outcome") in ("Correct", "Incorrect")]
    if not filtered:
        return None
    sliced = filtered[-window:]
    total = len(sliced)
    correct = sum(1 for e in sliced if e.get("outcome") == "Correct")
    return round(correct / total * 100, 2)


@router.get("/model/metrics", response_model=MetricsResponse)
def get_model_metrics():
    entries = load_predictions()
    accuracy_7 = compute_accuracy(entries, 7)
    accuracy_30 = compute_accuracy(entries, 30)
    accuracy_60 = compute_accuracy(entries, 60)
    spark = []
    for e in entries[-30:]:
        o = e.get("outcome")
        if o == "Correct":
            spark.append(1)
        elif o == "Incorrect":
            spark.append(0)
        else:
            spark.append(0)
    return {"accuracy_7": accuracy_7, "accuracy_30": accuracy_30, "accuracy_60": accuracy_60, "sparkline": spark}


@router.post("/feedback", response_model=FeedbackResponse)
def post_feedback(payload: FeedbackRequest):
    os.makedirs(os.path.dirname(FEEDBACK_FILE) or ".", exist_ok=True)
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []
    except Exception:
        feedbacks = []

    fb = {
        "timestamp": payload.timestamp or datetime.now(timezone.utc).isoformat(),
        "time_step": payload.time_step,
        "rating": int(payload.rating),
        "comment": payload.comment,
    }
    feedbacks.append(fb)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedbacks, f, indent=2)
    return {"status": "ok"}
