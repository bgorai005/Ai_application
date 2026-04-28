# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional


class PredictRequest(BaseModel):
    data: List[List[float]]  # 2D List: (seq_length, input_size)


class PredictResponse(BaseModel):
    #prediction: float        # for single prediction
    prediction: List[float]  # for multiple predictions


# Dashboard / UI models
class SignalContextItem(BaseModel):
    time_step: str
    Close: Optional[float] = None
    Volume: Optional[float] = None
    RSI: Optional[float] = None
    MACD: Optional[float] = None


class SignalEntry(BaseModel):
    timestamp: str
    signal: str
    confidence: float
    predicted: List[float]
    predicted_highs: List[float]
    predicted_lows: List[float]
    input_context: List[SignalContextItem]
    last_close: float
    predicted_high: float
    predicted_low: float
    actual_close: Optional[float] = None
    outcome: Optional[str] = None


class SignalHistoryResponse(BaseModel):
    history: List[SignalEntry]


class ContextResponse(BaseModel):
    input_context: List[SignalContextItem]


class ModelInfoResponse(BaseModel):
    api_status: str
    model_version: Optional[str] = None
    last_retrain: Optional[str] = None
    data_freshness: Optional[str] = None
    drift_status: Optional[str] = None


class MetricsResponse(BaseModel):
    accuracy_7: Optional[float] = None
    accuracy_30: Optional[float] = None
    accuracy_60: Optional[float] = None
    sparkline: List[int] = []
    # Backward-compatible aliases used by the Streamlit UI
    last_7: Optional[float] = None
    last_30: Optional[float] = None
    last_60: Optional[float] = None
    correct_series: List[bool] = []


class FeedbackRequest(BaseModel):
    timestamp: Optional[str] = None
    time_step: Optional[str] = None
    rating: int
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
