# API Specification

## Prediction API

### `POST /predict`

Request body:

```json
{
  "data": [[1.0], [1.1], [1.2]]
}
```

Response body:

```json
{
  "prediction": [1.23, 1.24, 1.25]
}
```

## Signal API

### `GET /signal/current`

Returns the most recent computed signal.

### `GET /signal/history?limit=30`

Returns the latest `limit` signal entries.

### `GET /signal/context?length=20`

Returns the last `length` rows of model input context.

### `GET /signal/accuracy`

Returns:

```json
{
  "accuracy_7": 70.0,
  "accuracy_30": 66.67,
  "accuracy_60": 61.11,
  "sparkline": [1, 0, 1]
}
```

## Model Health API

### `GET /model/info`

Returns readiness, last retrain time, data freshness, and drift status.

### `GET /model/metrics`

Returns the same accuracy metrics plus backward-compatible dashboard fields.

## Feedback API

### `POST /feedback`

Request body:

```json
{
  "timestamp": "2026-04-28T12:00:00Z",
  "time_step": "2026-04-28T11:55:00Z",
  "rating": 5,
  "comment": "Useful signal"
}
```

Response body:

```json
{
  "status": "ok"
}
```

## Data Contracts

- `PredictRequest.data` is a list of list of floats.
- `PredictResponse.prediction` is a list of floats.
- `MetricsResponse` includes `accuracy_7`, `accuracy_30`, `accuracy_60`, `sparkline`, and dashboard compatibility aliases.
