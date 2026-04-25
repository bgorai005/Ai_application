FROM python:3.10-slim AS streamlit

WORKDIR /app

# Use a small, focused requirements file for the Streamlit UI to avoid
# installing heavy training dependencies (xgboost, scikit-learn, etc.)
COPY streamlit_requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]


FROM python:3.10-slim AS mlflow

RUN pip install --no-cache-dir --prefer-binary mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlruns", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"]


FROM pytorch/pytorch:latest AS trainer

WORKDIR /app

# Install git so training runs can capture the commit SHA for reproducibility
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --prefer-binary \
    pandas \
    scikit-learn \
    matplotlib \
    joblib \
    mlflow-skinny \
    scipy \
    torchinfo \
    yfinance

COPY data_pipeline ./data_pipeline
COPY drift_detection ./drift_detection

CMD ["bash", "-lc", "cp ml_pipeline/config.ini ./config.ini && python data_pipeline/pipeline.py && python ml_pipeline/train.py"]
