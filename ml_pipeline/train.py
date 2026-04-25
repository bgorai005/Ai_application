# ml_pipeline/train.py
# /////////////////////////////////////////
# Load control parameters from config.ini
# /////////////////////////////////////////

import configparser
import os
import joblib
import json
import subprocess

# Load configuration
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "config.ini"))

# /////////////////////////////////////////
# mlflow setup
# /////////////////////////////////////////
import mlflow

# MLFLOW parameters
mlflow_tracking_uri = config.get(
    "MLFLOW", "tracking_uri", fallback="http://127.0.0.1:5000"
)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_tracking_uri)
mlflow_experiment_name = config.get(
    "MLFLOW", "experiment_name", fallback="DefaultValue_(Change_name)"
)
mlflow.set_tracking_uri(mlflow_tracking_uri)

mlflow.set_experiment("Nify_50_LSTM")

# ///////////////////////////////////////////
# logging setup
# ///////////////////////////////////////////
import logging


# Get current working directory
log_file_path = os.path.join(os.getcwd(), "mlflow_run.log")

# Set up the logger manually
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove all previous handlers (important in Jupyter)
logger.handlers.clear()

# Create and add FileHandler
file_handler = logging.FileHandler(log_file_path, mode="w")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Optional: also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def log_pytorch_model_if_available(model, artifact_path: str) -> None:
    try:
        import mlflow.pytorch as mlflow_pytorch

        # Use the safer 'pt2' serialization format when available to avoid
        # pickling-based formats which can be unsafe on deserialization.
        try:
            mlflow_pytorch.log_model(model, artifact_path, serialization_format="pt2")
        except TypeError:
            # Fallback for older mlflow versions that don't support serialization_format
            mlflow_pytorch.log_model(model, artifact_path)
    except Exception as e:
        logging.warning(
            f"Skipped MLflow PyTorch model logging for {artifact_path}: {e}"
        )


# ///////////////////////////////////////////
# importing libraries
# ///////////////////////////////////////////
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from utils import EarlyStopping, TimeSeriesDataset, create_sequences
from utils import train, validate, mean_absolute_percentage_error
from model_architecture import LSTMModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# ///////////////////////////////////////////
# seed for reproducibility
# ///////////////////////////////////////////


def set_seed(seed=42):
    """
    Set the seed for reproducibility in PyTorch, NumPy, and Python's random module.
    Ensures deterministic behavior on CPU, CUDA, and MPS.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Works for CPU & MPS

    if torch.backends.mps.is_available():
        logging.info(
            "MPS backend is available. It uses the global PyTorch seed."
        )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    logging.info("Deterministic algorithms set to True.")
    logging.info("Random seed set to: %d", seed)


# Example usage
global_seed = int(config.get("DATA", "global_seed", fallback=42))
set_seed(global_seed)


# ///////////////////////////////////////////
# Data Preprocessing
# This script is responsible for loading the dataset, normalizing it, and splitting it into training and testing sets.
# ///////////////////////////////////////////


# Parameters
dataset_path = config.get(
    "DATA",
    "dataset_path",
    fallback="./data/processed/nifty50_5min_features.csv",
)
train_size = config.getfloat("DATA", "train_size", fallback=0.9)
test_size = config.getfloat("DATA", "test_size", fallback=0.1)
target_column = config.get("DATA", "target_column", fallback="Close")
assert train_size + test_size == 1, "Train and test sizes must sum to 1"

# Load dataset
data = pd.read_csv(dataset_path)
logger.info(f"Loaded data shape: {data.shape}")

# Initialize scaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Dataset splitting
split_index = int(len(data) * train_size)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

# Reset indices
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# --- Compute baseline statistics and samples (before normalization) ---
os.makedirs("drift_detection", exist_ok=True)
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
baseline_stats = {}
for col in numeric_cols:
    vals = train_data[col].values
    baseline_stats[col] = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "count": int(len(vals)),
    }
baseline_stats_path = os.path.join("drift_detection", "baseline_stats.json")
with open(baseline_stats_path, "w") as f:
    json.dump(baseline_stats, f, indent=4)

# Save baseline samples for later KS tests (compressed)
baseline_samples_path = os.path.join("drift_detection", "baseline_samples.npz")
np.savez_compressed(
    baseline_samples_path, **{col: train_data[col].values for col in numeric_cols}
)

# Normalize the "Close" column in both train and test sets
train_data[target_column] = scaler.fit_transform(train_data[[target_column]])
test_data[target_column] = scaler.transform(test_data[[target_column]])
# Save the scaler for future use
os.makedirs("models", exist_ok=True)
scaler_path = os.path.join("models", "scaler.pkl")
joblib.dump(scaler, scaler_path)

logger.info(f"Train data shape: {train_data.shape}")
logger.info(f"Test data shape: {test_data.shape}")


# ///////////////////////////////////////////
# Model training
# ///////////////////////////////////////////
# MODEL parameters
SEQ_LENGTH = config.getint("MODEL", "seq_length", fallback=30)
BATCH_SIZE = config.getint("MODEL", "batch_size", fallback=64)
HIDDEN_SIZE = config.getint("MODEL", "hidden_size", fallback=64)
NUM_LAYERS = config.getint("MODEL", "num_layers", fallback=2)
INPUT_SIZE = config.getint("MODEL", "input_size", fallback=1)

train_seq, train_target = create_sequences(
    train_data[target_column].values, SEQ_LENGTH
)
test_seq, test_target = create_sequences(
    test_data[target_column].values, SEQ_LENGTH
)

train_dataset = TimeSeriesDataset(train_seq, train_target)
test_dataset = TimeSeriesDataset(test_seq, test_target)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ///////////////////////////////////////////
# Training Loop
# This script contains the main training loop.
# ///////////////////////////////////////////


# Initialize model
model = LSTMModel(
    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
)


criterion_train = nn.MSELoss()
criterion_val = nn.MSELoss()


# Initialize optimizer
starting_lr = config.getfloat("TRAINING", "starting_lr", fallback=0.001)
optimizer = optim.Adam(model.parameters(), lr=starting_lr)

# SCHEDULER parameters
scheduler_mode = config.get("SCHEDULER", "mode", fallback="min")
scheduler_factor = config.getfloat("SCHEDULER", "factor", fallback=0.25)
scheduler_patience = config.getint("SCHEDULER", "patience", fallback=2)
scheduler_cooldown = config.getint("SCHEDULER", "cooldown", fallback=0)
scheduler_min_lr = config.getfloat("SCHEDULER", "min_lr", fallback=1e-6)
scheduler_threshold = config.getfloat(
    "SCHEDULER", "threshold", fallback=0.0001
)
scheduler_threshold_mode = config.get(
    "SCHEDULER", "threshold_mode", fallback="abs"
)
# Initialize learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode=scheduler_mode,
    factor=scheduler_factor,
    patience=scheduler_patience,
    cooldown=scheduler_cooldown,
    min_lr=scheduler_min_lr,
    threshold=scheduler_threshold,
    threshold_mode=scheduler_threshold_mode,
    # verbose=True
)

# Initialize early stopping
early_stopping_patience = scheduler_cooldown + scheduler_patience + 4
early_stopping_delta = scheduler_threshold / 2
early_stopping = EarlyStopping(
    patience=early_stopping_patience,
    verbose=True,
    min_delta=early_stopping_delta,
)  # noqa: E501

max_num_epochs = config.getint("TRAINING", "max_num_epochs", fallback=30)

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.info(f"Using device: {device.type}")
# Move model to device
model.to(device)


# Log params as a dictionary
params = {
    "global_seed": global_seed,
    "train_batch_size": train_size,
    "test_batch_size": test_size,
    "eval_batch_size": BATCH_SIZE,
    "loss_function": "MSELoss",
    "model_SEQ_LENGTH": SEQ_LENGTH,
    "model_HIDDEN_SIZE": HIDDEN_SIZE,
    "model_BATCH_SIZE": BATCH_SIZE,
    "model_NUM_LAYERS": NUM_LAYERS,
    "learning_rate": starting_lr,
    "optimizer": "Adam",
    "scheduler_mode": scheduler_mode,
    "scheduler_factor": scheduler_factor,
    "scheduler_patience": scheduler_patience,
    "scheduler_cooldown": scheduler_cooldown,
    "scheduler_min_lr": scheduler_min_lr,
    "scheduler_therhold": scheduler_threshold,
    "scheduler_threshold_mode": scheduler_threshold_mode,
    "scheduler": "ReduceLROnPlateau",
    "early_stopping_patience": early_stopping_patience,
    "early_stopping_delta": early_stopping_delta,
    "max_epochs": max_num_epochs,
    "device": device.type,
}


# ///////////////////////////////////////////
# MLflow Logging
# This script logs parameters, metrics, and model artifacts to MLflow.
# ///////////////////////////////////////////
if mlflow.active_run():
    mlflow.end_run()

# Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    logging.info(f"Started MLflow run with ID: {run_id}")

    # Log parameters
    try:
        mlflow.log_params(params)
        logging.info("Logged params to MLflow")
    except Exception as e:
        logging.error(f"Failed to log parameters: {e}")
    # Log git commit SHA for reproducibility
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
    except Exception:
        git_commit = "unknown"
    try:
        mlflow.log_param("git_commit", git_commit)
    except Exception as e:
        logging.warning(f"Failed to log git commit to MLflow: {e}")

    # Log important precomputed artifacts (scaler, baseline stats/samples)
    try:
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path)
        if os.path.exists(baseline_stats_path):
            mlflow.log_artifact(baseline_stats_path)
        if os.path.exists(baseline_samples_path):
            mlflow.log_artifact(baseline_samples_path)
    except Exception as e:
        logging.warning(f"Failed to log precomputed artifacts to MLflow: {e}")

    # Save model summary
    try:
        model_summary = summary(
            model,
            input_size=(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
            device=device.type,
        )
        with open("model_report.txt", "w") as f:
            f.write("### MODEL ARCHITECTURE ###\n")
            f.write(str(model))
            f.write("\n\n### MODEL SUMMARY ###\n")
            f.write(str(model_summary))
        mlflow.log_artifact("model_report.txt")
        os.remove("model_report.txt")
    except Exception as e:
        logging.error(f"Failed to log model report: {e}")

    logging.info("Starting training...")

    # Initialize metrics
    best_rmse = float("inf")
    best_save_epoch = 0
    best_model_updated = False

    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for epoch in range(max_num_epochs):
        try:
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Epoch {epoch + 1}, LR: {current_lr}")
            mlflow.log_metric("learning_rate", current_lr, step=epoch + 1)

            train_loss = train(
                model, train_loader, criterion_train, optimizer, device, epoch
            )
            rmse, mse, mae, r2, mape = validate(
                model, test_loader, criterion_val, device, epoch, scaler
            )

            scheduler.step(rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_metric = rmse
                best_save_epoch = epoch
                best_train_loss = train_loss
                best_mse = mse
                best_mae = mae
                best_r2 = r2
                best_mape = mape
                # Save best model manually

                torch.save(model.state_dict(), "models/best_model.pth")
                best_model_updated = True

            log_pytorch_model_if_available(model, "latest_model")

            early_stopping(rmse, model)
            if early_stopping.early_stop:
                mlflow.log_param("best_epoch", best_save_epoch)
                break

        except Exception as e:
            logging.error(f"Training failed at epoch {epoch}: {e}")
            break

    if best_model_updated:
        # Log the best model to MLflow using the safer format and attempt
        # to register it in the MLflow Model Registry.
        log_pytorch_model_if_available(model, "best_model")
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            model_uri = f"runs:/{run_id}/best_model"
            try:
                mv = client.create_model_version(name="nifty-lstm", source=model_uri, run_id=run_id)
                logging.info(f"Registered model version: {mv.version} for model 'nifty-lstm'")
            except Exception as e:
                # If create_model_version fails (e.g., model not existing), try to create the registered model first
                try:
                    client.create_registered_model(name="nifty-lstm")
                    mv = client.create_model_version(name="nifty-lstm", source=model_uri, run_id=run_id)
                    logging.info(f"Registered model version: {mv.version} for model 'nifty-lstm'")
                except Exception as e2:
                    logging.warning(f"Failed to register model in MLflow registry: {e2}")
        except Exception as e:
            logging.warning(f"MLflow registry step skipped or failed: {e}")

    # Log final metrics
    try:
        mlflow.log_param("best_rmse", best_rmse)
        mlflow.log_param("best_epoch", best_save_epoch)
        mlflow.log_param("best_train_loss", best_train_loss)
        mlflow.log_param("best_mse", best_mse)
        mlflow.log_param("best_mae", best_mae)
        mlflow.log_param("best_r2", best_r2)
        mlflow.log_param("best_mape", best_mape)
    except Exception as e:
        logging.error(f"Failed to log best metrics: {e}")

    # Optional: run post-training optimizations (e.g., dynamic quantization)
    try:
        quantize_flag = config.getboolean("TRAINING", "quantize_after_training", fallback=False)
    except Exception:
        quantize_flag = False

    if quantize_flag:
        try:
            logging.info("Quantization flag set — running ml_pipeline/optimize.py")
            import subprocess
            subprocess.run(["python", "ml_pipeline/optimize.py"], check=True)
            quantized_path = os.path.join("models", "quantized_model.pth")
            if os.path.exists(quantized_path):
                try:
                    mlflow.log_artifact(quantized_path)
                except Exception as e:
                    logging.warning(f"Failed to log quantized model to MLflow: {e}")
        except Exception as e:
            logging.warning(f"Quantization step failed: {e}")

    # Handle log file
    try:
        if os.path.exists("mlflow_run.log"):
            mlflow.log_artifact("mlflow_run.log")
            os.remove("mlflow_run.log")
    except Exception as e:
        logging.error(f"Failed to handle log file: {e}")
    finally:
        logger.handlers.clear()


# ///////////////////////////////////////////
# Model Evaluation
# This script evaluates the best model on the test set and logs the results.
# ///////////////////////////////////////////

with mlflow.start_run(run_id=run_id) as run:
    # After loading your best model
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Inverse transform predictions and actual values
    y_pred_normalized = np.array(all_preds).reshape(-1, 1)
    y_true_normalized = np.array(all_targets).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_normalized).flatten()
    y_true = scaler.inverse_transform(y_true_normalized).flatten()

    # Plot with original scale values
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual Values", marker="o", linewidth=2)
    plt.plot(y_pred, label="Predicted Values", marker="x", linewidth=2)
    plt.title("Predicted vs Actual Stock Prices")
    plt.xlabel("Index")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("predictions_plot.png")
    mlflow.log_artifact("predictions_plot.png")
    os.remove("predictions_plot.png")
