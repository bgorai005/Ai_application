import logging
# ///////////////////////////////////////////
# Early Stopping
# This script implements early stopping to prevent overfitting.
# ///////////////////////////////////////////


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if (
            self.best_score is None
            or val_loss < self.best_score - self.min_delta
        ):
            self.best_score = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                logging.info(f"Monitored metric improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"Monitored metric did not improve. Counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.warning("Early stopping triggered.")


# ///////////////////////////////////////////
# TimeSeriesDataset
# Custom Torch dataloader class for the time series data.
# ///////////////////////////////////////////
from torch.utils.data import Dataset
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.x = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ///////////////////////////////////////////
# Data lag features creator function
# ///////////////////////////////////////////
import numpy as np


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# ///////////////////////////////////////////
# Training and Validation
# This script contains the training and validation loops.
# ///////////////////////////////////////////
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0

    for sequences, targets in train_loader:
        sequences, targets = (
            sequences.to(device),
            targets.to(device).unsqueeze(1),
        )

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
    logging.info(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")

    return avg_loss  # Returns average loss for the epoch


def mean_absolute_percentage_error(y_true, y_pred):
    return (
        np.mean(
            np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))
        )
        * 100
    )


def validate(model, val_loader, criterion, device, epoch, scaler):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences, targets = (
                sequences.to(device),
                targets.to(device).unsqueeze(1),
            )
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert to numpy arrays and inverse transform
    y_true_normalized = np.array(all_targets).reshape(-1, 1)
    y_pred_normalized = np.array(all_preds).reshape(-1, 1)

    # Inverse transform to original scale
    y_true = scaler.inverse_transform(y_true_normalized).flatten()
    y_pred = scaler.inverse_transform(y_pred_normalized).flatten()

    # Calculate metrics on original scale
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Logging
    logging.info(
        f"Epoch {epoch + 1} - Val MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}, MAPE: {mape:.6f}%"
    )
    mlflow.log_metric("val_MSE", mse, step=epoch + 1)
    mlflow.log_metric("val_RMSE", rmse, step=epoch + 1)
    mlflow.log_metric("val_MAE", mae, step=epoch + 1)
    mlflow.log_metric("val_R2", r2, step=epoch + 1)
    mlflow.log_metric("val_MAPE", mape, step=epoch + 1)

    return rmse, mse, mae, r2, mape  # Returns RMSE on original scale
