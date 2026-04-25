# app/model.py
import torch
import numpy as np
import configparser
import os
import joblib
#from ml_pipeline.train import scaler  # Uncomment if using a scaler

from ml_pipeline.model_architecture import LSTMModel

# --- Load config ---
config = configparser.ConfigParser()
config_filepath = os.path.join('./ml_pipeline', 'config.ini')
config.read(config_filepath)

# --- Model parameters ---
INPUT_SIZE = config.getint('MODEL', 'input_size')
HIDDEN_SIZE = config.getint('MODEL', 'hidden_size')
NUM_LAYERS = config.getint('MODEL', 'num_layers')

# --- Config ---
SCALER_PATH = config.get('scaler', 'scaler_path', fallback="./models/scaler.pkl")
MODEL_PATH = config.get('model', 'model_path', fallback="./models/best_model.pth")
QUANTIZED_MODEL_PATH = config.get('model', 'quantized_model_path', fallback="./models/quantized_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model once at server start ---
model = None
scaler = None


def load_model_and_scaler():
    global model, scaler
    # Load the scaler (if present) and model (if present). Fail gracefully if missing.
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            scaler = None

        # Prefer a quantized model if available (faster inference on edge)
        selected_model_path = (
            QUANTIZED_MODEL_PATH
            if os.path.exists(QUANTIZED_MODEL_PATH)
            else MODEL_PATH
        )

        if os.path.exists(selected_model_path):
            model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
            model.load_state_dict(torch.load(selected_model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
        else:
            # Model file not available yet (e.g., first deploy). Leave model as None.
            model = None
    except Exception as e:
        # On any failure, ensure model/scaler are unset so callers can fallback.
        import logging

        logging.error(f"Failed to load model/scaler: {e}")
        model = None
        scaler = None


load_model_and_scaler()

# --- Prediction function ---
#///////////////////////////////////////////////////////////////////////////////////////////////////////////
# This function takes a sequence of stock prices and predicts the next price

# def predict_stock_price(input_sequence: np.ndarray) -> float:
#     model_input = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         prediction = model(model_input)

#     prediction = prediction.cpu().numpy().flatten()
#     prediction = np.clip(prediction, 0, None)  # Ensures non-negative predictions
#     prediction = np.round(prediction, 3)  # Round-off to 3 decimal places
#     if scaler is not None:
#         prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
#     else:
#         prediction = prediction.reshape(-1, 1)

#     return prediction[0]
#///////////////////////////////////////////////////////////////////////////////////////////////////////////


# This function takes a sequence of stock prices and predicts the next 'steps' prices
def predict_stock_price(input_sequence: np.ndarray, steps: int = 30) -> list:
    sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            output = model(sequence)
            prediction = output.cpu().numpy().flatten()[0]
            prediction = np.clip(prediction, 0, None)
            prediction = np.round(prediction, 3)

            # Inverse transform if scaler exists
            if scaler is not None:
                prediction_scaled = scaler.inverse_transform(np.array([[prediction]]))[0][0]
            else:
                prediction_scaled = prediction

            predictions.append(prediction_scaled)

            # Create next input — reuse previous features structure (simplified)
            next_input = torch.tensor([[prediction] * sequence.shape[-1]], dtype=torch.float32).to(DEVICE)
            sequence = torch.cat((sequence[:, 1:, :], next_input.unsqueeze(0)), dim=1)

    return predictions
