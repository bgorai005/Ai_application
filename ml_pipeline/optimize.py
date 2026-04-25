import os
import torch
import joblib
import configparser
import logging


def quantize_model(model_path: str, output_path: str, input_size: int, hidden_size: int, num_layers: int):
    from ml_pipeline.model_architecture import LSTMModel

    device = torch.device("cpu")
    # Load float model
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Apply dynamic quantization (supports LSTM/Linear)
    try:
        qmodel = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
        )
        # Save quantized model state_dict
        torch.save(qmodel.state_dict(), output_path)
        logging.info(f"Saved quantized model to {output_path}")
    except Exception as e:
        logging.error(f"Quantization failed: {e}")


if __name__ == "__main__":
    # Read config
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"))

    model_path = cfg.get("model", "model_path", fallback="./models/best_model.pth")
    quantized_path = cfg.get("model", "quantized_model_path", fallback="./models/quantized_model.pth")
    input_size = cfg.getint("MODEL", "input_size", fallback=1)
    hidden_size = cfg.getint("MODEL", "hidden_size", fallback=64)
    num_layers = cfg.getint("MODEL", "num_layers", fallback=2)

    os.makedirs(os.path.dirname(quantized_path) or "./models", exist_ok=True)
    quantize_model(model_path, quantized_path, input_size, hidden_size, num_layers)
