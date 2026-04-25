# ml_pipeline/model_architecture.py
import torch
import torch.nn as nn


# ///////////////////////////////////////////
# LSTM Model Definition
# This script defines the LSTM model architecture.
# ///////////////////////////////////////////


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM returns (output, (h_n, c_n)) for hidden state and cell state
        out, _ = self.lstm(x)
        # Still using only the last time step
        return self.fc(out[:, -1, :])
