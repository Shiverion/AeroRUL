"""LSTM RUL regressor: a stacked LSTM reads the sensor-window sequence and the final
hidden state is projected to a single RUL estimate.
"""

from __future__ import annotations

import torch
from torch import nn

from aerorul.models.sequence_common import (
    TrainConfig,
    fit_model,
    load_sequence_model,
    make_dataloaders,
    predict_sequences,
    save_sequence_model,
)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # final layer's hidden state
        return self.head(last_hidden)


def train_lstm(train_df, feature_cols: list[str], config: TrainConfig | None = None):
    config = config or TrainConfig()
    train_loader, val_loader = make_dataloaders(train_df, feature_cols, config)
    model = LSTMRegressor(n_features=len(feature_cols))
    history = fit_model(model, train_loader, val_loader, config)
    return model, history


def predict_lstm(model: LSTMRegressor, X, rul_cap: float):
    return predict_sequences(model, X, rul_cap)


def save_lstm(model: LSTMRegressor, config: TrainConfig, path: str) -> None:
    save_sequence_model(model, config, path)


def load_lstm(path: str, n_features: int) -> tuple[LSTMRegressor, TrainConfig]:
    model = LSTMRegressor(n_features=n_features)
    config = load_sequence_model(model, path)
    return model, config
