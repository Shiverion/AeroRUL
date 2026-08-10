"""Transformer-encoder RUL regressor: self-attention over the sensor-window sequence lets
the model weigh any past cycle directly (no recurrence, no fixed receptive field), with a
learned positional encoding so it still knows *when* in the window each cycle occurred.
"""

from __future__ import annotations

import math

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


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.input_proj(x))
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # mean-pool over the window
        return self.head(pooled)


def train_transformer(train_df, feature_cols: list[str], config: TrainConfig | None = None):
    config = config or TrainConfig()
    train_loader, val_loader = make_dataloaders(train_df, feature_cols, config)
    model = TransformerRegressor(n_features=len(feature_cols))
    history = fit_model(model, train_loader, val_loader, config)
    return model, history


def predict_transformer(model: TransformerRegressor, X, rul_cap: float):
    return predict_sequences(model, X, rul_cap)


def save_transformer(model: TransformerRegressor, config: TrainConfig, path: str) -> None:
    save_sequence_model(model, config, path)


def load_transformer(path: str, n_features: int) -> tuple[TransformerRegressor, TrainConfig]:
    model = TransformerRegressor(n_features=n_features)
    config = load_sequence_model(model, path)
    return model, config
