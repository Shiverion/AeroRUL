"""Temporal Convolutional Network RUL regressor: stacked dilated causal convolutions
(Bai et al. 2018) give an exponentially growing receptive field without the sequential
bottleneck of an RNN — each layer only looks backward in time (causal), never at future
cycles, which matters for a model meant to run on live, still-accumulating sensor data.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from aerorul.models.sequence_common import (
    TrainConfig,
    fit_model,
    load_sequence_model,
    make_dataloaders,
    predict_sequences,
    save_sequence_model,
)


class _Chomp1d(nn.Module):
    """Trim the extra right-padding causal conv needs, so output length matches input."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: tuple[int, ...] = (32, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, seq_len, features) -> (batch, features, seq_len)
        out = self.tcn(x)
        last_step = out[:, :, -1]  # most recent cycle's representation
        return self.head(last_step)


def train_tcn(train_df, feature_cols: list[str], config: TrainConfig | None = None):
    config = config or TrainConfig()
    train_loader, val_loader = make_dataloaders(train_df, feature_cols, config)
    model = TCNRegressor(n_features=len(feature_cols))
    history = fit_model(model, train_loader, val_loader, config)
    return model, history


def predict_tcn(model: TCNRegressor, X, rul_cap: float):
    return predict_sequences(model, X, rul_cap)


def save_tcn(model: TCNRegressor, config: TrainConfig, path: str) -> None:
    save_sequence_model(model, config, path)


def load_tcn(path: str, n_features: int) -> tuple[TCNRegressor, TrainConfig]:
    model = TCNRegressor(n_features=n_features)
    config = load_sequence_model(model, path)
    return model, config
