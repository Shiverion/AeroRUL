"""Shared PyTorch plumbing for the sequence models (LSTM / TCN / Transformer): dataset
construction with unit-level train/val splitting, target scaling, and a generic training
loop with early stopping. Each model module (lstm.py, tcn.py, transformer.py) only needs to
define its `nn.Module` architecture and reuses everything here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from aerorul.features.engineering import build_sequences

DEFAULT_SEQ_LEN = 30


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_last_sequence_per_unit(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    label_col: str = "RUL",
    unit_col: str = "unit_number",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sequence per unit ending at its most recent cycle — the CMAPSS test protocol."""
    truncated = (
        df.sort_values("time_cycles")
        .groupby(unit_col, group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.tail(seq_len))
    )
    return build_sequences(truncated, feature_cols, seq_len, label_col, unit_col)


def train_val_split_by_unit(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 0, unit_col: str = "unit_number"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole units between train/val so overlapping sliding windows from the same
    engine never leak across the split (a per-row random split would let near-identical
    windows appear on both sides).
    """
    units = df[unit_col].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    val_df = df[df[unit_col].isin(val_units)]
    train_df = df[~df[unit_col].isin(val_units)]
    return train_df, val_df


@dataclass
class TrainConfig:
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    patience: int = 4
    val_fraction: float = 0.2
    rul_cap: float = 125.0  # used to scale targets to ~[0, 1] for stable training
    seed: int = 0


def make_dataloaders(
    train_df: pd.DataFrame, feature_cols: list[str], config: TrainConfig
) -> tuple[DataLoader, DataLoader]:
    fit_df, val_df = train_val_split_by_unit(train_df, config.val_fraction, config.seed)

    X_train, y_train, _ = build_sequences(fit_df, feature_cols, config.seq_len)
    X_val, y_val, _ = build_last_sequence_per_unit(val_df, feature_cols, config.seq_len)

    train_ds = SequenceDataset(X_train, y_train / config.rul_cap)
    val_ds = SequenceDataset(X_val, y_val / config.rul_cap)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def fit_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: TrainConfig
) -> dict[str, list[float]]:
    """Generic training loop with early stopping on validation MSE. Any of our sequence
    architectures plugs in here — they only differ in their forward() implementation.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze(-1)
                val_losses.append(loss_fn(preds, y_batch).item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_sequences(model: nn.Module, X: np.ndarray, rul_cap: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float()).squeeze(-1).numpy()
    return np.clip(preds * rul_cap, a_min=0, a_max=None)


def save_sequence_model(model: nn.Module, config: TrainConfig, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_sequence_model(model: nn.Module, path: str) -> TrainConfig:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint["config"]
