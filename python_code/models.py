"""Forecasting models for a univariate time series (e.g. temperature).

Both models consume the same windowed input shape produced by
``make_supervised_sequences``: ``(samples, lookback, 1)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class LSTMForecaster(nn.Module):
    """Single-layer LSTM forecaster predicting Gaussian parameters."""

    def __init__(self, *, units: int = 32, horizon: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True)
        self.head = nn.Linear(units, 2 * horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class DenseForecaster(nn.Module):
    """Classic feed-forward forecaster predicting Gaussian parameters."""

    def __init__(self, *, lookback: int, units: int = 32, horizon: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 2 * horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gaussian_parameters(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split model output into a mean and a strictly positive standard deviation."""
    mean, raw_sigma = prediction.chunk(2, dim=-1)
    sigma = F.softplus(raw_sigma) + 1e-6
    return mean, sigma


def gaussian_nll(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean Gaussian negative log likelihood for a batch."""
    mean, sigma = gaussian_parameters(prediction)
    return (
        0.5 * ((target - mean) / sigma).square()
        + sigma.log()
        + 0.5 * np.log(2 * np.pi)
    ).mean()


def predict_gaussian(model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict means and standard deviations for NumPy input windows."""
    model.eval()
    inputs = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        mean, sigma = gaussian_parameters(model(inputs))
    return mean.numpy(), sigma.numpy()


def train_forecaster(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    validation_split: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 5,
) -> dict[str, list[float]]:
    """Train any forecaster (LSTM, Dense, etc.) with early stopping on validation loss.

    Returns a history dict with ``loss`` and ``val_loss`` lists, one entry per epoch.
    """
    dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = gaussian_nll(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += gaussian_nll(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= max(n_val, 1)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# Backwards-compatible alias: training is model-agnostic (LSTM, Dense, etc.).
train_lstm_model = train_forecaster

