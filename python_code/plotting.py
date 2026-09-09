"""Plotting helpers for NOAA weather observations and model predictions."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_weather(
    cleaned: pd.DataFrame, *, station_name: str = "NOAA station"
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot cleaned temperature and wind speed observations."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(cleaned["DATE"], cleaned["temperature_c"], color="#d95f02", marker=".", linewidth=1)
    axes[0].set_ylabel("Temperature (C)")
    axes[0].set_title(f"{station_name}: NOAA Global Hourly")
    axes[0].grid(alpha=0.25)

    axes[1].plot(cleaned["DATE"], cleaned["wind_speed_mps"], color="#1b9e77", marker=".", linewidth=1)
    axes[1].set_ylabel("Wind speed (m/s)")
    axes[1].set_xlabel("Observation time (UTC)")
    axes[1].grid(alpha=0.25)
    return figure, (axes[0], axes[1])


def plot_predictions(
    actual: np.ndarray | Sequence[float],
    predicted: np.ndarray | Sequence[float],
    *,
    dates: Sequence | None = None,
    title: str = "Model predictions vs actual",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot actual vs. predicted values for a forecast, e.g. an LSTM output."""
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    x = dates if dates is not None else np.arange(len(actual))

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(x, actual, label="Actual", color="#377eb8", marker=".", linewidth=1)
    axis.plot(x, predicted, label="Predicted", color="#e41a1c", marker=".", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Time" if dates is None else "Observation time (UTC)")
    axis.set_ylabel("Value")
    axis.legend()
    axis.grid(alpha=0.25)
    return figure, axis


def plot_training_history(history, *, title: str = "Training history") -> tuple[plt.Figure, plt.Axes]:
    """Plot training/validation loss curves from a ``{"loss": [...], "val_loss": [...]}`` history dict."""
    log = history

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(log["loss"], label="Training loss", color="#377eb8")
    if "val_loss" in log:
        axis.plot(log["val_loss"], label="Validation loss", color="#e41a1c")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    return figure, axis
