"""Regression metrics for evaluating forecasting models."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_absolute_error(y_true: np.ndarray | Sequence[float], y_pred: np.ndarray | Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray | Sequence[float], y_pred: np.ndarray | Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray | Sequence[float], y_pred: np.ndarray | Sequence[float]
) -> float:
    """MAPE as a percentage; entries where ``y_true`` is zero are ignored."""
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r_squared(y_true: np.ndarray | Sequence[float], y_pred: np.ndarray | Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum == 0:
        return float("nan")
    return float(1 - residual_sum / total_sum)


def evaluate_forecast(
    y_true: np.ndarray | Sequence[float], y_pred: np.ndarray | Sequence[float]
) -> dict[str, float]:
    """Compute a standard set of regression metrics for a forecast."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }
