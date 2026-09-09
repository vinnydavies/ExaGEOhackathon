"""Backward-compatible re-exports; use the ``dataprocessing`` and ``plotting`` packages directly for new code."""

from __future__ import annotations

from dataprocessing import (
    NOAA_GLOBAL_HOURLY_URL,
    clean_global_hourly,
    fetch_global_hourly,
    make_supervised_sequences,
)
from plotting import plot_weather

__all__ = [
    "NOAA_GLOBAL_HOURLY_URL",
    "clean_global_hourly",
    "fetch_global_hourly",
    "make_supervised_sequences",
    "plot_weather",
]
