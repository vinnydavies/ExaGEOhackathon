# python_code

This folder contains Python scripts and modules for the project.

- `dataprocessing.py` — fetching and cleaning NOAA Global Hourly data, and building supervised sequences for models.
- `models.py` — PyTorch model definitions (e.g. an LSTM forecaster).
- `plotting.py` — plotting of raw/cleaned weather observations and model predictions.
- `evaluation.py` — regression metrics for evaluating model forecasts.
- `noaa_global_hourly.py` — backward-compatible re-exports of the `dataprocessing`/`plotting` functions.
