# python_code

This folder contains Python scripts and modules for the project.

- `dataprocessing.py` — fetching and cleaning NOAA Global Hourly data, and building supervised sequences for models.
- `models.py` — PyTorch LSTM and feed-forward/MLP forecasters that predict Gaussian means and standard deviations, plus a shared NLL training loop.
- `plotting.py` — plotting of raw/cleaned weather observations, model predictions, and optional 95% prediction intervals.
- `evaluation.py` — regression metrics for evaluating model forecasts.
- `noaa_global_hourly.py` — backward-compatible re-exports of the `dataprocessing`/`plotting` functions.
