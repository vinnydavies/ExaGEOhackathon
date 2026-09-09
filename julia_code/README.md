# julia_code

This folder contains Julia scripts and modules for the project.

- `noaa_global_hourly.jl` — fetching and cleaning NOAA Global Hourly data, building supervised sequences, and plotting (weather, training history, predictions).
- `models.jl` — Flux model definitions (an LSTM forecaster and a classic feed-forward/MLP forecaster) plus a shared, model-agnostic training loop.
- `evaluation.jl` — regression metrics for evaluating model forecasts.

The NOAA example requires the Julia packages `HTTP`, `JSON3`, `DataFrames`, `Plots`, and `Flux`.