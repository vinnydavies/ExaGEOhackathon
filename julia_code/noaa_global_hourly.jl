"""Helpers for exploring NOAA Global Hourly observations."""

using Dates
using DataFrames
using HTTP
using JSON3
using Plots

const NOAA_GLOBAL_HOURLY_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

"""Fetch Global Hourly records for one station and date range."""
function fetch_global_hourly(
    station::AbstractString,
    start_date::AbstractString,
    end_date::AbstractString;
    endpoint::AbstractString = NOAA_GLOBAL_HOURLY_URL,
)
    parameters = [
        "dataset" => "global-hourly",
        "stations" => station,
        "startDate" => start_date,
        "endDate" => end_date,
        "format" => "json",
        "units" => "metric",
        "includeAttributes" => "false",
    ]
    query = join(["$(HTTP.escapeuri(key))=$(HTTP.escapeuri(value))" for (key, value) in parameters], "&")
    response = HTTP.get("$(endpoint)?$(query)", ["User-Agent" => "ExaGEOhackathon/1.0"]; readtimeout = 60)
    payload = JSON3.read(String(response.body))

    isempty(payload) && return DataFrame()

    # records don't all share the same fields, so pad rows with `missing` for absent keys
    all_keys = reduce(union, (Symbol.(keys(record)) for record in payload))
    rows = [Dict(key => get(record, key, missing) for key in all_keys) for record in payload]
    return DataFrame(rows)
end

function packed_number(value, scale::Float64 = 1.0)
    ismissing(value) && return missing
    number = tryparse(Float64, first(split(string(value), ",")))
    (isnothing(number) || abs(number) >= 9990) && return missing
    return number * scale
end

function column_or_missing(records::DataFrame, name::Symbol)
    return name in propertynames(records) ? records[!, name] : fill(missing, nrow(records))
end

function parse_observation_time(value)
    ismissing(value) && return missing
    timestamp = replace(string(value), r"\..*$" => "", r"Z$" => "")
    try
        return DateTime(first(timestamp, min(lastindex(timestamp), 19)), dateformat"yyyy-mm-ddTHH:MM:SS")
    catch
        return missing
    end
end

function numeric_or_missing(value)
    ismissing(value) && return missing
    number = tryparse(Float64, string(value))
    return isnothing(number) ? missing : number
end

"""Clean common Global Hourly fields into analysis-ready columns."""
function clean_global_hourly(records::DataFrame)
    isempty(records) && return copy(records)

    cleaned = copy(records)
    rename!(cleaned, Symbol.(uppercase.(string.(names(cleaned)))))
    cleaned[!, :DATE] = parse_observation_time.(column_or_missing(cleaned, :DATE))
    cleaned[!, :temperature_c] = packed_number.(column_or_missing(cleaned, :TMP), 0.1)
    cleaned[!, :dew_point_c] = packed_number.(column_or_missing(cleaned, :DEW), 0.1)
    cleaned[!, :visibility_km] = packed_number.(column_or_missing(cleaned, :VIS), 0.1)

    wind_parts = split.(string.(coalesce.(column_or_missing(cleaned, :WND), "")), ",")
    wind_component(position) = [length(parts) >= position ? parts[position] : missing for parts in wind_parts]
    wind_direction = numeric_or_missing.(wind_component(1))
    cleaned[!, :wind_direction_deg] = [ismissing(v) || v == 999 ? missing : v for v in wind_direction]
    cleaned[!, :wind_type] = wind_component(2)
    wind_speed = numeric_or_missing.(wind_component(4))
    cleaned[!, :wind_speed_mps] = [ismissing(v) || v / 10 >= 999 ? missing : v / 10 for v in wind_speed]

    dropmissing!(cleaned, :DATE)
    cleaned = unique(cleaned, :DATE)
    sort!(cleaned, :DATE)
    return cleaned
end

"""Plot cleaned temperature and wind speed observations."""
function plot_weather(cleaned::DataFrame; station_name::AbstractString = "NOAA station")
    temperature = plot(
        cleaned.DATE,
        cleaned.temperature_c;
        color = "#d95f02",
        marker = :circle,
        markersize = 3,
        linewidth = 1,
        label = false,
        ylabel = "Temperature (C)",
        title = "$(station_name): NOAA Global Hourly",
        gridalpha = 0.25,
    )
    wind_speed = plot(
        cleaned.DATE,
        cleaned.wind_speed_mps;
        color = "#1b9e77",
        marker = :circle,
        markersize = 3,
        linewidth = 1,
        label = false,
        ylabel = "Wind speed (m/s)",
        xlabel = "Observation time (UTC)",
        gridalpha = 0.25,
    )
    return plot(temperature, wind_speed; layout = (2, 1), size = (900, 520))
end

"""Turn a 1D series into sliding-window `(X, y)` pairs for sequence models.

`X` has shape `(1, lookback, samples)` and `y` has shape `(horizon, samples)`,
matching the `(features, timesteps, batch)` convention used by `models.jl`.
"""
function make_supervised_sequences(values::AbstractVector; lookback::Int, horizon::Int = 1)
    series = Float32.(values)
    n_samples = length(series) - lookback - horizon + 1
    n_samples <= 0 && error("Not enough observations for the requested lookback/horizon.")

    X = Array{Float32}(undef, 1, lookback, n_samples)
    y = Array{Float32}(undef, horizon, n_samples)
    for i in 1:n_samples
        X[1, :, i] = series[i:(i + lookback - 1)]
        y[:, i] = series[(i + lookback):(i + lookback + horizon - 1)]
    end
    return X, y
end

"""Plot actual vs. predicted values for a forecast, e.g. an LSTM or Dense model output."""
function plot_predictions(actual, predicted; dates = nothing, title::AbstractString = "Model predictions vs actual")
    x = dates === nothing ? (1:length(actual)) : dates
    p = plot(
        x,
        actual;
        label = "Actual",
        color = "#377eb8",
        marker = :circle,
        markersize = 2,
        linewidth = 1,
        title = title,
        xlabel = dates === nothing ? "Time" : "Observation time (UTC)",
        ylabel = "Value",
        gridalpha = 0.25,
    )
    plot!(p, x, predicted; label = "Predicted", color = "#e41a1c", marker = :circle, markersize = 2, linewidth = 1)
    return p
end

"""Plot training/validation loss curves from a `Dict("loss" => ..., "val_loss" => ...)` history."""
function plot_training_history(history; title::AbstractString = "Training history")
    p = plot(
        history["loss"];
        label = "Training loss",
        color = "#377eb8",
        xlabel = "Epoch",
        ylabel = "Loss",
        title = title,
        gridalpha = 0.25,
    )
    haskey(history, "val_loss") && plot!(p, history["val_loss"]; label = "Validation loss", color = "#e41a1c")
    return p
end