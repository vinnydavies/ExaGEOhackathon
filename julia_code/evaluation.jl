"""Regression metrics for evaluating forecasting models."""

using Statistics: mean

function mean_absolute_error(y_true, y_pred)
    y_true, y_pred = Float64.(y_true), Float64.(y_pred)
    return mean(abs.(y_true .- y_pred))
end

function root_mean_squared_error(y_true, y_pred)
    y_true, y_pred = Float64.(y_true), Float64.(y_pred)
    return sqrt(mean((y_true .- y_pred) .^ 2))
end

"""MAPE as a percentage; entries where `y_true` is zero are ignored."""
function mean_absolute_percentage_error(y_true, y_pred)
    y_true, y_pred = Float64.(y_true), Float64.(y_pred)
    mask = y_true .!= 0
    !any(mask) && return NaN
    return mean(abs.((y_true[mask] .- y_pred[mask]) ./ y_true[mask])) * 100
end

function r_squared(y_true, y_pred)
    y_true, y_pred = Float64.(y_true), Float64.(y_pred)
    residual_sum = sum((y_true .- y_pred) .^ 2)
    total_sum = sum((y_true .- mean(y_true)) .^ 2)
    total_sum == 0 && return NaN
    return 1 - residual_sum / total_sum
end

"""Compute a standard set of regression metrics for a forecast."""
function evaluate_forecast(y_true, y_pred)
    return Dict(
        "mae" => mean_absolute_error(y_true, y_pred),
        "rmse" => root_mean_squared_error(y_true, y_pred),
        "mape" => mean_absolute_percentage_error(y_true, y_pred),
        "r2" => r_squared(y_true, y_pred),
    )
end
