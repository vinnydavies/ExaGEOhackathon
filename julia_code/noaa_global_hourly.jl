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
    return DataFrame([Dict(Symbol(key) => value for (key, value) in pairs(record)) for record in payload])
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
    cleaned[!, :wind_direction_deg] = numeric_or_missing.(wind_component(1))
    cleaned[cleaned.wind_direction_deg .== 999, :wind_direction_deg] .= missing
    cleaned[!, :wind_type] = wind_component(2)
    cleaned[!, :wind_speed_mps] = numeric_or_missing.(wind_component(4)) ./ 10
    cleaned[cleaned.wind_speed_mps .>= 999, :wind_speed_mps] .= missing

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