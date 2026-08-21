NOAA_GLOBAL_HOURLY_URL <- "https://www.ncei.noaa.gov/access/services/data/v1"

fetch_global_hourly <- function(station, start_date, end_date,
                                endpoint = NOAA_GLOBAL_HOURLY_URL) {
  parameters <- c(
    dataset = "global-hourly",
    stations = station,
    startDate = start_date,
    endDate = end_date,
    format = "json",
    units = "metric",
    includeAttributes = "false"
  )
  query <- paste(
    paste(names(parameters), utils::URLencode(parameters, reserved = TRUE), sep = "="),
    collapse = "&"
  )
  response <- httr2::request(paste0(endpoint, "?", query)) |>
    httr2::req_headers(`User-Agent` = "ExaGEOhackathon/1.0") |>
    httr2::req_perform()
  payload <- jsonlite::fromJSON(
    rawToChar(httr2::resp_body_raw(response)),
    simplifyDataFrame = TRUE
  )

  if (is.null(payload) || length(payload) == 0) {
    return(data.frame())
  }
  as.data.frame(payload, stringsAsFactors = FALSE)
}

packed_number <- function(value, scale = 1) {
  number <- suppressWarnings(as.numeric(sub(",.*", "", as.character(value))))
  number[abs(number) >= 9990] <- NA_real_
  number * scale
}

field_or_na <- function(records, field) {
  if (field %in% names(records)) records[[field]] else rep(NA_character_, nrow(records))
}

clean_global_hourly <- function(records) {
  if (nrow(records) == 0) {
    return(records)
  }

  cleaned <- records
  names(cleaned) <- toupper(names(cleaned))
  date_text <- sub("Z$", "", sub("\\..*$", "", as.character(field_or_na(cleaned, "DATE"))))
  cleaned$DATE <- suppressWarnings(as.POSIXct(
    date_text,
    format = "%Y-%m-%dT%H:%M:%S",
    tz = "UTC"
  ))
  cleaned$temperature_c <- packed_number(field_or_na(cleaned, "TMP"), 0.1)
  cleaned$dew_point_c <- packed_number(field_or_na(cleaned, "DEW"), 0.1)
  cleaned$visibility_km <- packed_number(field_or_na(cleaned, "VIS"), 0.1)

  wind_parts <- strsplit(
    ifelse(is.na(field_or_na(cleaned, "WND")), "", as.character(field_or_na(cleaned, "WND"))),
    ",",
    fixed = TRUE
  )
  wind_component <- function(position) {
    vapply(
      wind_parts,
      function(parts) if (length(parts) >= position) parts[[position]] else NA_character_,
      character(1)
    )
  }
  cleaned$wind_direction_deg <- suppressWarnings(as.numeric(wind_component(1)))
  cleaned$wind_direction_deg[which(cleaned$wind_direction_deg == 999)] <- NA_real_
  cleaned$wind_type <- wind_component(2)
  cleaned$wind_speed_mps <- suppressWarnings(as.numeric(wind_component(4))) / 10
  cleaned$wind_speed_mps[which(cleaned$wind_speed_mps >= 999)] <- NA_real_

  cleaned <- cleaned[!is.na(cleaned$DATE), , drop = FALSE]
  cleaned <- cleaned[!duplicated(cleaned$DATE), , drop = FALSE]
  cleaned[order(cleaned$DATE), , drop = FALSE]
}

plot_weather <- function(cleaned, station_name = "NOAA station") {
  temperature <- ggplot2::ggplot(cleaned, ggplot2::aes(DATE, temperature_c)) +
    ggplot2::geom_line(colour = "#d95f02") +
    ggplot2::geom_point(colour = "#d95f02") +
    ggplot2::labs(
      title = paste0(station_name, ": NOAA Global Hourly"),
      y = "Temperature (C)",
      x = NULL
    ) +
    ggplot2::theme_minimal()
  wind_speed <- ggplot2::ggplot(cleaned, ggplot2::aes(DATE, wind_speed_mps)) +
    ggplot2::geom_line(colour = "#1b9e77") +
    ggplot2::geom_point(colour = "#1b9e77") +
    ggplot2::labs(y = "Wind speed (m/s)", x = "Observation time (UTC)") +
    ggplot2::theme_minimal()
  list(temperature = temperature, wind_speed = wind_speed)
}