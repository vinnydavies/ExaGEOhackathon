# ExaGEOhackathon

Materials for the ExaGEO Hackathon.

## Repository Structure

- `python_code/` — Python scripts and modules.
- `python_notebooks/` — Jupyter notebooks.
- `julia_code/` — Julia scripts and modules.
- `julia_notebooks/` — Julia Jupyter notebooks.

## Data

The examples download weather observations from NOAA NCEI's
[Global Hourly dataset](https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly).

- **Format:** JSON returned by NOAA's public API.
- **Access:** No download is stored in this repository; notebooks retrieve data for a chosen station and date range when run.
