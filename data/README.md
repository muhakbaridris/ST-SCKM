# Data

This directory contains a small synthetic sample dataset for running the
examples without downloading the full California wildfire dataset.

## Files

- `sample_data.csv`: California-like synthetic wildfire observations with
  latitude, longitude, datetime, FRP, temperature, humidity, and wind speed.

## Full Dataset

The accepted manuscript uses MODIS FIRMS active fire detections and ERA5
meteorological variables for California from 2019 to 2024. The full raw dataset
is not committed to this repository because of size and provenance constraints.
Users can adapt `examples/run_example.py` to their own FIRMS/ERA5 CSV file.
