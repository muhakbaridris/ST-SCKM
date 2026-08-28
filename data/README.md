# Data

The package contains two compact datasets for running examples without a live
network dependency. They represent different domains so the software interface
is not demonstrated only through wildfire zoning.

## Files

- `sample_data.csv`: California-like synthetic wildfire observations with
  latitude, longitude, datetime, FRP, temperature, humidity, and wind speed.
- `usgs_california_2020_m3.csv`: archived USGS FDSN Event Web Service response
  containing 924 California earthquakes of magnitude 3 or greater during
  2020. The snapshot was retrieved on 2026-08-29 from the query recorded below.

## Earthquake snapshot provenance

Source service: U.S. Geological Survey Earthquake Catalog API.

```text
https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2020-01-01&endtime=2021-01-01&minmagnitude=3&minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114&orderby=time-asc
```

SHA-256:

```text
d6cb2abfecde6ebc6a63b753c184ea17f0d4d2de5ab87726bea1002d86a8735a
```

The archived response is used for reproducibility because catalog records may
be revised after their initial publication. The example is a software
illustration and is not a seismic-hazard analysis.

## Full Dataset

The accepted manuscript uses MODIS FIRMS active fire detections and ERA5
meteorological variables for California from 2019 to 2024. The full raw dataset
is not committed to this repository because of size and provenance constraints.
Users can adapt `examples/run_example.py` to their own FIRMS/ERA5 CSV file.
