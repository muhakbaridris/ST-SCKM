"""Access to bundled example data."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

EARTHQUAKE_DATA_SOURCE = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&"
    "starttime=2020-01-01&endtime=2021-01-01&minmagnitude=3&"
    "minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114&"
    "orderby=time-asc"
)
EARTHQUAKE_DATA_SHA256 = "d6cb2abfecde6ebc6a63b753c184ea17f0d4d2de5ab87726bea1002d86a8735a"


def load_sample_wildfire() -> pd.DataFrame:
    """Load the bundled 1,200-row synthetic wildfire dataset."""
    resource = files("stsckm").joinpath("data/sample_data.csv")
    with resource.open("r", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame


def load_sample_earthquakes() -> pd.DataFrame:
    """Load an archived 2020 USGS California earthquake catalog subset.

    The 924-event snapshot contains earthquakes of magnitude 3 or greater in
    the bounding box 32--42 degrees north and 125--114 degrees west. It was
    retrieved from the USGS FDSN Event Web Service on 2026-08-29. The package
    uses the archived file so examples do not depend on a live network query.
    """
    resource = files("stsckm").joinpath("data/usgs_california_2020_m3.csv")
    with resource.open("r", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["updated"] = pd.to_datetime(frame["updated"], utc=True)
    return frame
