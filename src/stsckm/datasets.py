"""Access to bundled example data."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def load_sample_wildfire() -> pd.DataFrame:
    """Load the bundled 1,200-row synthetic wildfire dataset."""
    resource = files("stsckm").joinpath("data/sample_data.csv")
    with resource.open("r", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame
