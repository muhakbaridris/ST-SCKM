from stsckm import (
    EARTHQUAKE_DATA_SHA256,
    EARTHQUAKE_DATA_SOURCE,
    STSCKM,
    GraphRegularizedKMeans,
    __version__,
    load_sample_earthquakes,
)


def test_public_imports():
    assert STSCKM.__name__ == "STSCKM"
    assert GraphRegularizedKMeans.__name__ == "GraphRegularizedKMeans"
    assert __version__ == "2.1.0"


def test_archived_earthquake_data_loads_with_provenance():
    frame = load_sample_earthquakes()
    assert len(frame) == 924
    assert {"latitude", "longitude", "time", "mag", "depth"}.issubset(frame.columns)
    assert frame["time"].dt.tz is not None
    assert EARTHQUAKE_DATA_SOURCE.startswith("https://earthquake.usgs.gov/")
    assert len(EARTHQUAKE_DATA_SHA256) == 64
