import numpy as np
import pandas as pd
import pytest

from stsckm import (
    add_default_features,
    add_point_event_features,
    generate_sample_wildfire_data,
    load_sample_wildfire,
    standardize_features,
)


def test_bundled_dataset_loads():
    frame = load_sample_wildfire()
    assert len(frame) == 1200
    assert pd.api.types.is_datetime64_any_dtype(frame["datetime"])


def test_default_features_are_created():
    frame = add_default_features(load_sample_wildfire().head(10))
    assert {"log_frp", "time_days", "x_proj", "y_proj"}.issubset(frame.columns)
    assert frame["time_days"].iloc[0] == 0


def test_standardization_returns_scaler():
    frame = add_default_features(load_sample_wildfire().head(20))
    values, scaler = standardize_features(frame, ["x_proj", "y_proj"])
    assert values.shape == (20, 2)
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-12)
    assert scaler.mean_.shape == (2,)


def test_generator_is_reproducible():
    first = generate_sample_wildfire_data(25, random_state=4)
    second = generate_sample_wildfire_data(25, random_state=4)
    pd.testing.assert_frame_equal(first, second)


def test_missing_columns_raise_clear_error():
    with pytest.raises(KeyError, match="missing required"):
        add_default_features(pd.DataFrame({"longitude": [1.0]}))


def test_generator_and_standardization_validate_inputs():
    with pytest.raises(ValueError, match="positive"):
        generate_sample_wildfire_data(0)
    frame = add_default_features(load_sample_wildfire().head(3))
    with pytest.raises(ValueError, match="columns"):
        standardize_features(frame, [])
    with pytest.raises(KeyError, match="missing requested"):
        standardize_features(frame, ["unknown"])


def test_add_point_event_features_supports_generic_columns():
    frame = pd.DataFrame(
        {
            "lon": [-120.0, -119.5],
            "lat": [35.0, 35.5],
            "occurred": ["2020-01-01", "2020-01-03"],
            "magnitude": [3.2, 4.1],
        }
    )
    result = add_point_event_features(
        frame,
        longitude_column="lon",
        latitude_column="lat",
        time_column="occurred",
        intensity_column="magnitude",
    )
    assert result["time_days"].tolist() == [0.0, 2.0]
    assert result["event_intensity"].tolist() == [3.2, 4.1]
    assert result["event_time"].dt.tz is not None
    assert np.isfinite(result[["x_proj", "y_proj"]]).all().all()
