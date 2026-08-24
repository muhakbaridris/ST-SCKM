import matplotlib
import pandas as pd
import pytest

from stsckm.plotting import plot_spatial_clusters

matplotlib.use("Agg")


def test_plot_spatial_clusters_returns_labeled_axes():
    frame = pd.DataFrame(
        {
            "longitude": [0.0, 0.1, 1.0, 1.1],
            "latitude": [0.0, 0.1, 1.0, 1.1],
            "cluster": [0, 0, 1, 1],
        }
    )
    axes = plot_spatial_clusters(frame, "cluster", point_size=5, alpha=0.5)
    assert axes.get_xlabel() == "longitude"
    assert axes.get_ylabel() == "latitude"
    assert len(axes.collections) == 2


def test_plot_spatial_clusters_rejects_missing_columns():
    with pytest.raises(KeyError, match="latitude"):
        plot_spatial_clusters(pd.DataFrame({"longitude": [0], "cluster": [0]}), "cluster")
