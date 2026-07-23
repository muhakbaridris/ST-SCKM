"""Optional plotting helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def plot_spatial_clusters(
    frame: pd.DataFrame,
    label_column: str,
    *,
    x: str = "longitude",
    y: str = "latitude",
    ax: Any = None,
    point_size: float = 12,
    alpha: float = 0.8,
) -> Any:
    """Plot a spatial partition and return the Matplotlib axes.

    Matplotlib is an optional dependency installed with ``stsckm[plot]``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "plot_spatial_clusters requires Matplotlib; install stsckm[plot]"
        ) from error

    for column in (x, y, label_column):
        if column not in frame:
            raise KeyError(f"{column!r} is not a column in frame")
    if ax is None:
        _, ax = plt.subplots()
    for label, subset in frame.groupby(label_column, observed=False):
        ax.scatter(
            subset[x],
            subset[y],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            label=str(label),
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=label_column)
    return ax
