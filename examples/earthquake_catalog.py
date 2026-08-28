"""Apply general graph-regularized K-means to an archived USGS catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stsckm import (
    GraphRegularizedKMeans,
    add_point_event_features,
    combine_adjacencies,
    evaluate_labels,
    load_sample_earthquakes,
    spatial_graph,
    standardize_features,
)


def main() -> None:
    """Fit a layered-graph model and print auditable summaries."""
    events = add_point_event_features(
        load_sample_earthquakes(),
        time_column="time",
        intensity_column="mag",
    )
    spatial, _ = standardize_features(events, ["x_proj", "y_proj"])
    temporal, _ = standardize_features(events, ["time_days"])
    attributes, _ = standardize_features(events, ["mag", "depth", "time_days"])

    spatial_layer = spatial_graph(
        spatial,
        n_neighbors=6,
        symmetrize="union",
    )
    temporal_layer = spatial_graph(
        temporal,
        n_neighbors=2,
        symmetrize="union",
    )
    layered_graph = combine_adjacencies(
        [spatial_layer, temporal_layer],
        weights=[0.8, 0.2],
        normalize="max",
        symmetrize="union",
    )

    model = GraphRegularizedKMeans(
        n_clusters=5,
        graph_penalty=0.75,
        feature_weights=[1.0, 0.5, 0.5],
        graph_symmetrize="union",
        random_state=42,
    ).fit(attributes, adjacency=layered_graph)
    events["cluster"] = model.labels_

    metrics = evaluate_labels(attributes, model.labels_)
    profile = (
        events.groupby("cluster", observed=False)
        .agg(
            n_events=("cluster", "size"),
            mean_magnitude=("mag", "mean"),
            mean_depth_km=("depth", "mean"),
            first_event=("event_time", "min"),
            last_event=("event_time", "max"),
        )
        .reset_index()
    )

    print(f"n_events={len(events)}")
    print(f"iterations={model.n_iter_}")
    print(f"objective={model.objective_:.8f}")
    print(pd.Series(metrics).to_string())
    print(pd.Series(model.graph_diagnostics_).to_string())
    print(profile.to_string(index=False))
    assert np.isfinite(model.objective_)


if __name__ == "__main__":
    main()
