"""Executable counterparts of the code listings in the manuscript.

The manuscript imports these functions directly so that the printed listings
and the submitted replication code cannot silently diverge.
"""

from __future__ import annotations

import numpy as np

from stsckm import (
    STSCKM,
    add_default_features,
    assign_risk_labels,
    cluster_connectivity,
    evaluate_labels,
    graph_diagnostics,
    load_sample_wildfire,
    parameter_search,
    spatial_graph,
    standardize_features,
)


def prepare_example():
    """Prepare the bundled point-event data for clustering."""
    events = add_default_features(load_sample_wildfire())
    X_spatial, spatial_scaler = standardize_features(
        events, ["x_proj", "y_proj"]
    )
    X_temporal, temporal_scaler = standardize_features(events, ["time_days"])
    return events, X_spatial, X_temporal, spatial_scaler, temporal_scaler


def fit_example(X_spatial, X_temporal):
    """Fit the model used in the main software illustration."""
    model = STSCKM(
        n_clusters=4,
        spatial_weight=0.5,
        temporal_weight=1.5,
        lambda_spatial=1.0,
        graph_type="knn",
        n_neighbors=5,
        graph_symmetrize="union",
        update_scheme="sequential",
        n_init=10,
        random_state=42,
    )
    labels = model.fit_predict(X_spatial, X_temporal)
    return model, labels


def evaluate_example(events, X_spatial, X_temporal, model):
    """Compute internal, graph, and post hoc summaries."""
    evaluation = np.column_stack([X_spatial, X_temporal])
    internal = evaluate_labels(evaluation, model.labels_)
    graph = graph_diagnostics(model.labels_, model.adjacency_)
    connectivity = cluster_connectivity(model.labels_, model.adjacency_)

    profiled = events.copy()
    profiled["cluster"] = model.labels_
    profiled["risk_zone"] = assign_risk_labels(
        profiled, label_column="cluster", intensity_column="log_frp"
    )
    return internal, graph, connectivity, profiled


def graph_examples(X_spatial, X_temporal):
    """Show radius and caller-supplied graph interfaces."""
    radius_model = STSCKM(
        n_clusters=4,
        graph_type="radius",
        radius=0.20,
        lambda_spatial=1.0,
        random_state=42,
    ).fit(X_spatial, X_temporal)

    adjacency = spatial_graph(
        X_spatial,
        graph_type="knn",
        n_neighbors=5,
        symmetrize="mutual",
    )
    custom_model = STSCKM(
        n_clusters=4,
        lambda_spatial=1.0,
        random_state=42,
    ).fit(X_spatial, X_temporal, adjacency=adjacency)
    return radius_model, custom_model


def tuning_example(X_spatial, X_temporal):
    """Search a compact grid with clustering and graph diagnostics."""
    return parameter_search(
        X_spatial,
        X_temporal,
        {
            "n_clusters": [3, 4],
            "lambda_spatial": [0.0, 0.5, 1.0, 2.0],
            "graph_symmetrize": ["union"],
        },
        estimator=STSCKM(n_neighbors=5, n_init=10, random_state=42),
    )


def main():
    """Run every manuscript listing as one smoke test."""
    events, X_spatial, X_temporal, _, _ = prepare_example()
    model, _ = fit_example(X_spatial, X_temporal)
    internal, graph, connectivity, _ = evaluate_example(
        events, X_spatial, X_temporal, model
    )
    radius_model, custom_model = graph_examples(X_spatial, X_temporal)
    tuning = tuning_example(X_spatial, X_temporal)

    print("Internal metrics:", internal)
    print("Graph diagnostics:", graph)
    print(connectivity.to_string(index=False))
    print("Radius graph diagnostics:", radius_model.graph_diagnostics_)
    print("Custom graph diagnostics:", custom_model.graph_diagnostics_)
    print(tuning.to_string(index=False))


if __name__ == "__main__":
    main()
