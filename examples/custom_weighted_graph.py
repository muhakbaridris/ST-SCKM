"""Demonstrate an asymmetric caller-supplied weighted graph."""

import numpy as np

from stsckm import STSCKM


def main() -> None:
    """Fit and inspect a six-observation custom-graph example."""
    spatial = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.0], [1.8, 0.0], [2.0, 0.1], [2.2, 0.0]]
    )
    temporal = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    adjacency = np.zeros((6, 6), dtype=float)
    adjacency[0, 1] = 2.0
    adjacency[1, 0] = 1.0
    adjacency[1, 2] = 3.0
    adjacency[2, 3] = 0.25
    adjacency[3, 4] = 3.0
    adjacency[4, 3] = 1.0
    adjacency[4, 5] = 2.0

    model = STSCKM(
        n_clusters=2,
        spatial_weight=1.0,
        temporal_weight=1.0,
        lambda_spatial=0.75,
        graph_symmetrize="none",
        random_state=42,
    ).fit(spatial, temporal, adjacency=adjacency)

    print("Labels:", model.labels_)
    print("Supplied graph:\n", model.adjacency_.toarray())
    print("Regularization graph:\n", model.regularization_adjacency_.toarray())
    print("Diagnostics:", model.graph_diagnostics_)


if __name__ == "__main__":
    main()
