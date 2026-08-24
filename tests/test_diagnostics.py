import numpy as np
import pytest

from stsckm import adjacency_disagreement, cluster_connectivity, graph_diagnostics


def test_weighted_adjacency_disagreement():
    labels = np.array([0, 0, 1])
    adjacency = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]], dtype=float)
    assert adjacency_disagreement(labels, adjacency) == 1 / 3


def test_connectivity_identifies_fragmented_cluster():
    labels = np.array([0, 1, 0, 1])
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    table = cluster_connectivity(labels, adjacency).set_index("cluster")
    assert table.loc[0, "n_components"] == 2
    assert table.loc[1, "n_components"] == 2
    summary = graph_diagnostics(labels, adjacency)
    assert summary["disconnected_clusters"] == 2
    assert summary["n_components_total"] == 4


def test_connected_partition_has_one_component_per_cluster():
    labels = np.array([0, 0, 1, 1])
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    summary = graph_diagnostics(labels, adjacency)
    assert summary["disconnected_clusters"] == 0
    assert summary["largest_component_fraction"] == 1.0


def test_zero_edge_graph_and_noise_only_partition():
    labels = np.array([-1, -1])
    adjacency = np.zeros((2, 2))
    assert np.isnan(adjacency_disagreement(labels, adjacency))
    assert cluster_connectivity(labels, adjacency).empty
    summary = graph_diagnostics(labels, adjacency)
    assert np.isnan(summary["neighbor_agreement"])
    assert summary["n_components_total"] == 0


@pytest.mark.parametrize("labels", [np.array([]), np.array([[0, 1]])])
def test_diagnostics_reject_invalid_labels(labels):
    with pytest.raises(ValueError):
        graph_diagnostics(labels, np.zeros((labels.size, labels.size)))
