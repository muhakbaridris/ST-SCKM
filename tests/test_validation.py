import numpy as np
import pandas as pd
import pytest

from stsckm import assign_risk_labels, evaluate_labels, neighbor_disagreement


def test_evaluate_labels_returns_three_metrics():
    X = np.array([[0.0], [0.1], [4.0], [4.1]])
    result = evaluate_labels(X, np.array([0, 0, 1, 1]))
    assert set(result) == {"silhouette", "calinski_harabasz", "davies_bouldin"}
    assert result["silhouette"] > 0


def test_neighbor_disagreement_known_value():
    labels = np.array([0, 0, 1])
    neighbors = np.array([[1], [0], [1]])
    assert neighbor_disagreement(labels, neighbors) == pytest.approx(1 / 3)


def test_risk_labels_follow_cluster_means():
    frame = pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1, -1],
            "intensity": [1.0, 2.0, 8.0, 10.0, 99.0],
        }
    )
    result = assign_risk_labels(
        frame,
        "cluster",
        "intensity",
        risk_labels=("Lower", "Higher"),
    )
    assert result.tolist() == ["Lower", "Lower", "Higher", "Higher", "Noise"]


def test_evaluate_labels_handles_noise_and_degenerate_partition():
    X = np.array([[0.0], [0.1], [2.0]])
    result = evaluate_labels(X, np.array([0, 0, -1]))
    assert all(np.isnan(value) for value in result.values())
    with pytest.raises(ValueError, match="aligned"):
        evaluate_labels(X, np.array([0, 1]))


def test_neighbor_disagreement_validates_shapes_and_indices():
    with pytest.raises(ValueError, match="one-dimensional"):
        neighbor_disagreement(np.array([[0, 1]]), np.array([[1], [0]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        neighbor_disagreement(np.array([0, 1]), np.array([1, 0]))
    with pytest.raises(ValueError, match="out-of-range"):
        neighbor_disagreement(np.array([0, 1]), np.array([[2], [0]]))
    assert np.isnan(neighbor_disagreement(np.array([0, 1]), np.empty((2, 0), dtype=int)))


def test_risk_labels_validate_columns_and_label_set():
    frame = pd.DataFrame({"cluster": [0], "intensity": [1.0]})
    with pytest.raises(KeyError):
        assign_risk_labels(frame, "unknown", "intensity")
    with pytest.raises(KeyError):
        assign_risk_labels(frame, "cluster", "unknown")
    with pytest.raises(ValueError, match="risk_labels"):
        assign_risk_labels(frame, "cluster", "intensity", risk_labels=())
