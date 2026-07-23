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
