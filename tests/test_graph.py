import numpy as np
import pytest

from stsckm.graph import knn_indices


def test_knn_shape_and_no_self_neighbors():
    spatial = np.arange(10, dtype=float).reshape(-1, 1)
    result = knn_indices(spatial, n_neighbors=3)
    assert result.shape == (10, 3)
    for index, row in enumerate(result):
        assert index not in row


def test_knn_caps_neighbor_count():
    result = knn_indices(np.array([[0.0], [1.0], [2.0]]), n_neighbors=20)
    assert result.shape == (3, 2)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_knn_rejects_invalid_neighbor_count(value):
    with pytest.raises(ValueError, match="n_neighbors"):
        knn_indices(np.array([[0.0], [1.0]]), n_neighbors=value)


def test_knn_requires_two_observations():
    with pytest.raises(ValueError, match="two observations"):
        knn_indices(np.array([[0.0]]), n_neighbors=1)
