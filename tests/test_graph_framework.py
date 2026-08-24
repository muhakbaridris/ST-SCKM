import numpy as np
import pytest
from scipy import sparse

from stsckm import adjacency_to_neighbors, spatial_graph, validate_adjacency


@pytest.fixture
def coordinates():
    return np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0]])


def test_radius_graph_supports_variable_degree(coordinates):
    graph = spatial_graph(coordinates, graph_type="radius", radius=0.15)
    assert sparse.isspmatrix_csr(graph)
    assert [len(row) for row in adjacency_to_neighbors(graph)] == [1, 1, 1, 1]


def test_union_symmetrization(coordinates):
    directed = spatial_graph(coordinates, n_neighbors=1, symmetrize="none")
    union = validate_adjacency(directed, symmetrize="union")
    np.testing.assert_array_equal(union.toarray(), union.toarray().T)


def test_mutual_symmetrization_removes_one_way_edge():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=float)
    mutual = validate_adjacency(adjacency, symmetrize="mutual")
    assert mutual.nnz == 2
    assert mutual[1, 2] == 0


def test_weighted_custom_adjacency_is_retained():
    adjacency = np.array([[0, 2.5], [0.5, 0]])
    graph = validate_adjacency(adjacency)
    np.testing.assert_allclose(graph.toarray(), adjacency)


@pytest.mark.parametrize(
    "adjacency",
    [np.ones((2, 3)), np.array([[0.0, -1.0], [0.0, 0.0]])],
)
def test_invalid_adjacency_is_rejected(adjacency):
    with pytest.raises(ValueError):
        validate_adjacency(adjacency)


def test_radius_must_be_positive(coordinates):
    with pytest.raises(ValueError, match="radius"):
        spatial_graph(coordinates, graph_type="radius", radius=0)


def test_validation_removes_diagonal_and_rejects_bad_modes():
    graph = validate_adjacency(np.eye(3))
    assert graph.nnz == 0
    with pytest.raises(ValueError, match="symmetrize"):
        validate_adjacency(np.eye(3), symmetrize="bad")


@pytest.mark.parametrize(
    "adjacency",
    [np.array([0.0, 1.0]), np.array([[0.0, np.inf], [0.0, 0.0]])],
)
def test_validation_rejects_invalid_shape_or_values(adjacency):
    with pytest.raises(ValueError):
        validate_adjacency(adjacency)


def test_spatial_graph_rejects_unknown_type(coordinates):
    with pytest.raises(ValueError, match="graph_type"):
        spatial_graph(coordinates, graph_type="unknown")
