import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score

from stsckm import STSCKM


@pytest.fixture
def simple_data():
    spatial = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
        ]
    )
    temporal = np.array([[0.0], [0.1], [0.2], [4.0], [4.1], [4.2]])
    return spatial, temporal


def test_fit_populates_public_attributes(simple_data):
    spatial, temporal = simple_data
    model = STSCKM(n_clusters=2, n_neighbors=2, random_state=7).fit(
        spatial, temporal
    )
    assert model.labels_.shape == (6,)
    assert model.neighbors_.shape == (6, 2)
    assert model.cluster_centers_spatial_.shape == (2, 2)
    assert model.cluster_centers_temporal_.shape == (2, 1)
    assert model.n_iter_ == len(model.objective_history_)
    assert model.objective_ == model.objective_history_[-1]
    assert np.isfinite(model.objective_)


def test_fit_is_reproducible_for_integer_seed(simple_data):
    spatial, temporal = simple_data
    first = STSCKM(n_clusters=2, random_state=13).fit_predict(spatial, temporal)
    second = STSCKM(n_clusters=2, random_state=13).fit_predict(spatial, temporal)
    np.testing.assert_array_equal(first, second)


def test_zero_penalty_matches_weighted_kmeans_partition(simple_data):
    spatial, temporal = simple_data
    constrained = STSCKM(
        n_clusters=2,
        lambda_spatial=0.0,
        random_state=3,
    ).fit_predict(spatial, temporal)
    reference = STSCKM(
        n_clusters=2,
        lambda_spatial=0.0,
        random_state=3,
        max_iter=1,
    ).fit_predict(spatial, temporal)
    assert adjusted_rand_score(constrained, reference) == 1.0


def test_sklearn_clone_preserves_parameters():
    model = STSCKM(n_clusters=3, lambda_spatial=0.25, random_state=99)
    copied = clone(model)
    assert copied.get_params() == model.get_params()


def test_objective_history_returns_copy(simple_data):
    spatial, temporal = simple_data
    model = STSCKM(n_clusters=2).fit(spatial, temporal)
    history = model.get_objective_history()
    history[0] = -999
    assert model.objective_history_[0] != -999


def test_sequential_objective_is_nonincreasing(simple_data):
    spatial, temporal = simple_data
    model = STSCKM(
        n_clusters=2,
        lambda_spatial=0.5,
        graph_symmetrize="union",
        update_scheme="sequential",
    ).fit(spatial, temporal)
    differences = np.diff(model.get_objective_history())
    assert np.all(differences <= 1e-10)


def test_custom_sparse_adjacency(simple_data):
    spatial, temporal = simple_data
    adjacency = np.zeros((6, 6))
    adjacency[np.arange(5), np.arange(1, 6)] = 1
    model = STSCKM(n_clusters=2, graph_symmetrize="union").fit(
        spatial,
        temporal,
        adjacency=adjacency,
    )
    assert model.graph_type_ == "custom"
    assert model.adjacency_.shape == (6, 6)
    assert model.graph_diagnostics_["n_components_total"] >= 2


def test_radius_graph_and_variable_neighbors(simple_data):
    spatial, temporal = simple_data
    model = STSCKM(n_clusters=2, graph_type="radius", radius=0.15).fit(
        spatial,
        temporal,
    )
    assert model.graph_type_ == "radius"
    assert model.neighbors_ is None or model.neighbors_.ndim == 2
    assert len(model.neighbor_indices_) == len(spatial)


def test_synchronous_update_scheme(simple_data):
    spatial, temporal = simple_data
    model = STSCKM(n_clusters=2, update_scheme="synchronous").fit(spatial, temporal)
    assert model.update_scheme == "synchronous"
    assert model.labels_.shape == (6,)


def test_transform_returns_unregularized_distances(simple_data):
    spatial, temporal = simple_data
    model = STSCKM(n_clusters=2).fit(spatial, temporal)
    distances = model.transform(spatial, temporal)
    assert distances.shape == (6, 2)
    assert np.all(distances >= 0)


def test_directed_graph_creates_symmetric_regularization(simple_data):
    spatial, temporal = simple_data
    adjacency = np.zeros((6, 6))
    adjacency[0, 1] = 2.0
    model = STSCKM(n_clusters=2).fit(spatial, temporal, adjacency=adjacency)
    regularization = model.regularization_adjacency_.toarray()
    assert regularization[0, 1] == regularization[1, 0] == 1.0


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"graph_type": "invalid"}, "graph_type"),
        ({"graph_symmetrize": "invalid"}, "graph_symmetrize"),
        ({"update_scheme": "invalid"}, "update_scheme"),
        ({"graph_type": "radius", "radius": None}, "radius"),
    ],
)
def test_invalid_graph_controls_are_rejected(simple_data, settings, message):
    spatial, temporal = simple_data
    with pytest.raises(ValueError, match=message):
        STSCKM(n_clusters=2, **settings).fit(spatial, temporal)


def test_transform_validates_fitted_dimensions(simple_data):
    spatial, temporal = simple_data
    with pytest.raises(Exception, match="not fitted"):
        STSCKM(n_clusters=2).transform(spatial, temporal)
    model = STSCKM(n_clusters=2).fit(spatial, temporal)
    with pytest.raises(ValueError, match="same number of rows"):
        model.transform(spatial, temporal[:-1])
    with pytest.raises(ValueError, match="spatial.*features"):
        model.transform(spatial[:, :1], temporal)
    with pytest.raises(ValueError, match="temporal.*features"):
        model.transform(spatial, np.column_stack([temporal, temporal]))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_clusters", 1),
        ("n_neighbors", 0),
        ("max_iter", 0),
        ("n_init", 0),
        ("spatial_weight", -1.0),
        ("temporal_weight", -1.0),
        ("lambda_spatial", -1.0),
        ("tol", -1.0),
    ],
)
def test_invalid_parameters_are_rejected(simple_data, parameter, value):
    spatial, temporal = simple_data
    settings = {parameter: value}
    with pytest.raises(ValueError, match=parameter):
        STSCKM(**settings).fit(spatial, temporal)


def test_weights_cannot_both_be_zero(simple_data):
    spatial, temporal = simple_data
    with pytest.raises(ValueError, match="cannot both be zero"):
        STSCKM(spatial_weight=0, temporal_weight=0).fit(spatial, temporal)


def test_matrices_must_have_equal_rows(simple_data):
    spatial, temporal = simple_data
    with pytest.raises(ValueError, match="same number of rows"):
        STSCKM(n_clusters=2).fit(spatial, temporal[:-1])


def test_cluster_count_cannot_exceed_sample_count(simple_data):
    spatial, temporal = simple_data
    with pytest.raises(ValueError, match="cannot exceed"):
        STSCKM(n_clusters=7).fit(spatial, temporal)
