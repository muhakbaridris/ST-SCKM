import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from stsckm import GraphRegularizedKMeans, spatial_graph


@pytest.fixture
def feature_data():
    X = np.array(
        [
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.9],
            [0.2, 0.1, 1.1],
            [4.0, 4.1, 8.0],
            [4.1, 4.0, 8.2],
            [4.2, 4.1, 7.9],
        ]
    )
    return X, X[:, :2]


def test_general_estimator_populates_public_state(feature_data):
    X, graph_features = feature_data
    model = GraphRegularizedKMeans(
        n_clusters=2,
        feature_weights=[1.0, 1.0, 0.25],
        graph_penalty=0.5,
        n_neighbors=2,
        graph_symmetrize="union",
        random_state=4,
    ).fit(X, graph_features=graph_features)

    assert model.labels_.shape == (6,)
    assert model.cluster_centers_.shape == (2, 3)
    assert model.feature_weights_.tolist() == [1.0, 1.0, 0.25]
    assert model.adjacency_.shape == (6, 6)
    assert model.n_features_in_ == 3
    assert np.isfinite(model.objective_)
    assert model.n_iter_ == len(model.objective_history_)


def test_general_estimator_transform_and_centroid_prediction(feature_data):
    X, graph_features = feature_data
    model = GraphRegularizedKMeans(n_clusters=2, random_state=2).fit(
        X,
        graph_features=graph_features,
    )
    distances = model.transform(X)
    predicted = model.predict(X)
    assert distances.shape == (6, 2)
    np.testing.assert_array_equal(predicted, np.argmin(distances, axis=1))


def test_general_estimator_accepts_custom_graph(feature_data):
    X, graph_features = feature_data
    adjacency = spatial_graph(graph_features, n_neighbors=2, symmetrize="mutual")
    model = GraphRegularizedKMeans(n_clusters=2, random_state=5).fit(
        X,
        adjacency=adjacency,
    )
    assert model.graph_type_ == "custom"
    assert model.regularization_adjacency_.shape == adjacency.shape


def test_general_estimator_is_cloneable_and_pipeline_compatible(feature_data):
    X, graph_features = feature_data
    model = GraphRegularizedKMeans(n_clusters=2, feature_weights=(1.0, 1.0, 0.5))
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    labels = copied.fit_predict(X, y=np.zeros(len(X)), graph_features=graph_features)
    assert labels.shape == (6,)


def test_dataframe_feature_names_are_recorded(feature_data):
    X, graph_features = feature_data
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = GraphRegularizedKMeans(n_clusters=2).fit(
        frame,
        graph_features=graph_features,
    )
    assert model.feature_names_in_.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "feature_weights",
    [[1.0, 2.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
)
def test_invalid_feature_weights_are_rejected(feature_data, feature_weights):
    X, _ = feature_data
    with pytest.raises(ValueError, match="feature_weights|positive"):
        GraphRegularizedKMeans(
            n_clusters=2,
            feature_weights=feature_weights,
        ).fit(X)


def test_graph_features_must_align_and_cannot_accompany_adjacency(feature_data):
    X, graph_features = feature_data
    model = GraphRegularizedKMeans(n_clusters=2)
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(X, graph_features=graph_features[:-1])
    with pytest.raises(ValueError, match="not both"):
        model.fit(X, adjacency=np.eye(len(X)), graph_features=graph_features)


def test_radius_graph_and_scalar_feature_weight(feature_data):
    X, graph_features = feature_data
    model = GraphRegularizedKMeans(
        n_clusters=2,
        feature_weights=2.0,
        graph_type="radius",
        radius=0.5,
    ).fit(X, graph_features=graph_features)
    assert model.graph_type_ == "radius"
    np.testing.assert_array_equal(model.feature_weights_, np.full(3, 2.0))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_clusters": 1}, "n_clusters"),
        ({"graph_penalty": -1}, "graph_penalty"),
        ({"graph_type": "bad"}, "graph_type"),
        ({"graph_symmetrize": "bad"}, "graph_symmetrize"),
        ({"update_scheme": "bad"}, "update_scheme"),
        ({"graph_type": "radius", "radius": None}, "radius"),
    ],
)
def test_invalid_general_parameters_are_rejected(feature_data, settings, message):
    X, _ = feature_data
    with pytest.raises(ValueError, match=message):
        GraphRegularizedKMeans(**settings).fit(X)


def test_general_cluster_count_cannot_exceed_sample_count(feature_data):
    X, _ = feature_data
    with pytest.raises(ValueError, match="cannot exceed"):
        GraphRegularizedKMeans(n_clusters=7).fit(X)
