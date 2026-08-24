import numpy as np

from stsckm import STSCKM, fit_stability, parameter_search


def sample_data():
    spatial = np.array([[0, 0], [0.1, 0], [0, 0.1], [3, 3], [3.1, 3], [3, 3.1]])
    temporal = np.array([[0], [0.1], [0.2], [3], [3.1], [3.2]])
    return spatial, temporal


def test_parameter_search_returns_quality_and_graph_metrics():
    spatial, temporal = sample_data()
    result = parameter_search(
        spatial,
        temporal,
        {"lambda_spatial": [0.0, 0.5], "n_neighbors": [1]},
        estimator=STSCKM(n_clusters=2, random_state=0),
    )
    assert len(result) == 2
    assert {
        "silhouette",
        "edge_disagreement",
        "disconnected_clusters",
        "objective",
    }.issubset(result.columns)


def test_fit_stability_is_reproducible_for_separated_data():
    spatial, temporal = sample_data()
    result = fit_stability(
        STSCKM(n_clusters=2, lambda_spatial=0),
        spatial,
        temporal,
        seeds=(1, 2, 3),
    )
    assert result.mean_adjusted_rand == 1.0
    assert len(result.pairwise_adjusted_rand) == 3


def test_fit_stability_requires_two_seeds():
    spatial, temporal = sample_data()
    try:
        fit_stability(STSCKM(n_clusters=2), spatial, temporal, seeds=(1,))
    except ValueError as error:
        assert "two seeds" in str(error)
    else:
        raise AssertionError("fit_stability should reject one seed")
