import numpy as np

from stsckm.distance import weighted_spatiotemporal_distance


def test_weighted_distance_known_values():
    spatial = np.array([[0.0, 0.0], [2.0, 0.0]])
    temporal = np.array([[0.0], [2.0]])
    spatial_centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    temporal_centers = np.array([[0.0], [1.0]])

    result = weighted_spatiotemporal_distance(
        spatial,
        temporal,
        spatial_centers,
        temporal_centers,
        spatial_weight=2.0,
        temporal_weight=0.5,
    )

    expected = np.array([[0.0, 2.5], [10.0, 2.5]])
    np.testing.assert_allclose(result, expected)
