from __future__ import annotations

import numpy as np


def _instance(points: list[tuple[float, float]]):
    from tp2.models import TSPInstance

    coords = np.asarray(points, dtype=float)
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.rint(np.sqrt((delta * delta).sum(axis=2))).astype(int)
    return TSPInstance("feature_case", coords, distances)


def test_features_are_deterministic_and_cover_controls() -> None:
    from tp2.features import extract_features

    instance = _instance([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)])
    first = extract_features(instance)
    second = extract_features(instance)

    assert first == second
    assert first["dimension"] == 5
    assert first["hull_fraction"] == 4 / 5
    assert first["aspect_ratio"] == 1.0
    assert first["mst_max_degree"] >= 1
    assert first["mst_weight_normalized"] >= 0


def test_degenerate_and_duplicate_configurations_have_finite_features() -> None:
    from tp2.features import extract_features

    instances = [
        _instance([(0, 0), (1, 0), (2, 0), (3, 0)]),
        _instance([(0, 0), (0, 0), (0, 0), (0, 0)]),
    ]
    for instance in instances:
        features = extract_features(instance)
        assert all(np.isfinite(value) for value in features.values())
        assert features["nearest_neighbor_cv"] == 0.0


def test_scale_and_translation_preserve_shape_features() -> None:
    from tp2.features import extract_features

    base = _instance([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)])
    transformed = _instance([(10, -7), (14, -7), (14, -3), (10, -3), (12, -5)])

    base_features = extract_features(base)
    transformed_features = extract_features(transformed)
    for key in (
        "aspect_ratio",
        "hull_fraction",
        "spatial_entropy",
        "mst_max_degree",
        "mst_degree_mean",
        "mst_degree_std",
        "nearest_neighbor_cv",
    ):
        assert base_features[key] == transformed_features[key]
