from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_instance_copies_arrays_and_exposes_read_only_data() -> None:
    from tp2.models import TSPInstance

    coordinates = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    distances = np.array([[0, 3, 4], [3, 0, 5], [4, 5, 0]])

    instance = TSPInstance(
        name="triangle",
        coordinates=coordinates,
        distance_matrix=distances,
    )
    coordinates[0, 0] = 99.0
    distances[0, 1] = 99

    assert instance.name == "triangle"
    assert instance.dimension == 3
    assert instance.edge_weight_type == "EUC_2D"
    assert instance.coordinates[0, 0] == 0.0
    assert instance.distance_matrix[0, 1] == 3
    assert not instance.coordinates.flags.writeable
    assert not instance.distance_matrix.flags.writeable
    with pytest.raises(ValueError):
        instance.coordinates.setflags(write=True)
    with pytest.raises(ValueError):
        instance.distance_matrix.setflags(write=True)


@pytest.mark.parametrize(
    ("coordinates", "distances", "message"),
    [
        ([[0.0, 0.0]], [[0, 1], [1, 0]], "dimensions"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0, 1], [2, 0]], "symmetric"),
        ([[0.0, 0.0], [1.0, 1.0]], [[1, 1], [1, 0]], "diagonal"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0, -1], [-1, 0]], "negative"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0, -0.5], [0.9, 0]], "negative"),
    ],
)
def test_instance_rejects_invalid_distance_matrix(
    coordinates: list[list[float]],
    distances: list[list[int]],
    message: str,
) -> None:
    from tp2.models import TSPInstance

    with pytest.raises(ValueError, match=message):
        TSPInstance(
            name="invalid",
            coordinates=coordinates,
            distance_matrix=distances,
        )


def test_instance_preserves_large_int64_weights_exactly() -> None:
    from tp2.models import TSPInstance

    maximum = np.iinfo(np.int64).max
    instance = TSPInstance(
        name="large-weights",
        coordinates=[[0.0, 0.0], [1.0, 0.0]],
        distance_matrix=[[0, maximum], [maximum, 0]],
    )

    assert int(instance.distance_matrix[0, 1]) == maximum


def test_tour_uses_open_cycle_representation() -> None:
    from tp2.models import Tour

    assert Tour((0, 2, 1)).vertices == (0, 2, 1)

    with pytest.raises(ValueError, match="must not repeat"):
        Tour((0, 2, 1, 0))


def test_tour_normalizes_numpy_integer_indices_and_rejects_booleans() -> None:
    from tp2.models import Tour

    assert Tour(np.arange(3)).vertices == (0, 1, 2)
    assert all(type(vertex) is int for vertex in Tour(np.arange(3)).vertices)

    with pytest.raises(ValueError, match="integer indices"):
        Tour((False, 1, 2))


def test_solver_result_preserves_metrics_without_mutable_aliases() -> None:
    from tp2.models import SolverResult, SolverStatus, Tour

    timings = {"total": 0.25}
    counters = {"expanded": 7}
    result = SolverResult(
        tour=Tour((0, 1, 2)),
        cost=12,
        status=SolverStatus.OPTIMAL,
        timings=timings,
        counters=counters,
    )
    timings["total"] = 99.0
    counters["expanded"] = 99

    assert result.timings == {"total": 0.25}
    assert result.counters == {"expanded": 7}
    with pytest.raises(TypeError):
        result.timings["other"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.cost = 13  # type: ignore[misc]

