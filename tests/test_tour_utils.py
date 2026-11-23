from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def square_instance():
    from tp2.models import TSPInstance

    return TSPInstance(
        name="square",
        coordinates=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ),
        distance_matrix=np.array(
            [
                [0, 2, 3, 2],
                [2, 0, 2, 3],
                [3, 2, 0, 2],
                [2, 3, 2, 0],
            ]
        ),
    )


def test_validate_tour_accepts_each_vertex_exactly_once(square_instance) -> None:
    from tp2.models import Tour
    from tp2.tour_utils import validate_tour

    tour = Tour((0, 1, 2, 3))

    assert validate_tour(square_instance, tour) is None


@pytest.mark.parametrize(
    ("route", "message"),
    [
        ((0, 1, 1, 3), "repeated"),
        ((0, 1, 2), "exactly 4"),
        ((0, 1, 2, 4), "out of range"),
        ((0, 1, 2, 3, 0), "must not repeat"),
    ],
)
def test_validate_tour_rejects_invalid_routes(
    square_instance,
    route: tuple[int, ...],
    message: str,
) -> None:
    from tp2.tour_utils import validate_tour

    with pytest.raises(ValueError, match=message):
        validate_tour(square_instance, route)


def test_tour_cost_includes_the_return_edge(square_instance) -> None:
    from tp2.models import Tour
    from tp2.tour_utils import tour_cost

    assert tour_cost(square_instance, Tour((0, 1, 2, 3))) == 8
    assert tour_cost(square_instance, (0, 2, 1, 3)) == 10


def test_tour_cost_accumulates_large_weights_without_numpy_overflow() -> None:
    from tp2.models import TSPInstance
    from tp2.tour_utils import tour_cost

    weight = 2**62
    instance = TSPInstance("large", [[0., 0.], [1., 0.]], [[0, weight], [weight, 0]])

    assert tour_cost(instance, (0, 1)) == 2**63


def test_validate_tour_accepts_numpy_indices_but_rejects_booleans(
    square_instance,
) -> None:
    from tp2.tour_utils import validate_tour

    assert validate_tour(square_instance, np.arange(4)) is None
    with pytest.raises(ValueError, match="integer indices"):
        validate_tour(square_instance, (False, 1, 2, 3))

