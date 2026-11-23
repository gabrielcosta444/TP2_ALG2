from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def square_instance():
    from tp2.models import TSPInstance

    return TSPInstance(
        name="square",
        coordinates=np.array(
            [[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.]]
        ),
        distance_matrix=np.array(
            [[0, 2, 3, 2, 2], [2, 0, 2, 3, 1], [3, 2, 0, 2, 1],
             [2, 3, 2, 0, 1], [2, 1, 1, 1, 0]]
        ),
    )


def test_trivial_tour_uses_vertex_order_and_reports_cost(square_instance) -> None:
    from tp2.tour_utils import tour_cost
    from tp2.warm_starts import trivial_tour

    result = trivial_tour(square_instance)

    assert result.strategy == "trivial"
    assert result.tour.vertices == (0, 1, 2, 3, 4)
    assert result.cost == tour_cost(square_instance, result.tour)
    assert result.elapsed_seconds >= 0


def test_nearest_neighbor_multistart_uses_best_start_and_id_ties(square_instance) -> None:
    from tp2.warm_starts import nearest_neighbor_multistart

    result = nearest_neighbor_multistart(square_instance)

    assert sorted(result.tour.vertices) == list(range(square_instance.dimension))
    assert result.cost == 8
    assert result.strategy == "nearest_neighbor_multistart"


def test_convex_hull_insertion_includes_interior_vertices(square_instance) -> None:
    from tp2.warm_starts import convex_hull_cheapest_insertion

    result = convex_hull_cheapest_insertion(square_instance)

    assert sorted(result.tour.vertices) == list(range(square_instance.dimension))
    assert result.cost == 8
    assert result.strategy == "convex_hull_cheapest_insertion"


def test_warm_starts_reject_empty_instance() -> None:
    from tp2.models import TSPInstance
    from tp2.warm_starts import convex_hull_cheapest_insertion, nearest_neighbor_multistart, trivial_tour

    instance = TSPInstance("empty", np.empty((0, 2)), np.empty((0, 0), dtype=int))
    for function in (trivial_tour, nearest_neighbor_multistart, convex_hull_cheapest_insertion):
        with pytest.raises(ValueError, match="at least one"):
            function(instance)


def test_convex_hull_insertion_uses_costs_when_all_coordinates_are_equal() -> None:
    from tp2.models import TSPInstance
    from tp2.warm_starts import convex_hull_cheapest_insertion

    instance = TSPInstance(
        "same-coordinates",
        [[0., 0.]] * 4,
        [[0, 9, 1, 2], [9, 0, 2, 1], [1, 2, 0, 9], [2, 1, 9, 0]],
    )

    result = convex_hull_cheapest_insertion(instance)

    assert result.cost == 6
