from __future__ import annotations

import numpy as np
import pytest


def _square():
    from tp2.models import TSPInstance

    return TSPInstance(
        "square",
        [[0., 0.], [2., 0.], [2., 2.], [0., 2.]],
        [[0, 2, 3, 2], [2, 0, 2, 3], [3, 2, 0, 2], [2, 3, 2, 0]],
    )


def test_twice_around_returns_valid_shortcut_tour_and_phase_timings() -> None:
    from tp2.tour_utils import tour_cost, validate_tour
    from tp2.twice_around import twice_around_tree

    instance = _square()
    result = twice_around_tree(instance)

    validate_tour(instance, result.tour)
    assert result.tour.vertices == (0, 1, 2, 3)
    assert result.cost == tour_cost(instance, result.tour) == 8
    assert result.strategy == "twice_around_the_tree"
    assert set(result.phase_timings) == {"mst", "traversal", "route"}
    assert all(value >= 0 for value in result.phase_timings.values())


def test_twice_around_uses_sorted_child_ties() -> None:
    from tp2.twice_around import twice_around_tree

    result = twice_around_tree(_square())

    assert result.tour.vertices[1:] == (1, 2, 3)


def test_twice_around_rejects_empty_instance() -> None:
    from tp2.models import TSPInstance
    from tp2.twice_around import twice_around_tree

    instance = TSPInstance("empty", np.empty((0, 2)), np.empty((0, 0), dtype=int))
    with pytest.raises(ValueError, match="at least one"):
        twice_around_tree(instance)
