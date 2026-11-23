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


def test_odd_vertices_and_perfect_matching() -> None:
    from tp2.christofides import minimum_weight_perfect_matching, odd_degree_vertices
    from tp2.mst import prim_mst

    instance = _square()
    mst = prim_mst(instance)
    odd = odd_degree_vertices(mst)
    matching = minimum_weight_perfect_matching(instance, odd)

    assert odd == (2, 3)
    assert len(matching) == len(odd) // 2
    assert sorted(vertex for edge in matching for vertex in edge) == list(odd)
    assert all(left < right for left, right in matching)


def test_christofides_returns_valid_tour_and_phase_timings() -> None:
    from tp2.christofides import christofides
    from tp2.tour_utils import tour_cost, validate_tour

    instance = _square()
    result = christofides(instance)

    validate_tour(instance, result.tour)
    assert result.cost == tour_cost(instance, result.tour) == 8
    assert result.strategy == "christofides"
    assert set(result.phase_timings) == {"mst", "matching", "euler", "shortcutting"}
    assert all(value >= 0 for value in result.phase_timings.values())


def test_christofides_rejects_empty_instance() -> None:
    from tp2.christofides import christofides
    from tp2.models import TSPInstance

    instance = TSPInstance("empty", np.empty((0, 2)), np.empty((0, 0), dtype=int))
    with pytest.raises(ValueError, match="at least one"):
        christofides(instance)
