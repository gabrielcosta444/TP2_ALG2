from __future__ import annotations

import numpy as np


def _square():
    from tp2.models import TSPInstance

    return TSPInstance(
        "square",
        [[0., 0.], [2., 0.], [2., 2.], [0., 2.]],
        [[0, 2, 3, 2], [2, 0, 2, 3], [3, 2, 0, 2], [2, 3, 2, 0]],
    )


def test_branch_and_bound_proves_small_optimum() -> None:
    from tp2.branch_and_bound import branch_and_bound
    from tp2.models import SolverStatus
    from tp2.tour_utils import validate_tour

    result = branch_and_bound(_square())

    assert result.status is SolverStatus.OPTIMAL
    assert result.cost == 8
    assert result.tour is not None
    validate_tour(_square(), result.tour)
    assert result.counters["nodes_expanded"] > 0
    assert result.counters["bounds_pruned"] >= 0


def test_branch_and_bound_timeout_returns_feasible_incumbent() -> None:
    from tp2.branch_and_bound import branch_and_bound
    from tp2.models import SolverStatus, TSPInstance
    from tp2.tour_utils import validate_tour

    instance = TSPInstance(
        "larger",
        np.column_stack((np.arange(10), np.zeros(10))),
        np.fromfunction(lambda i, j: np.abs(i - j), (10, 10), dtype=int),
    )
    result = branch_and_bound(instance, time_limit_seconds=0)

    assert result.status is SolverStatus.TIMEOUT
    assert result.tour is not None and result.cost is not None
    validate_tour(instance, result.tour)
    assert result.counters["timeouts"] == 1


def test_branch_and_bound_rejects_empty_instance() -> None:
    from tp2.branch_and_bound import branch_and_bound
    from tp2.models import TSPInstance

    instance = TSPInstance("empty", np.empty((0, 2)), np.empty((0, 0), dtype=int))
    try:
        branch_and_bound(instance)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty instances must be rejected")
