from __future__ import annotations

import numpy as np
import pytest


def _instance():
    from tp2.models import TSPInstance

    return TSPInstance(
        name="square",
        coordinates=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        distance_matrix=np.array(
            [[0, 2, 2, 2], [2, 0, 2, 2], [2, 2, 0, 2], [2, 2, 2, 0]]
        ),
    )


def test_prim_returns_spanning_tree_and_deterministic_ties() -> None:
    from tp2.mst import prim_mst

    result = prim_mst(_instance())

    assert result.edges == ((0, 1), (0, 2), (0, 3))
    assert result.cost == 6
    assert result.degrees == (3, 1, 1, 1)
    assert result.parent == (-1, 0, 0, 0)


def test_prim_handles_single_vertex() -> None:
    from tp2.models import TSPInstance
    from tp2.mst import prim_mst

    instance = TSPInstance("one", [[0., 0.]], [[0]])
    result = prim_mst(instance)

    assert result.edges == ()
    assert result.cost == 0
    assert result.degrees == (0,)


def test_prim_rejects_empty_instance() -> None:
    from tp2.mst import prim_mst

    with pytest.raises(ValueError, match="at least one"):
        prim_mst(TSPInstance_for_empty())


def TSPInstance_for_empty():
    from tp2.models import TSPInstance

    return TSPInstance("empty", np.empty((0, 2)), np.empty((0, 0), dtype=int))
