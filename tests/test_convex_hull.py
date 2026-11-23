from __future__ import annotations

import numpy as np


def test_convex_hull_returns_outer_ids_in_counterclockwise_order() -> None:
    from tp2.convex_hull import convex_hull

    coordinates = np.array(
        [[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.], [1., 0.]]
    )

    assert convex_hull(coordinates) == (0, 1, 2, 3)


def test_convex_hull_drops_collinear_boundary_points_and_duplicates() -> None:
    from tp2.convex_hull import convex_hull

    coordinates = np.array(
        [[0., 0.], [1., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]]
    )

    assert convex_hull(coordinates) == (0, 2, 3, 4)


def test_convex_hull_handles_collinear_and_small_inputs() -> None:
    from tp2.convex_hull import convex_hull

    assert convex_hull(np.array([[0., 0.], [1., 1.], [2., 2.]])) == (0, 2)
    assert convex_hull(np.array([[0., 0.]])) == (0,)
    assert convex_hull(np.empty((0, 2))) == ()


def test_convex_hull_preserves_orientation_for_large_integer_coordinates() -> None:
    from tp2.convex_hull import convex_hull

    coordinates = np.array(
        [[0., 0.], [100_000_000., 99_999_999.], [100_000_001., 100_000_000.]]
    )

    assert convex_hull(coordinates) == (0, 1, 2)
