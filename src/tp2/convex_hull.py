"""Fecho convexo 2-D por monotonic chain."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    values = tuple(float(value) for point in (a, b, c) for value in point)
    if all(value.is_integer() and abs(value) <= 2**53 for value in values):
        ax, ay, bx, by, cx, cy = (int(value) for value in values)
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return float(
        np.longdouble(b[0] - a[0]) * np.longdouble(c[1] - a[1])
        - np.longdouble(b[1] - a[1]) * np.longdouble(c[0] - a[0])
    )


def convex_hull(coordinates: ArrayLike) -> tuple[int, ...]:
    """Retorna IDs originais do fecho em ordem anti-horária."""

    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("coordinates must have shape (n, 2)")
    unique: dict[tuple[float, float], int] = {}
    for index, point in enumerate(points):
        unique.setdefault((float(point[0]), float(point[1])), index)
    ordered = sorted((point, index) for point, index in unique.items())
    if len(ordered) <= 1:
        return tuple(index for _, index in ordered)

    lower: list[tuple[tuple[float, float], int]] = []
    for item in ordered:
        while len(lower) >= 2 and _cross(
            np.asarray(lower[-2][0]), np.asarray(lower[-1][0]), np.asarray(item[0])
        ) <= 0:
            lower.pop()
        lower.append(item)
    upper: list[tuple[tuple[float, float], int]] = []
    for item in reversed(ordered):
        while len(upper) >= 2 and _cross(
            np.asarray(upper[-2][0]), np.asarray(upper[-1][0]), np.asarray(item[0])
        ) <= 0:
            upper.pop()
        upper.append(item)
    return tuple(index for _, index in lower[:-1] + upper[:-1])
