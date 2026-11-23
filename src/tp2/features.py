"""Features geomÃ©tricas e combinatÃ³rias para a anÃ¡lise das instÃ¢ncias."""

from __future__ import annotations

import math

import numpy as np

from tp2.convex_hull import convex_hull
from tp2.models import TSPInstance
from tp2.mst import prim_mst


def _spatial_entropy(coordinates: np.ndarray, grid_size: int) -> float:
    minimum = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - minimum
    normalized = np.zeros_like(coordinates, dtype=float)
    nonzero = span > 0
    normalized[:, nonzero] = (coordinates[:, nonzero] - minimum[nonzero]) / span[nonzero]
    cells = np.minimum((normalized * grid_size).astype(int), grid_size - 1)
    counts = np.zeros((grid_size, grid_size), dtype=int)
    for x_cell, y_cell in cells:
        counts[x_cell, y_cell] += 1
    probabilities = counts[counts > 0] / coordinates.shape[0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum = math.log(grid_size * grid_size)
    return entropy / maximum if maximum else 0.0


def extract_features(instance: TSPInstance, *, grid_size: int = 4) -> dict[str, float]:
    """Retorna um vetor determinÃ­stico de features sem NaN ou infinito."""

    if instance.dimension == 0:
        raise ValueError("instance must contain at least one vertex")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    coordinates = np.asarray(instance.coordinates, dtype=float)
    n = instance.dimension
    sides = coordinates.max(axis=0) - coordinates.min(axis=0)
    largest_side = float(max(sides))
    smallest_side = float(min(sides))
    if largest_side == 0:
        aspect_ratio = 1.0
    elif smallest_side == 0:
        aspect_ratio = largest_side
    else:
        aspect_ratio = largest_side / smallest_side

    hull_size = len(convex_hull(coordinates))
    mst = prim_mst(instance)
    degrees = np.asarray(mst.degrees, dtype=float)
    if n > 1:
        nearest = np.min(
            np.where(np.eye(n, dtype=bool), np.inf, instance.distance_matrix),
            axis=1,
        ).astype(float)
        nearest_mean = float(np.mean(nearest))
        nearest_cv = float(np.std(nearest) / nearest_mean) if nearest_mean else 0.0
        pairwise = instance.distance_matrix[np.triu_indices(n, k=1)].astype(float)
        mean_pairwise = float(np.mean(pairwise)) if pairwise.size else 0.0
    else:
        nearest_cv = 0.0
        mean_pairwise = 0.0

    diagonal = math.hypot(*sides)
    normalized_mst = (
        float(mst.cost) / ((n - 1) * diagonal)
        if n > 1 and diagonal > 0
        else 0.0
    )
    normalized_mean = mean_pairwise / diagonal if diagonal > 0 else 0.0

    return {
        "dimension": float(n),
        "aspect_ratio": float(aspect_ratio),
        "hull_fraction": float(hull_size / n),
        "spatial_entropy": _spatial_entropy(coordinates, grid_size),
        "mst_max_degree": float(np.max(degrees)),
        "mst_degree_mean": float(np.mean(degrees)),
        "mst_degree_std": float(np.std(degrees)),
        "nearest_neighbor_cv": nearest_cv,
        "mean_pairwise_distance": mean_pairwise,
        "density_mean": normalized_mean,
        "mst_weight_normalized": normalized_mst,
    }
