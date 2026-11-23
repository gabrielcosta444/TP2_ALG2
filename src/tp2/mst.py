"""Árvore geradora mínima por Prim sobre matriz densa."""

from __future__ import annotations

from dataclasses import dataclass

from tp2.models import TSPInstance


@dataclass(frozen=True, slots=True)
class MSTResult:
    edges: tuple[tuple[int, int], ...]
    cost: int
    parent: tuple[int, ...]
    degrees: tuple[int, ...]


def prim_mst(instance: TSPInstance) -> MSTResult:
    """Calcula uma MST com desempates determinísticos por índice."""

    n = instance.dimension
    if n == 0:
        raise ValueError("instance must contain at least one vertex")
    if n == 1:
        return MSTResult((), 0, (-1,), (0,))

    matrix = instance.distance_matrix
    selected = [False] * n
    best = [None] * n
    parent = [-1] * n
    best[0] = 0
    edges: list[tuple[int, int]] = []
    total = 0

    for _ in range(n):
        candidates = [vertex for vertex in range(n) if not selected[vertex] and best[vertex] is not None]
        if not candidates:
            raise ValueError("distance matrix graph is disconnected")
        vertex = min(candidates, key=lambda item: (best[item], item))
        selected[vertex] = True
        if parent[vertex] != -1:
            edge = (min(parent[vertex], vertex), max(parent[vertex], vertex))
            edges.append(edge)
            total += int(matrix[parent[vertex], vertex])
        for neighbor in range(n):
            if not selected[neighbor]:
                weight = int(matrix[vertex, neighbor])
                if best[neighbor] is None or (weight, vertex) < (best[neighbor], parent[neighbor]):
                    best[neighbor] = weight
                    parent[neighbor] = vertex

    degrees = [0] * n
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    return MSTResult(tuple(edges), total, tuple(parent), tuple(degrees))
