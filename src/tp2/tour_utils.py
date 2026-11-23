"""Operações de validação e custo de ciclos Hamiltonianos."""

from __future__ import annotations

from collections.abc import Iterable

from tp2.models import TSPInstance, Tour


def _vertices(route: Tour | Iterable[int]) -> tuple[int, ...]:
    if isinstance(route, Tour):
        return route.vertices
    return Tour(route).vertices


def _validated_vertices(
    instance: TSPInstance,
    route: Tour | Iterable[int],
) -> tuple[int, ...]:
    vertices = _vertices(route)
    if len(vertices) != instance.dimension:
        raise ValueError(
            f"tour must contain exactly {instance.dimension} vertices"
        )
    if any(vertex < 0 or vertex >= instance.dimension for vertex in vertices):
        raise ValueError("tour vertex is out of range")
    if len(set(vertices)) != len(vertices):
        raise ValueError("tour contains repeated vertices")
    return vertices


def validate_tour(instance: TSPInstance, route: Tour | Iterable[int]) -> None:
    """Valida se ``route`` é um ciclo Hamiltoniano na representação aberta."""

    _validated_vertices(instance, route)


def tour_cost(instance: TSPInstance, route: Tour | Iterable[int]) -> int:
    """Calcula o custo do ciclo, incluindo a aresta de retorno ao início."""

    vertices = _validated_vertices(instance, route)
    return sum(
        int(instance.distance_matrix[vertices[index], vertices[(index + 1) % len(vertices)]])
        for index in range(len(vertices))
    )
