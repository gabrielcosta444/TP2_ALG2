"""Heurísticas usadas para inicializar o limite superior do B&B."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from tp2.convex_hull import convex_hull
from tp2.models import TSPInstance, Tour
from tp2.profiling import profile_call
from tp2.tour_utils import tour_cost


@dataclass(frozen=True, slots=True)
class WarmStartResult:
    strategy: str
    tour: Tour
    cost: int
    elapsed_seconds: float
    phase_timings: Mapping[str, float] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_timings", MappingProxyType(dict(self.phase_timings)))


def _require_nonempty(instance: TSPInstance) -> None:
    if instance.dimension == 0:
        raise ValueError("instance must contain at least one vertex")


def trivial_tour(instance: TSPInstance) -> WarmStartResult:
    """Retorna a rota 0, 1, ..., N-1."""

    _require_nonempty(instance)
    def solve() -> tuple[Tour, int]:
        tour = Tour(tuple(range(instance.dimension)))
        return tour, tour_cost(instance, tour)

    profiled = profile_call(solve)
    tour, cost = profiled.value
    return WarmStartResult("trivial", tour, cost, profiled.elapsed_seconds)


def _nearest_neighbor(instance: TSPInstance, start: int) -> Tour:
    route = [start]
    unvisited = set(range(instance.dimension))
    unvisited.remove(start)
    while unvisited:
        current = route[-1]
        next_vertex = min(
            unvisited,
            key=lambda vertex: (int(instance.distance_matrix[current, vertex]), vertex),
        )
        route.append(next_vertex)
        unvisited.remove(next_vertex)
    return Tour(tuple(route))


def nearest_neighbor_multistart(instance: TSPInstance) -> WarmStartResult:
    """Executa vizinho mais próximo a partir de todos os vértices."""

    _require_nonempty(instance)

    def solve() -> tuple[Tour, int]:
        candidates = (_nearest_neighbor(instance, start) for start in range(instance.dimension))
        tour = min(candidates, key=lambda candidate: (tour_cost(instance, candidate), candidate.vertices))
        return tour, tour_cost(instance, tour)

    profiled = profile_call(solve)
    tour, cost = profiled.value
    return WarmStartResult(
        "nearest_neighbor_multistart", tour, cost, profiled.elapsed_seconds
    )


def convex_hull_cheapest_insertion(instance: TSPInstance) -> WarmStartResult:
    """Inicia pelo fecho convexo e insere internos pelo menor delta global."""

    _require_nonempty(instance)

    def solve() -> tuple[Tour, int]:
        hull = list(convex_hull(instance.coordinates))
        route = hull
        remaining = set(range(instance.dimension)) - set(route)
        while remaining:
            best_choice: tuple[int, int, int] | None = None
            for vertex in sorted(remaining):
                for position in range(len(route)):
                    left = route[position]
                    right = route[(position + 1) % len(route)]
                    delta = (
                        int(instance.distance_matrix[left, vertex])
                        + int(instance.distance_matrix[vertex, right])
                        - int(instance.distance_matrix[left, right])
                    )
                    choice = (delta, vertex, position + 1)
                    if best_choice is None or choice < best_choice:
                        best_choice = choice
            assert best_choice is not None
            _, vertex, insertion_position = best_choice
            route.insert(insertion_position, vertex)
            remaining.remove(vertex)
        tour = Tour(tuple(route))
        return tour, tour_cost(instance, tour)

    profiled = profile_call(solve)
    tour = profiled.value[0]
    return WarmStartResult(
        "convex_hull_cheapest_insertion", tour, profiled.value[1], profiled.elapsed_seconds
    )
