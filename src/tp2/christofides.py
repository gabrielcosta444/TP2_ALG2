"""Heurística de Christofides para instâncias métricas simétricas."""

from __future__ import annotations

import networkx as nx

from tp2.models import TSPInstance, Tour
from tp2.mst import MSTResult, prim_mst
from tp2.multigraph import eulerian_circuit
from tp2.profiling import profile_call
from tp2.tour_utils import tour_cost
from tp2.warm_starts import WarmStartResult


def odd_degree_vertices(mst: MSTResult) -> tuple[int, ...]:
    """Retorna, em ordem crescente, os vÃ©rtices de grau Ã­mpar da MST."""

    return tuple(vertex for vertex, degree in enumerate(mst.degrees) if degree % 2)


def minimum_weight_perfect_matching(
    instance: TSPInstance,
    vertices: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Calcula um matching perfeito mÃ­nimo no subgrafo completo induzido."""

    ordered = tuple(sorted(vertices))
    if len(ordered) % 2:
        raise ValueError("matching vertices must have even cardinality")
    if len(set(ordered)) != len(ordered):
        raise ValueError("matching vertices must be unique")
    if any(vertex < 0 or vertex >= instance.dimension for vertex in ordered):
        raise ValueError("matching vertex is out of range")
    if not ordered:
        return ()

    graph = nx.Graph()
    graph.add_nodes_from(ordered)
    for position, left in enumerate(ordered):
        for right in ordered[position + 1 :]:
            graph.add_edge(
                left,
                right,
                weight=int(instance.distance_matrix[left, right]),
            )
    matching = nx.min_weight_matching(graph, weight="weight")
    edges = tuple(sorted((min(left, right), max(left, right)) for left, right in matching))
    covered = tuple(vertex for edge in edges for vertex in edge)
    if len(edges) != len(ordered) // 2 or tuple(sorted(covered)) != ordered:
        raise ValueError("matching algorithm did not return a perfect matching")
    return edges


def _shortcut_eulerian(circuit: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    route: list[int] = []
    for vertex in circuit:
        if vertex not in seen:
            seen.add(vertex)
            route.append(vertex)
    return tuple(route)


def _build_tour_and_cost(
    instance: TSPInstance,
    route: tuple[int, ...],
) -> tuple[Tour, int]:
    tour = Tour(route)
    return tour, tour_cost(instance, tour)


def christofides(instance: TSPInstance) -> WarmStartResult:
    """Executa MST, matching, circuito Euleriano e shortcutting."""

    if instance.dimension == 0:
        raise ValueError("instance must contain at least one vertex")

    mst_profile = profile_call(prim_mst, instance)
    odd = odd_degree_vertices(mst_profile.value)
    matching_profile = profile_call(
        minimum_weight_perfect_matching,
        instance,
        odd,
    )
    multigraph_edges = mst_profile.value.edges + matching_profile.value
    euler_profile = profile_call(
        eulerian_circuit,
        multigraph_edges,
        start=0,
    )
    shortcutting_profile = profile_call(
        _shortcut_eulerian,
        euler_profile.value,
    )
    route_profile = profile_call(
        _build_tour_and_cost,
        instance,
        shortcutting_profile.value,
    )
    tour, cost = route_profile.value
    timings = {
        "mst": mst_profile.elapsed_seconds,
        "matching": matching_profile.elapsed_seconds,
        "euler": euler_profile.elapsed_seconds,
        "shortcutting": shortcutting_profile.elapsed_seconds + route_profile.elapsed_seconds,
    }
    return WarmStartResult(
        strategy="christofides",
        tour=tour,
        cost=cost,
        elapsed_seconds=sum(timings.values()),
        phase_timings=timings,
    )
