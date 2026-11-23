"""Aproximação Twice-around-the-tree para TSP métrico."""

from __future__ import annotations

from tp2.models import TSPInstance, Tour
from tp2.mst import prim_mst
from tp2.profiling import profile_call
from tp2.tour_utils import tour_cost
from tp2.warm_starts import WarmStartResult


def _preorder_from_mst(parent: tuple[int, ...]) -> tuple[int, ...]:
    children: list[list[int]] = [[] for _ in parent]
    for vertex, father in enumerate(parent):
        if father >= 0:
            children[father].append(vertex)
    for siblings in children:
        siblings.sort()
    order: list[int] = []
    stack = [0]
    while stack:
        vertex = stack.pop()
        order.append(vertex)
        stack.extend(reversed(children[vertex]))
    return tuple(order)


def _build_tour_and_cost(
    instance: TSPInstance,
    vertices: tuple[int, ...],
) -> tuple[Tour, int]:
    tour = Tour(vertices)
    return tour, tour_cost(instance, tour)


def twice_around_tree(instance: TSPInstance) -> WarmStartResult:
    """Duplica conceitualmente a MST, percorre-a e faz shortcutting."""

    if instance.dimension == 0:
        raise ValueError("instance must contain at least one vertex")

    mst_profile = profile_call(prim_mst, instance)
    traversal_profile = profile_call(_preorder_from_mst, mst_profile.value.parent)
    route_profile = profile_call(
        _build_tour_and_cost,
        instance,
        traversal_profile.value,
    )
    tour, cost = route_profile.value
    return WarmStartResult(
        strategy="twice_around_the_tree",
        tour=tour,
        cost=cost,
        elapsed_seconds=(
            mst_profile.elapsed_seconds
            + traversal_profile.elapsed_seconds
            + route_profile.elapsed_seconds
        ),
        phase_timings={
            "mst": mst_profile.elapsed_seconds,
            "traversal": traversal_profile.elapsed_seconds,
            "route": route_profile.elapsed_seconds,
        },
    )
