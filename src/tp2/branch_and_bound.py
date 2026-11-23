"""Branch-and-Bound exato para o TSP simÃ©trico."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

from tp2.models import SolverResult, SolverStatus, TSPInstance, Tour
from tp2.tour_utils import tour_cost, validate_tour


def branch_and_bound(
    instance: TSPInstance,
    *,
    time_limit_seconds: float | None = None,
    initial_tour: Tour | Iterable[int] | None = None,
    mst_cache_limit: int = 4096,
) -> SolverResult:
    """Busca um ciclo Ã³timo, devolvendo o incumbent se o limite for atingido."""

    n = instance.dimension
    if n == 0:
        raise ValueError("instance must contain at least one vertex")
    if time_limit_seconds is not None and time_limit_seconds < 0:
        raise ValueError("time_limit_seconds must be nonnegative")
    if mst_cache_limit < 0:
        raise ValueError("mst_cache_limit must be nonnegative")

    started = time.monotonic()
    deadline = (
        None
        if time_limit_seconds is None
        else started + time_limit_seconds
    )
    counters: dict[str, int] = {
        "nodes_generated": 0,
        "nodes_expanded": 0,
        "bounds_pruned": 0,
        "dominance_pruned": 0,
        "symmetry_pruned": 0,
        "max_depth": 1,
        "incumbent_updates": 0,
        "bound_calls": 0,
        "bound_cache_hits": 0,
        "bound_cache_misses": 0,
        "timeouts": 0,
    }
    bound_elapsed = 0.0
    timed_out = False

    if initial_tour is None:
        incumbent_tour = Tour(range(n))
    else:
        incumbent_tour = initial_tour if isinstance(initial_tour, Tour) else Tour(initial_tour)
        validate_tour(instance, incumbent_tour)
    incumbent_cost = tour_cost(instance, incumbent_tour)

    if n == 1:
        elapsed = time.monotonic() - started
        return SolverResult(
            incumbent_tour,
            incumbent_cost,
            SolverStatus.OPTIMAL,
            timings={"total": elapsed, "bound": 0.0},
            counters=counters,
        )

    all_mask = (1 << n) - 1
    mst_cache: dict[int, int] = {}

    def mst_cost(mask: int) -> int:
        nonlocal bound_elapsed
        begin = time.perf_counter()
        counters["bound_calls"] += 1
        if mask in mst_cache:
            counters["bound_cache_hits"] += 1
            bound_elapsed += time.perf_counter() - begin
            return mst_cache[mask]
        counters["bound_cache_misses"] += 1
        vertices = [vertex for vertex in range(n) if mask & (1 << vertex)]
        if len(vertices) <= 1:
            result = 0
        else:
            selected = {vertices[0]}
            total = 0
            while len(selected) < len(vertices):
                best: tuple[int, int] | None = None
                for left in sorted(selected):
                    for right in vertices:
                        if right in selected:
                            continue
                        candidate = (int(instance.distance_matrix[left, right]), right)
                        if best is None or candidate < best:
                            best = candidate
                assert best is not None
                weight, vertex = best
                total += weight
                selected.add(vertex)
            result = total
        if len(mst_cache) < mst_cache_limit:
            mst_cache[mask] = result
        bound_elapsed += time.perf_counter() - begin
        return result

    def lower_bound(current: int, visited_mask: int, partial_cost: int) -> int:
        remaining_mask = all_mask ^ visited_mask
        if remaining_mask == 0:
            return partial_cost + int(instance.distance_matrix[current, 0])
        remaining = [vertex for vertex in range(1, n) if remaining_mask & (1 << vertex)]
        to_current = min(int(instance.distance_matrix[current, vertex]) for vertex in remaining)
        to_start = min(int(instance.distance_matrix[0, vertex]) for vertex in remaining)
        return partial_cost + mst_cost(remaining_mask) + to_current + to_start

    dominance: dict[tuple[int, int], int] = {}
    path = [0]
    visited_mask = 1

    def dfs(current: int, mask: int, partial_cost: int) -> None:
        nonlocal incumbent_cost, incumbent_tour, timed_out
        if timed_out:
            return
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            counters["timeouts"] = 1
            return
        counters["nodes_expanded"] += 1
        counters["max_depth"] = max(counters["max_depth"], len(path))
        state = (current, mask)
        previous = dominance.get(state)
        if previous is not None and partial_cost >= previous:
            counters["dominance_pruned"] += 1
            return
        dominance[state] = partial_cost
        if mask == all_mask:
            if n > 2 and path[1] > path[-1]:
                counters["symmetry_pruned"] += 1
                return
            total_cost = partial_cost + int(instance.distance_matrix[current, 0])
            if total_cost < incumbent_cost:
                incumbent_cost = total_cost
                incumbent_tour = Tour(tuple(path))
                counters["incumbent_updates"] += 1
            return
        if lower_bound(current, mask, partial_cost) >= incumbent_cost:
            counters["bounds_pruned"] += 1
            return
        candidates = [vertex for vertex in range(1, n) if not mask & (1 << vertex)]
        candidates.sort(key=lambda vertex: (int(instance.distance_matrix[current, vertex]), vertex))
        for vertex in candidates:
            counters["nodes_generated"] += 1
            path.append(vertex)
            dfs(
                vertex,
                mask | (1 << vertex),
                partial_cost + int(instance.distance_matrix[current, vertex]),
            )
            path.pop()
            if timed_out:
                return

    if deadline is not None and time.monotonic() >= deadline:
        timed_out = True
        counters["timeouts"] = 1
    else:
        dfs(0, visited_mask, 0)

    elapsed = time.monotonic() - started
    status = SolverStatus.TIMEOUT if timed_out else SolverStatus.OPTIMAL
    return SolverResult(
        incumbent_tour,
        incumbent_cost,
        status,
        timings={"total": elapsed, "bound": bound_elapsed},
        counters=counters,
    )
