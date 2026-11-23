"""Operações sobre multigrafos não direcionados."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def eulerian_circuit(
    edges: Iterable[tuple[int, int]],
    *,
    start: int,
) -> tuple[int, ...]:
    """Retorna circuito Euleriano, preservando arestas paralelas."""

    normalized = [(int(left), int(right)) for left, right in edges]
    if not normalized:
        return (start,)
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    degree: dict[int, int] = defaultdict(int)
    for edge_id, (left, right) in enumerate(normalized):
        adjacency[left].append((right, edge_id))
        adjacency[right].append((left, edge_id))
        degree[left] += 1
        degree[right] += 1
    if start not in degree:
        raise ValueError("start vertex is not in the graph")
    if any(value % 2 for value in degree.values()):
        raise ValueError("all vertices must have even degree")

    reachable = {start}
    stack = [start]
    while stack:
        vertex = stack.pop()
        for neighbor, _ in adjacency[vertex]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                stack.append(neighbor)
    if any(vertex not in reachable for vertex, value in degree.items() if value > 0):
        raise ValueError("graph must be connected")

    used: set[int] = set()
    stack = [start]
    circuit: list[int] = []
    while stack:
        vertex = stack[-1]
        while adjacency[vertex] and adjacency[vertex][-1][1] in used:
            adjacency[vertex].pop()
        if not adjacency[vertex]:
            circuit.append(stack.pop())
            continue
        neighbor, edge_id = adjacency[vertex].pop()
        if edge_id in used:
            continue
        used.add(edge_id)
        stack.append(neighbor)
    circuit.reverse()
    if len(used) != len(normalized):
        raise ValueError("could not consume every edge")
    return tuple(circuit)
