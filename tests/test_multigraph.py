from __future__ import annotations

import pytest


def test_hierholzer_uses_every_parallel_edge_once() -> None:
    from tp2.multigraph import eulerian_circuit

    edges = ((0, 1), (0, 1), (1, 2), (1, 2), (0, 2), (0, 2))
    circuit = eulerian_circuit(edges, start=0)

    assert circuit[0] == circuit[-1] == 0
    assert len(circuit) == len(edges) + 1
    assert sorted(tuple(sorted((circuit[i], circuit[i + 1]))) for i in range(len(edges))) == sorted(edges)


def test_hierholzer_rejects_non_eulerian_degree_sequence() -> None:
    from tp2.multigraph import eulerian_circuit

    with pytest.raises(ValueError, match="even degree"):
        eulerian_circuit(((0, 1), (1, 2)), start=0)


def test_hierholzer_rejects_disconnected_edges() -> None:
    from tp2.multigraph import eulerian_circuit

    with pytest.raises(ValueError, match="connected"):
        eulerian_circuit(((0, 1), (0, 1), (2, 3), (2, 3)), start=0)
