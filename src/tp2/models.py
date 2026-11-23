"""Modelos compartilhados pelos algoritmos do TSP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from operator import index
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SolverStatus(StrEnum):
    """Estados possíveis de uma execução de algoritmo."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    ERROR = "error"


def _immutable_array(array: NDArray) -> NDArray:
    """Copia um array para um buffer de bytes que não pode ser reaberto para escrita."""

    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True, init=False)
class TSPInstance:
    """Instância completa do TSP com dados defensivamente copiados."""

    name: str
    coordinates: NDArray[np.float64]
    distance_matrix: NDArray[np.int64]
    edge_weight_type: str

    def __init__(
        self,
        name: str,
        coordinates: ArrayLike,
        distance_matrix: ArrayLike,
        edge_weight_type: str = "EUC_2D",
    ) -> None:
        coordinates_array = np.array(coordinates, dtype=np.float64, copy=True)
        distances_input = np.array(distance_matrix, copy=True)

        if coordinates_array.ndim != 2 or coordinates_array.shape[1:] != (2,):
            raise ValueError("coordinates must have dimensions (n, 2)")

        dimension = coordinates_array.shape[0]
        if distances_input.shape != (dimension, dimension):
            raise ValueError("distance matrix dimensions must be (n, n)")
        if not np.all(np.isfinite(coordinates_array)):
            raise ValueError("coordinates must contain only finite values")

        if np.issubdtype(distances_input.dtype, np.bool_):
            raise ValueError("distance matrix weights must be integers")
        if np.issubdtype(distances_input.dtype, np.integer):
            if (
                np.issubdtype(distances_input.dtype, np.unsignedinteger)
                and np.any(distances_input > np.iinfo(np.int64).max)
            ):
                raise ValueError("distance matrix weights exceed int64 range")
            distances_array = distances_input.astype(np.int64, copy=True)
        elif np.issubdtype(distances_input.dtype, np.floating):
            if not np.all(np.isfinite(distances_input)):
                raise ValueError("distance matrix must contain only finite values")
            if np.any(distances_input < 0):
                raise ValueError("distance matrix cannot contain negative values")
            if np.any(distances_input != np.trunc(distances_input)):
                raise ValueError("distance matrix weights must be integers")
            if np.any(distances_input > np.iinfo(np.int64).max):
                raise ValueError("distance matrix weights exceed int64 range")
            distances_array = distances_input.astype(np.int64)
        else:
            raise ValueError("distance matrix weights must be numeric integers")

        if np.any(distances_array < 0):
            raise ValueError("distance matrix cannot contain negative values")
        if not np.array_equal(distances_array, distances_array.T):
            raise ValueError("distance matrix must be symmetric")
        if np.any(np.diag(distances_array) != 0):
            raise ValueError("distance matrix diagonal must contain only zeros")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinates", _immutable_array(coordinates_array))
        object.__setattr__(self, "distance_matrix", _immutable_array(distances_array))
        object.__setattr__(self, "edge_weight_type", edge_weight_type)

    @property
    def dimension(self) -> int:
        """Número de vértices da instância."""

        return int(self.coordinates.shape[0])


@dataclass(frozen=True, slots=True, init=False)
class Tour:
    """Ciclo representado sem repetir o vértice inicial ao final."""

    vertices: tuple[int, ...]

    def __init__(self, vertices: Iterable[int]) -> None:
        normalized: list[int] = []
        for vertex in vertices:
            if isinstance(vertex, (bool, np.bool_)):
                raise ValueError("tour vertices must be integer indices")
            try:
                normalized.append(index(vertex))
            except TypeError as error:
                raise ValueError("tour vertices must be integer indices") from error

        normalized_vertices = tuple(normalized)
        if (
            len(normalized_vertices) > 1
            and normalized_vertices[0] == normalized_vertices[-1]
        ):
            raise ValueError("tour must not repeat the initial vertex at the end")
        object.__setattr__(self, "vertices", normalized_vertices)


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Resultado uniforme devolvido por qualquer solucionador."""

    tour: Tour | None
    cost: int | None
    status: SolverStatus
    timings: Mapping[str, float] = field(default_factory=dict)
    counters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "counters", MappingProxyType(dict(self.counters)))

