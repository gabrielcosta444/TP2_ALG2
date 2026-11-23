"""Leitura de instâncias TSPLIB do tipo TSP/EUC_2D."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tp2.models import TSPInstance


class TSPLibParseError(ValueError):
    """Indica que uma instância não segue o subconjunto TSPLIB suportado."""


@dataclass(frozen=True, slots=True)
class TSPLibMetadata:
    name: str
    dimension: int
    edge_weight_type: str


def euc_2d_distance_matrix(
    coordinates: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Calcula pesos EUC_2D usando o arredondamento ``nint`` da TSPLIB."""

    deltas = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    euclidean = np.sqrt(np.sum(deltas * deltas, axis=2))
    if not np.all(np.isfinite(euclidean)):
        raise TSPLibParseError("EUC_2D distances must be finite")
    whole = np.floor(euclidean)
    rounded = whole + ((euclidean - whole) >= 0.5)
    if np.any(rounded > np.iinfo(np.int64).max):
        raise TSPLibParseError("EUC_2D distance exceeds int64 range")
    distances = rounded.astype(np.int64)
    np.fill_diagonal(distances, 0)
    return distances


def _header_fields(lines: list[str]) -> tuple[dict[str, str], int]:
    headers: dict[str, str] = {}
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() == "NODE_COORD_SECTION":
            return headers, index + 1
        if line.upper() == "EOF" or line.upper().endswith("_SECTION"):
            break

        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts
        headers[key.strip().upper()] = value.strip()

    raise TSPLibParseError("missing NODE_COORD_SECTION")


def _validated_headers(headers: dict[str, str]) -> tuple[str, int]:
    for required in ("NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE"):
        if not headers.get(required):
            raise TSPLibParseError(f"missing mandatory header {required}")

    if headers["TYPE"].upper() != "TSP":
        raise TSPLibParseError("TYPE must be TSP")
    if headers["EDGE_WEIGHT_TYPE"].upper() != "EUC_2D":
        raise TSPLibParseError("only EDGE_WEIGHT_TYPE EUC_2D is supported")

    try:
        dimension = int(headers["DIMENSION"])
    except ValueError as error:
        raise TSPLibParseError("DIMENSION must be a positive integer") from error
    if dimension <= 0:
        raise TSPLibParseError("DIMENSION must be a positive integer")
    return headers["NAME"], dimension


def _coordinates(
    lines: list[str],
    start: int,
    dimension: int,
) -> NDArray[np.float64]:
    by_id: dict[int, tuple[float, float]] = {}

    for raw_line in lines[start:]:
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF" or upper.endswith("_SECTION"):
            break

        parts = line.split()
        if len(parts) != 3:
            raise TSPLibParseError("each coordinate row must contain ID, x and y")
        try:
            node_id = int(parts[0])
            x_coordinate = float(parts[1])
            y_coordinate = float(parts[2])
        except ValueError as error:
            raise TSPLibParseError("invalid node ID or coordinate value") from error
        if node_id in by_id:
            raise TSPLibParseError(f"duplicate node ID {node_id}")
        by_id[node_id] = (x_coordinate, y_coordinate)

    if len(by_id) != dimension:
        raise TSPLibParseError(
            f"expected {dimension} coordinate rows, found {len(by_id)}"
        )
    expected_ids = set(range(1, dimension + 1))
    if set(by_id) != expected_ids:
        raise TSPLibParseError("node IDs must be exactly the integers from 1 to DIMENSION")

    return np.array([by_id[node_id] for node_id in range(1, dimension + 1)])


def parse_tsplib(text: str) -> TSPInstance:
    """Interpreta o subconjunto TSP/EUC_2D de um arquivo TSPLIB."""

    lines = text.lstrip("\ufeff").splitlines()
    if any(line.strip().upper() == "FIXED_EDGES_SECTION" for line in lines):
        raise TSPLibParseError("FIXED_EDGES_SECTION is not supported")
    headers, coordinate_start = _header_fields(lines)
    name, dimension = _validated_headers(headers)
    coordinates = _coordinates(lines, coordinate_start, dimension)
    distances = euc_2d_distance_matrix(coordinates)

    try:
        return TSPInstance(
            name=name,
            coordinates=coordinates,
            distance_matrix=distances,
            edge_weight_type="EUC_2D",
        )
    except ValueError as error:
        raise TSPLibParseError(str(error)) from error


def load_tsplib(path: str | PathLike[str]) -> TSPInstance:
    """Lê uma instância TSPLIB UTF-8/ASCII do disco."""

    return parse_tsplib(Path(path).read_text(encoding="utf-8-sig"))


def parse_tsplib_metadata(text: str) -> TSPLibMetadata:
    """Lê apenas os metadados necessários, sem construir a matriz de custos."""

    lines = text.lstrip("\ufeff").splitlines()
    headers, _ = _header_fields(lines)
    name, dimension = _validated_headers(headers)
    return TSPLibMetadata(name, dimension, "EUC_2D")


def load_tsplib_metadata(path: str | PathLike[str]) -> TSPLibMetadata:
    """Lê apenas o cabeçalho de uma instância no disco."""

    return parse_tsplib_metadata(Path(path).read_text(encoding="utf-8-sig"))

