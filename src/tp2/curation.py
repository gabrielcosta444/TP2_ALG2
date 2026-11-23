"""Aquisição e derivação reproduzível de instâncias TSP/EUC_2D."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np

from tp2.models import TSPInstance
from tp2.tsplib import euc_2d_distance_matrix, parse_tsplib


@dataclass(frozen=True, slots=True)
class OfficialSource:
    name: str
    url: str
    dimension: int
    sha256: str


OFFICIAL_SOURCES = {
    "eil51": OfficialSource(
        "eil51",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/eil51.tsp.gz",
        51,
        "66c05ad077ffd64d5778e358bbd16be049f8c8ae880cbd24f034487331d1ccce",
    ),
    "berlin52": OfficialSource(
        "berlin52",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp.gz",
        52,
        "8496a5838133e3c3eb25a9fa8893b1ac302593fc8dc325350643a0cabd8abadc",
    ),
    "st70": OfficialSource(
        "st70",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/st70.tsp.gz",
        70,
        "60c09e8886609bf22d6dc699fe7d7aa1924207d084578aad1a377ec65cf15b1b",
    ),
}

@dataclass(frozen=True, slots=True)
class SourceRecord:
    name: str
    dimension: int
    source_url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DerivedInstance:
    instance: TSPInstance
    source_name: str
    seed: int
    source_vertex_ids: tuple[int, ...]


def validate_official_source(path: str | Path, source: OfficialSource) -> TSPInstance:
    """Confere identidade do arquivo local antes de usá-lo como fonte."""

    payload = Path(path).read_bytes()
    if sha256(payload).hexdigest() != source.sha256:
        raise ValueError(f"{source.name}: source checksum mismatch")
    instance = parse_tsplib(payload.decode("utf-8-sig"))
    if instance.name.lower() != source.name or instance.dimension != source.dimension:
        raise ValueError(f"{source.name}: source metadata mismatch")
    return instance


def serialize_tsplib(instance: TSPInstance) -> str:
    """Serializa uma instância TSP/EUC_2D com IDs sequenciais de 1 a N."""

    lines = [
        f"NAME: {instance.name}",
        "TYPE: TSP",
        f"DIMENSION: {instance.dimension}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]
    lines.extend(
        f"{index} {x:.17g} {y:.17g}"
        for index, (x, y) in enumerate(instance.coordinates, start=1)
    )
    lines.append("EOF")
    return "\n".join(lines) + "\n"


def fetch_source(
    url: str,
    destination: str | Path,
    *,
    expected_name: str | None = None,
    expected_dimension: int | None = None,
    expected_sha256: str | None = None,
) -> SourceRecord:
    """Baixa um arquivo gzip, valida TSPLIB e só então o publica no destino."""

    destination_path = Path(destination)
    with urlopen(url) as response:
        compressed = response.read()
    payload = gzip.decompress(compressed)
    text = payload.decode("utf-8-sig")
    instance = parse_tsplib(text)
    actual_sha256 = sha256(payload).hexdigest()
    if expected_name is not None and instance.name.lower() != expected_name.lower():
        raise ValueError(
            f"source name mismatch: expected {expected_name}, got {instance.name}"
        )
    if expected_dimension is not None and instance.dimension != expected_dimension:
        raise ValueError(
            "source dimension mismatch: "
            f"expected {expected_dimension}, got {instance.dimension}"
        )
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ValueError("source checksum mismatch")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination_path)

    return SourceRecord(
        name=instance.name,
        dimension=instance.dimension,
        source_url=url,
        sha256=actual_sha256,
    )


def derive_subinstance(
    source: TSPInstance,
    source_name: str,
    *,
    size: int,
    seed: int,
) -> DerivedInstance:
    """Seleciona uma subinstância induzida, determinística e sem reposição."""

    if not 1 <= size <= source.dimension:
        raise ValueError(f"size must be between 1 and {source.dimension}")
    if source.edge_weight_type != "EUC_2D":
        raise ValueError("subinstances can only be derived from EUC_2D sources")

    generator = np.random.default_rng(seed)
    selected = np.sort(generator.choice(source.dimension, size=size, replace=False))
    instance = TSPInstance(
        name=f"{source_name}_n{size}",
        coordinates=source.coordinates[selected],
        distance_matrix=euc_2d_distance_matrix(source.coordinates[selected]),
        edge_weight_type="EUC_2D",
    )
    return DerivedInstance(
        instance=instance,
        source_name=source_name,
        seed=seed,
        source_vertex_ids=tuple(int(index) + 1 for index in selected),
    )
