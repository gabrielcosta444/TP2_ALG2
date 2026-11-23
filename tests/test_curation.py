from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest


EIL_LIKE_TEXT = """\
NAME: sample
TYPE: TSP
DIMENSION: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 1 0
3 2 0
4 3 0
5 4 0
EOF
"""


def test_fetch_source_decompresses_and_validates_tsplib(
    tmp_path: Path,
) -> None:
    from tp2.curation import fetch_source

    archive = tmp_path / "sample.tsp.gz"
    archive.write_bytes(gzip.compress(EIL_LIKE_TEXT.encode("utf-8")))
    destination = tmp_path / "raw" / "sample.tsp"

    record = fetch_source(archive.as_uri(), destination)

    assert destination.read_text(encoding="utf-8") == EIL_LIKE_TEXT
    assert record.name == "sample"
    assert record.dimension == 5
    assert len(record.sha256) == 64


def test_fetch_source_does_not_create_destination_for_invalid_payload(
    tmp_path: Path,
) -> None:
    from tp2.curation import fetch_source

    archive = tmp_path / "bad.tsp.gz"
    archive.write_bytes(gzip.compress(b"not a TSPLIB instance"))
    destination = tmp_path / "raw" / "bad.tsp"

    with pytest.raises(ValueError):
        fetch_source(archive.as_uri(), destination)

    assert not destination.exists()


def test_fetch_source_preserves_existing_file_when_identity_does_not_match(
    tmp_path: Path,
) -> None:
    from tp2.curation import fetch_source

    archive = tmp_path / "wrong.tsp.gz"
    archive.write_bytes(gzip.compress(EIL_LIKE_TEXT.encode("utf-8")))
    destination = tmp_path / "raw" / "eil51.tsp"
    destination.parent.mkdir()
    destination.write_text("known-good", encoding="utf-8")

    with pytest.raises(ValueError, match="dimension"):
        fetch_source(
            archive.as_uri(),
            destination,
            expected_name="sample",
            expected_dimension=51,
        )

    assert destination.read_text(encoding="utf-8") == "known-good"


def test_derive_subinstance_is_stable_and_records_original_ids() -> None:
    from tp2.curation import derive_subinstance, serialize_tsplib
    from tp2.tsplib import parse_tsplib

    source = parse_tsplib(EIL_LIKE_TEXT)
    first = derive_subinstance(source, "sample", size=3, seed=20_260_003)
    second = derive_subinstance(source, "sample", size=3, seed=20_260_003)

    assert first.source_name == "sample"
    assert first.source_vertex_ids == second.source_vertex_ids
    assert first.instance.dimension == 3
    assert first.instance.edge_weight_type == "EUC_2D"
    assert first.source_vertex_ids == tuple(sorted(first.source_vertex_ids))
    assert parse_tsplib(serialize_tsplib(first.instance)).dimension == 3


def test_derive_subinstance_rejects_invalid_size() -> None:
    from tp2.curation import derive_subinstance
    from tp2.tsplib import parse_tsplib

    source = parse_tsplib(EIL_LIKE_TEXT)

    with pytest.raises(ValueError, match="between 1 and 5"):
        derive_subinstance(source, "sample", size=6, seed=1)


def test_derive_subinstance_rejects_non_euc_2d_source() -> None:
    from tp2.curation import derive_subinstance
    from tp2.models import TSPInstance

    explicit = TSPInstance(
        name="explicit",
        coordinates=[[0.0, 0.0], [3.0, 4.0]],
        distance_matrix=[[0, 42], [42, 0]],
        edge_weight_type="EXPLICIT",
    )

    with pytest.raises(ValueError, match="EUC_2D"):
        derive_subinstance(explicit, "explicit", size=2, seed=1)


def test_serialize_tsplib_preserves_float64_coordinates() -> None:
    from tp2.curation import serialize_tsplib
    from tp2.models import TSPInstance
    from tp2.tsplib import parse_tsplib

    instance = TSPInstance(
        name="precision",
        coordinates=[[0.0, 0.0], [0.4999999999999, 0.0]],
        distance_matrix=[[0, 0], [0, 0]],
    )

    restored = parse_tsplib(serialize_tsplib(instance))

    assert restored.coordinates[1, 0] == instance.coordinates[1, 0]
    assert int(restored.distance_matrix[0, 1]) == 0


def test_validate_curated_provenance_accepts_project_dataset() -> None:
    from tp2.dataset import validate_curated_provenance

    project_root = Path(__file__).parents[1]

    summary = validate_curated_provenance(
        project_root / "data" / "manifests" / "instances.csv",
        project_root / "data" / "manifests" / "provenance.json",
        project_root / "data" / "raw",
        project_root / "data" / "raw" / "sources",
    )

    assert (summary.total, summary.main_range, summary.stress_range) == (12, 10, 2)

