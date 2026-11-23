from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


VALID_INSTANCE = """\
COMMENT: headers may be unordered
DIMENSION : 3
EDGE_WEIGHT_TYPE: EUC_2D
TYPE : TSP
NAME: tiny
NODE_COORD_SECTION
3 -3 4
1 0 0
2 0 2.5
EOF
"""


def test_parse_tsplib_sorts_ids_and_uses_tsplib_rounding() -> None:
    from tp2.tsplib import parse_tsplib

    instance = parse_tsplib(VALID_INSTANCE)

    assert instance.name == "tiny"
    assert instance.dimension == 3
    assert instance.edge_weight_type == "EUC_2D"
    np.testing.assert_array_equal(
        instance.coordinates,
        np.array([[0.0, 0.0], [0.0, 2.5], [-3.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        instance.distance_matrix,
        np.array([[0, 3, 5], [3, 0, 3], [5, 3, 0]]),
    )


def test_euc_2d_rounding_does_not_change_large_integer_distance() -> None:
    from tp2.tsplib import euc_2d_distance_matrix

    large_integer = 2**52 + 1
    distances = euc_2d_distance_matrix(
        np.array([[0.0, 0.0], [float(large_integer), 0.0]])
    )

    assert int(distances[0, 1]) == large_integer


def test_load_tsplib_reads_utf8_bom_file(tmp_path: Path) -> None:
    from tp2.tsplib import load_tsplib

    path = tmp_path / "tiny.tsp"
    path.write_text("\ufeff" + VALID_INSTANCE, encoding="utf-8")

    assert load_tsplib(path).name == "tiny"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("TYPE : TSP", "TYPE : ATSP", "TYPE.*TSP"),
        ("EDGE_WEIGHT_TYPE: EUC_2D", "EDGE_WEIGHT_TYPE: GEO", "EUC_2D"),
        ("DIMENSION : 3", "DIMENSION : zero", "DIMENSION"),
    ],
)
def test_parse_tsplib_rejects_unsupported_or_invalid_headers(
    old: str,
    new: str,
    message: str,
) -> None:
    from tp2.tsplib import TSPLibParseError, parse_tsplib

    with pytest.raises(TSPLibParseError, match=message):
        parse_tsplib(VALID_INSTANCE.replace(old, new))


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        ("1 0 0\n2 1 1\nEOF", "expected 3"),
        ("1 0 0\n1 1 1\n3 2 2\nEOF", "duplicate"),
        ("1 0 0\n2 1 1\n4 2 2\nEOF", "IDs"),
        ("1 0 0\n2 one 1\n3 2 2\nEOF", "coordinate"),
    ],
)
def test_parse_tsplib_rejects_invalid_coordinate_sections(
    coordinates: str,
    message: str,
) -> None:
    from tp2.tsplib import TSPLibParseError, parse_tsplib

    malformed = VALID_INSTANCE.split("NODE_COORD_SECTION", 1)[0]
    malformed += "NODE_COORD_SECTION\n" + coordinates

    with pytest.raises(TSPLibParseError, match=message):
        parse_tsplib(malformed)


def test_parse_tsplib_requires_all_mandatory_headers() -> None:
    from tp2.tsplib import TSPLibParseError, parse_tsplib

    with pytest.raises(TSPLibParseError, match="NAME"):
        parse_tsplib(VALID_INSTANCE.replace("NAME: tiny\n", ""))


def test_parse_tsplib_rejects_required_fixed_edges() -> None:
    from tp2.tsplib import TSPLibParseError, parse_tsplib

    with pytest.raises(TSPLibParseError, match="FIXED_EDGES_SECTION"):
        parse_tsplib(
            VALID_INSTANCE.replace(
                "EOF",
                "FIXED_EDGES_SECTION\n1 2\n-1\nEOF",
            )
        )

