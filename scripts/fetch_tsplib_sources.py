"""Baixa as três fontes oficiais usadas na curadoria EUC_2D."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tp2.curation import OFFICIAL_SOURCES, SourceRecord, fetch_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "sources",
    )
    arguments = parser.parse_args()

    records: list[SourceRecord] = []
    try:
        for source in OFFICIAL_SOURCES.values():
            record = fetch_source(
                source.url,
                arguments.output_dir / f"{source.name}.tsp",
                expected_name=source.name,
                expected_dimension=source.dimension,
                expected_sha256=source.sha256,
            )
            records.append(record)
    except Exception as error:
        print(f"Source acquisition failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps([asdict(record) for record in records], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

