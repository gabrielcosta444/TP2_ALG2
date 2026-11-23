"""Valida a base TSPLIB antes de iniciar os experimentos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tp2.dataset import (  # noqa: E402
    DatasetValidationError,
    validate_curated_provenance,
    validate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "instances.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    provenance_path = arguments.provenance or arguments.manifest.with_name(
        "provenance.json"
    )
    sources_dir = arguments.sources_dir or arguments.raw_dir / "sources"

    try:
        if provenance_path.exists():
            summary = validate_curated_provenance(
                arguments.manifest,
                provenance_path,
                arguments.raw_dir,
                sources_dir,
            )
        else:
            summary = validate_dataset(arguments.manifest, arguments.raw_dir)
    except (DatasetValidationError, OSError) as error:
        print(f"Dataset invalid: {error}", file=sys.stderr)
        return 1

    print(
        f"Dataset valid: {summary.total} instances, "
        f"{summary.main_range} main, {summary.stress_range} stress"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

