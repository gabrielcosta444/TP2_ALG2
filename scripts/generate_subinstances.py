"""Gera a seleção experimental EUC_2D e seus arquivos de proveniência."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tp2.curation import (  # noqa: E402
    OFFICIAL_SOURCES,
    derive_subinstance,
    serialize_tsplib,
    validate_official_source,
)


SUBINSTANCE_SIZES = (10, 12, 15, 18, 20, 25, 28, 30, 35, 40)
MANIFEST_FIELDS = (
    "filename",
    "name",
    "dimension",
    "edge_weight_type",
    "optimal_cost",
    "source_url",
    "sha256",
)


def _write_file(path: Path, text: str) -> str:
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "sources",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "instances.csv",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "provenance.json",
    )
    arguments = parser.parse_args()

    try:
        sources = {
            name: validate_official_source(
                arguments.sources_dir / f"{name}.tsp",
                source,
            )
            for name, source in OFFICIAL_SOURCES.items()
        }
    except Exception as error:
        print(f"Cannot read TSPLIB sources: {error}", file=sys.stderr)
        return 1

    arguments.raw_dir.mkdir(parents=True, exist_ok=True)
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.provenance.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    provenance: dict[str, object] = {"instances": []}

    for size in SUBINSTANCE_SIZES:
        seed = 20_260_000 + size
        derived = derive_subinstance(sources["eil51"], "eil51", size=size, seed=seed)
        filename = f"eil51_n{size}.tsp"
        checksum = _write_file(
            arguments.raw_dir / filename,
            serialize_tsplib(derived.instance),
        )
        rows.append(
            {
                "filename": filename,
                "name": derived.instance.name,
                "dimension": size,
                "edge_weight_type": "EUC_2D",
                "optimal_cost": "NA",
                "source_url": OFFICIAL_SOURCES["eil51"].url,
                "sha256": checksum,
            }
        )
        provenance["instances"].append(
            {
                "filename": filename,
                "kind": "derived",
                "source_name": "eil51",
                "source_url": OFFICIAL_SOURCES["eil51"].url,
                "source_sha256": sha256(
                    (arguments.sources_dir / "eil51.tsp").read_bytes()
                ).hexdigest(),
                "seed": seed,
                "source_vertex_ids": list(derived.source_vertex_ids),
                "sha256": checksum,
            }
        )

    for name, optimum in (("berlin52", 7542), ("st70", 675)):
        filename = f"{name}.tsp"
        source_file = arguments.sources_dir / filename
        destination = arguments.raw_dir / filename
        payload = source_file.read_bytes()
        destination.write_bytes(payload)
        checksum = sha256(payload).hexdigest()
        instance = sources[name]
        rows.append(
            {
                "filename": filename,
                "name": instance.name,
                "dimension": instance.dimension,
                "edge_weight_type": instance.edge_weight_type,
                "optimal_cost": optimum,
                "source_url": OFFICIAL_SOURCES[name].url,
                "sha256": checksum,
            }
        )
        provenance["instances"].append(
            {
                "filename": filename,
                "kind": "official",
                "source_name": name,
                "source_url": OFFICIAL_SOURCES[name].url,
                "source_sha256": checksum,
                "sha256": checksum,
            }
        )

    with arguments.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    arguments.provenance.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} instances in {arguments.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
