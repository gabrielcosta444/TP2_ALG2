"""Validação do manifesto e dos arquivos usados nos experimentos."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from tp2.curation import (
    OFFICIAL_SOURCES,
    derive_subinstance,
    serialize_tsplib,
    validate_official_source,
)
from tp2.tsplib import TSPLibParseError, load_tsplib, load_tsplib_metadata


class DatasetValidationError(ValueError):
    """Indica que a base experimental não satisfaz o plano do trabalho."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    filename: str
    name: str
    dimension: int
    edge_weight_type: str
    optimal_cost: int | None
    source_url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    total: int
    main_range: int
    stress_range: int


_REQUIRED_COLUMNS = {
    "filename",
    "name",
    "dimension",
    "edge_weight_type",
    "optimal_cost",
    "source_url",
    "sha256",
}


def _load_manifest(path: Path) -> list[ManifestEntry]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise DatasetValidationError(
                    f"manifest is missing columns: {', '.join(sorted(missing))}"
                )

            entries: list[ManifestEntry] = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    values = {
                        key: row.get(key).strip() if row.get(key) is not None else ""
                        for key in _REQUIRED_COLUMNS
                    }
                    empty_fields = [key for key, value in values.items() if not value]
                    if empty_fields:
                        raise DatasetValidationError(
                            f"missing {', '.join(sorted(empty_fields))} "
                            f"on line {line_number}"
                        )
                    entries.append(
                        ManifestEntry(
                            filename=values["filename"],
                            name=values["name"],
                            dimension=int(values["dimension"]),
                            edge_weight_type=values["edge_weight_type"].upper(),
                            optimal_cost=(
                                None
                                if values["optimal_cost"].upper() == "NA"
                                else int(values["optimal_cost"])
                            ),
                            source_url=values["source_url"],
                            sha256=values["sha256"].lower(),
                        )
                    )
                except DatasetValidationError:
                    raise
                except (TypeError, ValueError) as error:
                    raise DatasetValidationError(
                        f"invalid manifest values on line {line_number}"
                    ) from error
    except UnicodeError as error:
        raise DatasetValidationError(f"manifest is not valid UTF-8: {path}") from error
    return entries


def _validate_selection(entries: list[ManifestEntry]) -> DatasetSummary:
    total = len(entries)
    if not 12 <= total <= 16:
        raise DatasetValidationError("dataset must contain between 12 and 16 instances")

    names = [entry.name for entry in entries]
    filenames = [entry.filename for entry in entries]
    if len(set(names)) != total:
        raise DatasetValidationError("duplicate instance name in manifest")
    if len(set(filenames)) != total:
        raise DatasetValidationError("duplicate instance filename in manifest")

    for entry in entries:
        if entry.edge_weight_type != "EUC_2D":
            raise DatasetValidationError(
                f"{entry.name}: edge weight type must be EUC_2D"
            )
        if entry.optimal_cost is not None and entry.optimal_cost <= 0:
            raise DatasetValidationError(f"{entry.name}: optimal cost must be positive")
        if not (10 <= entry.dimension <= 40 or 45 <= entry.dimension <= 75):
            raise DatasetValidationError(
                f"{entry.name}: dimension must be in 10..40 or 45..75"
            )

    main_range = sum(entry.dimension <= 40 for entry in entries)
    stress_range = sum(entry.dimension >= 45 for entry in entries)
    if main_range / total < 0.8:
        raise DatasetValidationError("at least 80% of instances must have N <= 40")
    if stress_range == 0:
        raise DatasetValidationError("dataset must include at least one stress instance")
    return DatasetSummary(total, main_range, stress_range)


def validate_dataset(
    manifest_path: str | Path,
    raw_dir: str | Path,
) -> DatasetSummary:
    """Valida seleção, checksums e metadados de todas as instâncias."""

    entries = _load_manifest(Path(manifest_path))
    summary = _validate_selection(entries)
    raw_path = Path(raw_dir).resolve()
    resolved_files: set[Path] = set()

    for entry in entries:
        instance_path = (raw_path / entry.filename).resolve()
        if not instance_path.is_relative_to(raw_path):
            raise DatasetValidationError(
                f"instance files must stay inside the raw directory: {entry.filename}"
            )
        if instance_path in resolved_files:
            raise DatasetValidationError(
                f"duplicate resolved instance file: {entry.filename}"
            )
        resolved_files.add(instance_path)
        if not instance_path.is_file():
            raise DatasetValidationError(f"missing instance file: {entry.filename}")
        actual_checksum = sha256(instance_path.read_bytes()).hexdigest()
        if actual_checksum != entry.sha256:
            raise DatasetValidationError(f"checksum mismatch for {entry.filename}")
        try:
            metadata = load_tsplib_metadata(instance_path)
        except (TSPLibParseError, UnicodeError) as error:
            raise DatasetValidationError(f"{entry.filename}: {error}") from error
        if metadata.name != entry.name:
            raise DatasetValidationError(f"name mismatch for {entry.filename}")
        if metadata.dimension != entry.dimension:
            raise DatasetValidationError(f"dimension mismatch for {entry.filename}")
        if metadata.edge_weight_type != entry.edge_weight_type:
            raise DatasetValidationError(f"edge weight type mismatch for {entry.filename}")
        try:
            load_tsplib(instance_path)
        except (TSPLibParseError, UnicodeError) as error:
            raise DatasetValidationError(f"{entry.filename}: {error}") from error

    return summary


def validate_curated_provenance(
    manifest_path: str | Path,
    provenance_path: str | Path,
    raw_dir: str | Path,
    sources_dir: str | Path,
) -> DatasetSummary:
    """Confere a seleção aprovada e reproduz as subinstâncias a partir da proveniência."""

    manifest = Path(manifest_path)
    raw_path = Path(raw_dir)
    source_path = Path(sources_dir)
    summary = validate_dataset(manifest, raw_path)
    entries = {entry.filename: entry for entry in _load_manifest(manifest)}
    expected_sizes = (10, 12, 15, 18, 20, 25, 28, 30, 35, 40)
    expected_files = {f"eil51_n{size}.tsp" for size in expected_sizes}
    expected_files |= {"berlin52.tsp", "st70.tsp"}
    if set(entries) != expected_files:
        raise DatasetValidationError("manifest does not match the approved instance set")

    try:
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        records = provenance["instances"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DatasetValidationError("invalid curated provenance file") from error
    if not isinstance(records, list) or len(records) != len(expected_files):
        raise DatasetValidationError("provenance must contain exactly 12 instance records")
    by_filename = {record.get("filename"): record for record in records if isinstance(record, dict)}
    if set(by_filename) != expected_files or len(by_filename) != len(records):
        raise DatasetValidationError("provenance filenames do not match the manifest")

    sources = {
        name: validate_official_source(source_path / f"{name}.tsp", source)
        for name, source in OFFICIAL_SOURCES.items()
    }
    for size in expected_sizes:
        filename = f"eil51_n{size}.tsp"
        entry = entries[filename]
        record = by_filename[filename]
        seed = 20_260_000 + size
        derived = derive_subinstance(sources["eil51"], "eil51", size=size, seed=seed)
        expected_payload = serialize_tsplib(derived.instance).encode("utf-8")
        expected_checksum = sha256(expected_payload).hexdigest()
        if (
            record.get("kind") != "derived"
            or record.get("source_name") != "eil51"
            or record.get("source_url") != OFFICIAL_SOURCES["eil51"].url
            or record.get("source_sha256") != OFFICIAL_SOURCES["eil51"].sha256
            or record.get("seed") != seed
            or record.get("source_vertex_ids") != list(derived.source_vertex_ids)
            or record.get("sha256") != expected_checksum
            or entry.sha256 != expected_checksum
            or entry.optimal_cost is not None
        ):
            raise DatasetValidationError(f"invalid derived provenance for {filename}")

    for name, optimum in (("berlin52", 7542), ("st70", 675)):
        filename = f"{name}.tsp"
        entry = entries[filename]
        record = by_filename[filename]
        source = OFFICIAL_SOURCES[name]
        if (
            record.get("kind") != "official"
            or record.get("source_name") != name
            or record.get("source_url") != source.url
            or record.get("source_sha256") != source.sha256
            or record.get("sha256") != source.sha256
            or entry.sha256 != source.sha256
            or entry.optimal_cost != optimum
        ):
            raise DatasetValidationError(f"invalid official provenance for {filename}")
    return summary
