from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest


MANIFEST_FIELDS = (
    "filename",
    "name",
    "dimension",
    "edge_weight_type",
    "optimal_cost",
    "source_url",
    "sha256",
)


def _instance_text(name: str, dimension: int) -> str:
    coordinates = "\n".join(
        f"{vertex_id} {vertex_id - 1} 0"
        for vertex_id in range(1, dimension + 1)
    )
    return (
        f"NAME: {name}\n"
        "TYPE: TSP\n"
        f"DIMENSION: {dimension}\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\n"
        "NODE_COORD_SECTION\n"
        f"{coordinates}\n"
        "EOF\n"
    )


def _write_dataset(
    root: Path,
    dimensions: list[int],
    *,
    edge_weight_type: str = "EUC_2D",
) -> tuple[Path, Path]:
    raw_dir = root / "raw"
    raw_dir.mkdir()
    manifest_path = root / "instances.csv"
    rows: list[dict[str, str | int]] = []

    for index, dimension in enumerate(dimensions):
        name = f"case{index}"
        filename = f"{name}.tsp"
        content = _instance_text(name, dimension)
        (raw_dir / filename).write_bytes(content.encode("utf-8"))
        rows.append(
            {
                "filename": filename,
                "name": name,
                "dimension": dimension,
                "edge_weight_type": edge_weight_type,
                "optimal_cost": 2 * (dimension - 1),
                "source_url": f"https://example.test/{filename}",
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path, raw_dir


def _rewrite_manifest(manifest_path: Path, mutate) -> None:
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_dataset_accepts_required_size_distribution(tmp_path: Path) -> None:
    from tp2.dataset import validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )

    summary = validate_dataset(manifest_path, raw_dir)

    assert summary.total == 12
    assert summary.main_range == 10
    assert summary.stress_range == 2


def test_validate_dataset_rejects_less_than_eighty_percent_main_range(
    tmp_path: Path,
) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 45, 46, 47],
    )

    with pytest.raises(DatasetValidationError, match="80%"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_rejects_checksum_mismatch(tmp_path: Path) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    first_file = raw_dir / "case0.tsp"
    first_file.write_text(first_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="checksum"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_rejects_manifest_metadata_mismatch(
    tmp_path: Path,
) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
        edge_weight_type="GEO",
    )

    with pytest.raises(DatasetValidationError, match="EUC_2D"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_accepts_na_optimum_for_derived_instances(
    tmp_path: Path,
) -> None:
    from tp2.dataset import validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    _rewrite_manifest(
        manifest_path,
        lambda rows: rows[0].update(optimal_cost="NA"),
    )

    assert validate_dataset(manifest_path, raw_dir).total == 12


def test_validate_dataset_rejects_duplicate_files(tmp_path: Path) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    _rewrite_manifest(
        manifest_path,
        lambda rows: rows[-1].update(
            filename=rows[0]["filename"],
            sha256=rows[0]["sha256"],
        ),
    )

    with pytest.raises(DatasetValidationError, match="duplicate.*filename"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_requires_source_url(tmp_path: Path) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    _rewrite_manifest(
        manifest_path,
        lambda rows: rows[0].update(source_url=""),
    )

    with pytest.raises(DatasetValidationError, match="source"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_rejects_paths_outside_raw_directory(
    tmp_path: Path,
) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    _rewrite_manifest(
        manifest_path,
        lambda rows: rows[0].update(filename="../outside.tsp"),
    )

    with pytest.raises(DatasetValidationError, match="inside.*raw"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_reports_truncated_csv_row(tmp_path: Path) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path = tmp_path / "instances.csv"
    manifest_path.write_text(
        ",".join(MANIFEST_FIELDS) + "\ncase.tsp,case,10\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="line 2"):
        validate_dataset(manifest_path, tmp_path)


def test_validate_dataset_checks_header_dimension_before_coordinates(
    tmp_path: Path,
) -> None:
    from tp2.dataset import DatasetValidationError, validate_dataset

    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    first_file = raw_dir / "case0.tsp"
    changed = first_file.read_text(encoding="utf-8").replace(
        "DIMENSION: 10",
        "DIMENSION: 1000000000",
    )
    first_file.write_bytes(changed.encode("utf-8"))
    changed_checksum = sha256(changed.encode("utf-8")).hexdigest()
    _rewrite_manifest(
        manifest_path,
        lambda rows: rows[0].update(sha256=changed_checksum),
    )

    with pytest.raises(DatasetValidationError, match="dimension mismatch"):
        validate_dataset(manifest_path, raw_dir)


def test_validate_dataset_script_reports_valid_summary(tmp_path: Path) -> None:
    manifest_path, raw_dir = _write_dataset(
        tmp_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 70],
    )
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_dataset.py"),
            "--manifest",
            str(manifest_path),
            "--raw-dir",
            str(raw_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "12 instances" in completed.stdout
    assert "10 main" in completed.stdout
    assert "2 stress" in completed.stdout


def test_validate_dataset_script_returns_error_for_invalid_base(
    tmp_path: Path,
) -> None:
    manifest_path, raw_dir = _write_dataset(tmp_path, [10])
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_dataset.py"),
            "--manifest",
            str(manifest_path),
            "--raw-dir",
            str(raw_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Dataset invalid:" in completed.stderr
