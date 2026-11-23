from __future__ import annotations

import csv
from pathlib import Path


def test_run_job_emits_features_and_metrics(tmp_path: Path) -> None:
    from tp2.experiment import ManifestJob, run_job
    from tp2.curation import serialize_tsplib
    from tp2.models import TSPInstance

    instance = TSPInstance(
        "tiny",
        [[0., 0.], [1., 0.], [1., 1.]],
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    )
    path = tmp_path / "tiny.tsp"
    path.write_text(serialize_tsplib(instance), encoding="utf-8")
    row = run_job(ManifestJob("tiny", path, 3, 3), "branch_and_bound", "trivial")

    assert row["status"] == "optimal"
    assert row["cost"] == "3"
    assert row["feature_dimension"] == "3"
    assert int(row["peak_memory_bytes"]) >= 0


def test_atomic_csv_replaces_and_resume_key_is_readable(tmp_path: Path) -> None:
    from tp2.experiment import atomic_write_csv

    path = tmp_path / "results.csv"
    rows = [{"instance": "a", "algorithm": "x", "warm_start": "none", "status": "feasible"}]
    atomic_write_csv(path, rows)
    atomic_write_csv(path, rows + [{"instance": "b", "algorithm": "x", "warm_start": "none", "status": "timeout"}])
    with path.open(encoding="utf-8", newline="") as stream:
        loaded = list(csv.DictReader(stream))
    assert [row["instance"] for row in loaded] == ["a", "b"]
    assert not path.with_name("results.csv.tmp").exists()


def test_merge_rejects_duplicate_experiment_keys(tmp_path: Path) -> None:
    from tp2.experiment import atomic_write_csv, merge_result_csvs

    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    row = {"instance": "a", "algorithm": "x", "warm_start": "none", "status": "feasible"}
    atomic_write_csv(first, [row])
    atomic_write_csv(second, [row])
    try:
        merge_result_csvs([first, second], tmp_path / "merged.csv")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate keys must be rejected")
