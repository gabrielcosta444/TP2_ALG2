"""ExecuÃ§Ã£o isolada e persistente da bateria experimental."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
from queue import Empty
import time
import tracemalloc
from typing import Any

from tp2.branch_and_bound import branch_and_bound
from tp2.christofides import christofides
from tp2.features import extract_features
from tp2.models import SolverStatus, TSPInstance
from tp2.tsplib import load_tsplib
from tp2.twice_around import twice_around_tree
from tp2.warm_starts import (
    convex_hull_cheapest_insertion,
    nearest_neighbor_multistart,
    trivial_tour,
)

WARM_START_FUNCTIONS = {
    "trivial": trivial_tour,
    "nearest_neighbor_multistart": nearest_neighbor_multistart,
    "convex_hull_cheapest_insertion": convex_hull_cheapest_insertion,
    "twice_around_the_tree": twice_around_tree,
    "christofides": christofides,
}
APPROXIMATION_FUNCTIONS = {
    "twice_around_the_tree": twice_around_tree,
    "christofides": christofides,
}
CSV_FIELDS = (
    "instance", "N", "algorithm", "warm_start", "status", "cost", "optimal",
    "gap", "total_seconds", "warm_start_seconds", "solver_seconds",
    "phase_timings", "peak_memory_bytes", "counters", "seed", "timestamp_utc",
    "python_version", "platform",
)


@dataclass(frozen=True, slots=True)
class ManifestJob:
    instance: str
    path: Path
    dimension: int
    optimal: int | None


def load_manifest_jobs(manifest_path: str | Path, raw_dir: str | Path) -> tuple[ManifestJob, ...]:
    jobs: list[ManifestJob] = []
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            optimal = None if row["optimal_cost"].upper() == "NA" else int(row["optimal_cost"])
            jobs.append(ManifestJob(row["name"], Path(raw_dir) / row["filename"], int(row["dimension"]), optimal))
    return tuple(jobs)


def _empty_counters() -> dict[str, int]:
    return {}


def _row_for_result(
    instance: TSPInstance,
    algorithm: str,
    warm_start: str,
    result: Any,
    *,
    warm_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
    seed: int = 0,
) -> dict[str, str]:
    cost = getattr(result, "cost", None)
    optimal = None
    gap = None
    status = result.status.value if hasattr(result, "status") else SolverStatus.FEASIBLE.value
    if cost is not None and optimal is not None:
        gap = float(cost) / optimal - 1.0
    timings = getattr(result, "timings", {})
    total = float(getattr(result, "elapsed_seconds", timings.get("total", 0.0)))
    if hasattr(result, "phase_timings"):
        phase_timings = dict(result.phase_timings)
        total = float(result.elapsed_seconds)
    else:
        phase_timings = dict(timings)
    return {
        "instance": instance.name,
        "N": str(instance.dimension),
        "algorithm": algorithm,
        "warm_start": warm_start,
        "status": status,
        "cost": "NA" if cost is None else str(cost),
        "optimal": "NA" if optimal is None else str(optimal),
        "gap": "NA" if gap is None else f"{gap:.12g}",
        "total_seconds": f"{total:.12g}",
        "warm_start_seconds": f"{warm_seconds:.12g}",
        "solver_seconds": f"{max(0.0, total - warm_seconds):.12g}",
        "phase_timings": json.dumps(phase_timings, sort_keys=True),
        "peak_memory_bytes": str(peak_memory_bytes),
        "counters": json.dumps(dict(getattr(result, "counters", _empty_counters())), sort_keys=True),
        "seed": str(seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def run_job(
    job: ManifestJob,
    algorithm: str,
    warm_start: str = "none",
    *,
    timeout_seconds: float | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Executa uma combinaÃ§Ã£o no processo atual e devolve uma linha tabular."""

    instance = load_tsplib(job.path)
    features = extract_features(instance)
    started = time.perf_counter()
    tracemalloc.start()
    try:
        if algorithm in APPROXIMATION_FUNCTIONS:
            result = APPROXIMATION_FUNCTIONS[algorithm](instance)
            result_warm_seconds = 0.0
            status = SolverStatus.FEASIBLE
        elif algorithm == "branch_and_bound":
            if warm_start not in WARM_START_FUNCTIONS:
                raise ValueError(f"unknown warm-start: {warm_start}")
            warm = WARM_START_FUNCTIONS[warm_start](instance)
            result = branch_and_bound(
                instance,
                time_limit_seconds=timeout_seconds,
                initial_tour=warm.tour,
            )
            result_warm_seconds = warm.elapsed_seconds
            status = result.status
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if algorithm in APPROXIMATION_FUNCTIONS:
        row = _row_for_result(
            instance,
            algorithm,
            warm_start,
            result,
            warm_seconds=result_warm_seconds,
            peak_memory_bytes=peak,
            seed=seed,
        )
        row["status"] = status.value
    else:
        row = _row_for_result(
            instance,
            algorithm,
            warm_start,
            result,
            warm_seconds=result_warm_seconds,
            peak_memory_bytes=peak,
            seed=seed,
        )
    row["total_seconds"] = f"{time.perf_counter() - started:.12g}"
    row["solver_seconds"] = f"{max(0.0, float(row['total_seconds']) - float(row['warm_start_seconds'])):.12g}"
    row.update({f"feature_{key}": f"{value:.12g}" for key, value in features.items()})
    return row


def _isolated_worker(queue: Any, job: ManifestJob, algorithm: str, warm_start: str, timeout: float | None, seed: int) -> None:
    try:
        queue.put(run_job(job, algorithm, warm_start, timeout_seconds=timeout, seed=seed))
    except Exception as error:  # pragma: no cover - safety boundary for child process
        queue.put({"error": f"{type(error).__name__}: {error}"})


def _timeout_row(job: ManifestJob, algorithm: str, warm_start: str, seed: int) -> dict[str, str]:
    instance = load_tsplib(job.path)
    row = {
        "instance": instance.name,
        "N": str(instance.dimension),
        "algorithm": algorithm,
        "warm_start": warm_start,
        "status": "timeout",
        "cost": "NA",
        "optimal": "NA",
        "gap": "NA",
        "total_seconds": "NA",
        "warm_start_seconds": "NA",
        "solver_seconds": "NA",
        "phase_timings": "{}",
        "peak_memory_bytes": "NA",
        "counters": "{}",
        "seed": str(seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    row.update({f"feature_{key}": f"{value:.12g}" for key, value in extract_features(instance).items()})
    return row


def run_isolated_job(job: ManifestJob, algorithm: str, warm_start: str = "none", *, timeout_seconds: float = 1800.0, seed: int = 0) -> dict[str, str]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_isolated_worker, args=(queue, job, algorithm, warm_start, timeout_seconds, seed))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return _timeout_row(job, algorithm, warm_start, seed)
    try:
        result = queue.get(timeout=2.0)
    except Empty as error:
        raise RuntimeError("isolated experiment exited without a result") from error
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def atomic_write_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CSV_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def merge_result_csvs(inputs: list[str | Path], output: str | Path) -> list[dict[str, str]]:
    """Une partições e rejeita chaves experimentais duplicadas."""

    rows: list[dict[str, str]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for input_path in inputs:
        with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row.get("instance"), row.get("algorithm"), row.get("warm_start"))
                if key in seen:
                    raise ValueError(f"duplicate experiment key: {key}")
                seen.add(key)
                rows.append(row)
    atomic_write_csv(output, rows)
    return rows


def run_batch(
    manifest_path: str | Path,
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    mode: str = "smoke",
    timeout_seconds: float = 1800.0,
    seed: int = 0,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> list[dict[str, str]]:
    """Executa smoke, piloto (3 instÃ¢ncias) ou campanha completa com retomada."""

    jobs = list(load_manifest_jobs(manifest_path, raw_dir))
    if mode == "smoke":
        jobs = jobs[:1]
    elif mode == "pilot":
        jobs = jobs[:3]
    elif mode != "full":
        raise ValueError("mode must be smoke, pilot or full")
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count is not None:
        if shard_count < 1 or shard_index is None or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard specification")
        jobs = [job for index, job in enumerate(jobs) if index % shard_count == shard_index]
    algorithms = (("twice_around_the_tree", "none"), ("christofides", "none"))
    algorithms += tuple(("branch_and_bound", warm) for warm in WARM_START_FUNCTIONS)
    existing: list[dict[str, str]] = []
    destination = Path(output_path)
    if destination.is_file():
        with destination.open("r", encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
    completed = {
        (row.get("instance"), row.get("algorithm"), row.get("warm_start"))
        for row in existing
        if row.get("status") in {"optimal", "feasible", "timeout"}
    }
    rows = list(existing)
    for job in jobs:
        for algorithm, warm_start in algorithms:
            key = (job.instance, algorithm, warm_start)
            if key in completed:
                continue
            row = run_isolated_job(job, algorithm, warm_start, timeout_seconds=timeout_seconds, seed=seed)
            if row.get("optimal") == "NA" and job.optimal is not None:
                row["optimal"] = str(job.optimal)
                if row.get("cost") not in {None, "NA"}:
                    row["gap"] = f"{float(row['cost']) / job.optimal - 1.0:.12g}"
            rows.append(row)
            atomic_write_csv(destination, rows)
    atomic_write_csv(destination, rows)
    return rows
