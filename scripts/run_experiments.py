"""Executa a bateria experimental em modo smoke, pilot ou full."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tp2.experiment import run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/instances.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/results/experiments.csv"))
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    args = parser.parse_args()
    rows = run_batch(
        args.manifest,
        args.raw_dir,
        args.output,
        mode=args.mode,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(f"Stored {len(rows)} experiment rows in {args.output}")


if __name__ == "__main__":
    main()
