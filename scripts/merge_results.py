"""Une CSVs de partições experimentais e valida chaves únicas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tp2.experiment import merge_result_csvs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = merge_result_csvs(args.inputs, args.output)
    print(f"Merged {len(rows)} experiment rows into {args.output}")


if __name__ == "__main__":
    main()
