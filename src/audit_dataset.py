"""Audit an authorised local dataset before any conversion or training."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a dataset audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Authorised local dataset directory.")
    parser.add_argument("--report", type=Path, required=True, help="Local path for the audit report.")
    return parser.parse_args()


def main() -> None:
    """Validate inputs and reserve the audit implementation for approved data."""
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_dir}")
    raise NotImplementedError("TODO: confirm dataset licence and card, then implement a deterministic audit.")


if __name__ == "__main__":
    main()
