"""Create reproducible, person-level dataset splits where identifiers exist."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse split-manifest arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Approved local dataset manifest.")
    parser.add_argument("--seed", type=int, required=True, help="Fixed random seed for reproducibility.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Ignored local split output directory.")
    return parser.parse_args()


def main() -> None:
    """Require a manifest before implementing split generation."""
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {args.manifest}")
    raise NotImplementedError("TODO: preserve person-level separation when subject information is available.")


if __name__ == "__main__":
    main()
