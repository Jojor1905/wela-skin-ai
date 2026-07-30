"""Validate approved YOLO-format labels for the one-class MVP."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse label directory and reporting options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True, help="Ignored local YOLO label directory.")
    parser.add_argument("--report", type=Path, required=True, help="Local validation report path.")
    return parser.parse_args()


def main() -> None:
    """Check input existence before later validation implementation."""
    args = parse_args()
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {args.labels_dir}")
    raise NotImplementedError("TODO: validate class 0 and normalized bounding boxes using the approved label guide.")


if __name__ == "__main__":
    main()
