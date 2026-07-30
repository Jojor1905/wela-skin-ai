"""Convert approved source annotations to the documented YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse source and destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Authorised source annotation path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Ignored local YOLO label directory.")
    return parser.parse_args()


def main() -> None:
    """Validate source availability before approved conversion work."""
    args = parse_args()
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotation source does not exist: {args.annotations}")
    raise NotImplementedError("TODO: implement only after LABEL_GUIDE.md and data documentation are approved.")


if __name__ == "__main__":
    main()
