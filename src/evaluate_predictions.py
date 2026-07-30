"""Evaluate authorised prediction outputs without making medical claims."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse prediction, ground-truth, and report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Ignored local prediction file.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Authorised local ground-truth file.")
    parser.add_argument("--report", type=Path, required=True, help="Local metrics report path.")
    return parser.parse_args()


def main() -> None:
    """Check required inputs before evaluation logic is approved and implemented."""
    args = parse_args()
    for path in (args.predictions, args.ground_truth):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input does not exist: {path}")
    raise NotImplementedError("TODO: define documented detection metrics and limitations before evaluation.")


if __name__ == "__main__":
    main()
