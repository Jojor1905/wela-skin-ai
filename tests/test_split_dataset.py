"""Duplicate-aware split tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from split_dataset import SplitItem, assign_splits  # noqa: E402


class DuplicateAwareSplitTests(unittest.TestCase):
    def test_duplicate_group_stays_together_and_seed_is_deterministic(self) -> None:
        items = [
            SplitItem("a", 10, 1, "hash-a"), SplitItem("b", 5, 0, "hash-a"),
            SplitItem("c", 8, 2), SplitItem("d", 3, 0), SplitItem("e", 4, 1),
            SplitItem("f", 2, 0), SplitItem("g", 1, 0),
        ]
        first = assign_splits(items, 0.70, 0.15, 0.15, 42)
        second = assign_splits(items, 0.70, 0.15, 0.15, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["a"], first["b"])


if __name__ == "__main__":
    unittest.main()
