"""Coordinate-conversion tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convert_voc_to_yolo import convert_box  # noqa: E402


class ConvertBoxTests(unittest.TestCase):
    def test_converts_voc_coordinates_without_clamping(self) -> None:
        box = convert_box(10, 20, 30, 60, 100, 200)
        self.assertEqual((box.x_center, box.y_center, box.width, box.height), (0.2, 0.2, 0.2, 0.2))

    def test_rejects_coordinates_outside_the_actual_image(self) -> None:
        with self.assertRaises(ValueError):
            convert_box(0, 0, 101, 10, 100, 100)


if __name__ == "__main__":
    unittest.main()
