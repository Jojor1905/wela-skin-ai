"""Tests for conversion-integrity preview review helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from automated_preview_review import Box, match_boxes, parse_yolo, review_preview  # noqa: E402


class AutomatedPreviewReviewTests(unittest.TestCase):
    def test_exact_pascal_voc_to_yolo_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "sample.txt"
            label_path.write_text("0 0.2 0.4 0.2 0.4\n", encoding="utf-8")
            boxes, errors = parse_yolo(label_path, (100, 100))
        self.assertEqual(errors, [])
        self.assertEqual(len(boxes), 1)
        for actual, expected in zip((boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax), (10.0, 20.0, 30.0, 60.0)):
            self.assertAlmostEqual(actual, expected)

    def test_one_pixel_rounding_tolerance(self) -> None:
        source = [Box(100, 100, 1100, 1100)]
        converted = [Box(101, 100, 1101, 1100)]
        matches, unmatched_source, unmatched_converted = match_boxes(source, converted)
        self.assertEqual((unmatched_source, unmatched_converted), (0, 0))
        self.assertEqual(max(abs(a - b) for a, b in zip((100, 100, 1100, 1100), (101, 100, 1101, 1100))), 1)
        self.assertGreaterEqual(matches[0][2], 0.99)

    def test_invalid_coordinates_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "invalid.txt"
            label_path.write_text("0 1.1 0.5 0.2 0.2\n", encoding="utf-8")
            _, errors = parse_yolo(label_path, (100, 100))
        self.assertTrue(any(error.startswith("invalid_normalized_coordinate") for error in errors))

    def test_object_count_mismatch_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, sources, dataset = root / "xml", root / "source", root / "dataset"
            annotations.mkdir(); sources.mkdir(); (dataset / "images/train").mkdir(parents=True); (dataset / "labels/train").mkdir(parents=True)
            Image.new("RGB", (100, 100)).save(sources / "sample.jpg")
            Image.new("RGB", (100, 100)).save(dataset / "images/train/sample.jpg")
            (annotations / "sample.xml").write_text("<annotation><filename>sample.jpg</filename><size><width>100</width><height>100</height></size><object><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object></annotation>", encoding="utf-8")
            (dataset / "labels/train/sample.txt").write_text("", encoding="utf-8")
            result = review_preview({"image_id": "sample", "split": "train", "preview_path": "images/sample.jpg", "sampling_reason": "test"}, {"image_path": "images/train/sample.jpg", "label_path": "labels/train/sample.txt", "small_box_count": "0", "duplicate_group_id": ""}, annotations, sources, dataset, False, False, False, False)
        self.assertEqual(result["automated_status"], "Reject")
        self.assertIn("source_and_converted_object_counts_differ", result["automated_reasons"])

    def test_duplicate_group_crossing_splits_rejects(self) -> None:
        from automated_preview_review import manifest_checks
        rows = [
            {"image_id": "first", "split": "train", "image_path": "images/train/first.jpg", "object_count": "1", "duplicate_group_id": "group"},
            {"image_id": "second", "split": "test", "image_path": "images/test/second.jpg", "object_count": "1", "duplicate_group_id": "group"},
        ]
        _, _, _, crossings, _ = manifest_checks(rows)
        self.assertTrue(crossings["group"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = review_preview(
                {"image_id": "first", "split": "train", "preview_path": "images/first.jpg", "sampling_reason": "test"},
                {"image_path": "images/train/first.jpg", "label_path": "labels/train/first.txt", "small_box_count": "0", "duplicate_group_id": "group"},
                root / "xml", root / "source", root / "dataset", True, False, False, False,
            )
        self.assertEqual(result["automated_status"], "Reject")
        self.assertIn("duplicate_group_crosses_splits", result["automated_reasons"])


if __name__ == "__main__":
    unittest.main()
