"""Tests for the full-dataset conversion-integrity readiness checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from full_dataset_readiness_check import Box, match_boxes, parse_yolo, review_record, select_pass_sample  # noqa: E402


class FullDatasetReadinessTests(unittest.TestCase):
    def test_exact_voc_to_yolo_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.2 0.4 0.2 0.4\n", encoding="utf-8")
            boxes, errors = parse_yolo(path, (100, 100))
        self.assertEqual(errors, [])
        self.assertAlmostEqual(boxes[0].xmin, 10.0)
        self.assertAlmostEqual(boxes[0].ymax, 60.0)

    def test_one_pixel_tolerance(self) -> None:
        matches, unmatched_source, unmatched_converted = match_boxes([Box(0, 0, 1000, 1000)], [Box(1, 0, 1001, 1000)])
        self.assertEqual((unmatched_source, unmatched_converted), (0, 0))
        self.assertGreaterEqual(matches[0][2], 0.99)

    def test_object_count_mismatch_and_unmatched_boxes_reject(self) -> None:
        result = self._review_with_label("")
        self.assertEqual(result["automated_status"], "REJECT")
        self.assertIn("source_and_converted_object_counts_differ", result["automated_reasons"])
        self.assertIn("unmatched_source_or_converted_boxes", result["automated_reasons"])

    def test_unmatched_boxes_reject_with_matching_counts(self) -> None:
        result = self._review_with_label("0 0.85 0.85 0.1 0.1\n")
        self.assertEqual(result["automated_status"], "REJECT")
        self.assertIn("unmatched_source_or_converted_boxes", result["automated_reasons"])

    def test_invalid_class_id_rejects(self) -> None:
        result = self._review_with_label("1 0.15 0.15 0.1 0.1\n")
        self.assertEqual(result["automated_status"], "REJECT")
        self.assertIn("invalid_class_id", result["automated_reasons"])

    def test_invalid_coordinates_reject(self) -> None:
        result = self._review_with_label("0 1.1 0.15 0.1 0.1\n")
        self.assertEqual(result["automated_status"], "REJECT")
        self.assertIn("invalid_normalized_coordinate", result["automated_reasons"])

    def test_dimension_mismatch_rejects(self) -> None:
        result = self._review_with_label("0 0.15 0.15 0.1 0.1\n", converted_size=(99, 100))
        self.assertEqual(result["automated_status"], "REJECT")
        self.assertIn("source_xml_and_converted_dimensions_conflict", result["automated_reasons"])

    def test_duplicate_group_crossing_splits_rejects(self) -> None:
        result = self._review_with_label("0 0.15 0.15 0.1 0.1\n", duplicate_crosses=True)
        self.assertEqual(result["automated_status"], "REJECT")
        self.assertIn("duplicate_group_crosses_splits", result["automated_reasons"])

    def test_pass_sampling_is_deterministic(self) -> None:
        records = [{"image_id": f"{split}_{number}", "split": split, "automated_status": "PASS"} for split, total in (("train", 5), ("val", 4), ("test", 4)) for number in range(total)]
        self.assertEqual(select_pass_sample(records), select_pass_sample(records))
        self.assertEqual({"train": 4, "val": 3, "test": 3}, {split: sum(row["split"] == split for row in select_pass_sample(records)) for split in ("train", "val", "test")})

    def _review_with_label(self, label: str, converted_size: tuple[int, int] = (100, 100), duplicate_crosses: bool = False) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, sources, dataset = root / "xml", root / "source", root / "dataset"
            annotations.mkdir(); sources.mkdir(); (dataset / "images/train").mkdir(parents=True); (dataset / "labels/train").mkdir(parents=True)
            Image.new("RGB", (100, 100)).save(sources / "item.jpg")
            Image.new("RGB", converted_size).save(dataset / "images/train/item.jpg")
            (annotations / "item.xml").write_text("<annotation><filename>item.jpg</filename><size><width>100</width><height>100</height></size><object><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object></annotation>", encoding="utf-8")
            (dataset / "labels/train/item.txt").write_text(label, encoding="utf-8")
            return review_record({"image_id": "item", "split": "train", "image_path": "images/train/item.jpg", "label_path": "labels/train/item.txt", "object_count": "1", "duplicate_group_id": ""}, annotations, sources, dataset, "", duplicate_crosses, False, False, False)


if __name__ == "__main__":
    unittest.main()
