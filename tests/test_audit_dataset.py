"""Tests for separate-directory VOC image reference resolution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_dataset import audit  # noqa: E402


class SeparateImageDirectoryTests(unittest.TestCase):
    """Verify image references can resolve outside the VOC annotation root."""

    def test_resolves_path_and_stem_and_reports_extension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_root = Path(temporary_directory)
            dataset_root = temp_root / "Detection" / "VOC2007"
            annotation_dir = dataset_root / "Annotations"
            images_dir = temp_root / "Classification" / "JPEGImages"
            annotation_dir.mkdir(parents=True)
            images_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(images_dir / "lesion.png")
            (annotation_dir / "sample.xml").write_text(
                """<annotation><filename>lesion.jpg</filename><path>/unused/lesion.jpg</path>
                <size><width>20</width><height>20</height></size><object><name>fore</name>
                <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>12</xmax><ymax>12</ymax>
                </bndbox></object></annotation>""",
                encoding="utf-8",
            )

            report = audit(dataset_root, images_dir, small_box_max_side=10)

            self.assertEqual(report["summary"]["total_images"], 1)
            self.assertEqual(report["summary"]["missing_images_for_xml"], 0)
            self.assertEqual(report["summary"]["missing_xml_for_images"], 0)
            self.assertEqual(report["summary"]["filename_extension_mismatches"], 1)
            self.assertEqual(
                report["findings"]["filename_extension_mismatches"][0]["path"],
                "Annotations/sample.xml",
            )

    def test_reports_images_without_xml_in_separate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_root = Path(temporary_directory)
            dataset_root = temp_root / "Detection" / "VOC2007"
            images_dir = temp_root / "Classification" / "JPEGImages"
            (dataset_root / "Annotations").mkdir(parents=True)
            images_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(images_dir / "orphan.jpg")

            report = audit(dataset_root, images_dir, small_box_max_side=10)

            self.assertEqual(report["summary"]["missing_xml_for_images"], 1)
            self.assertEqual(
                report["findings"]["missing_xml_for_images"][0]["path"],
                "../../Classification/JPEGImages/orphan.jpg",
            )


if __name__ == "__main__":
    unittest.main()
