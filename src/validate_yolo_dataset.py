"""Validate the structure, labels, split safety, and totals of a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def is_absolute_value(value: str) -> bool:
    """Detect POSIX and Windows absolute paths in an output value."""
    return Path(value).is_absolute() or (len(value) >= 3 and value[1:3] == ":\\")


def validate(dataset_root: Path) -> dict[str, Any]:
    """Return structured validation results without changing the dataset."""
    manifest_path = dataset_root / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest is missing: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    issues: list[str] = []
    required_columns = {"image_id", "split", "image_path", "label_path", "object_count", "small_box_count", "duplicate_group_id"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("Split manifest is empty or missing required columns.")
    image_paths: set[str] = set()
    label_paths: set[str] = set()
    seen_ids: dict[str, set[str]] = defaultdict(set)
    duplicate_splits: dict[str, set[str]] = defaultdict(set)
    split_images = Counter()
    split_objects = Counter()
    split_small_boxes = Counter()
    converted_objects = 0
    for row in rows:
        image_id, split = row["image_id"], row["split"]
        if split not in {"train", "val", "test"}:
            issues.append(f"Invalid split for {image_id}: {split}")
            continue
        for field in ("image_path", "label_path", "source_image_path", "source_xml_path"):
            if field in row and is_absolute_value(row[field]):
                issues.append(f"Absolute path in manifest {field} for {image_id}: {row[field]}")
        image_path = dataset_root / row["image_path"]
        label_path = dataset_root / row["label_path"]
        if not image_path.is_file():
            issues.append(f"Processed image is missing: {row['image_path']}")
        if not label_path.is_file():
            issues.append(f"Processed label is missing: {row['label_path']}")
        image_paths.add(row["image_path"])
        label_paths.add(row["label_path"])
        seen_ids[image_id].add(split)
        if row["duplicate_group_id"]:
            duplicate_splits[row["duplicate_group_id"]].add(split)
        split_images[split] += 1
        try:
            expected_objects = int(row["object_count"])
            small_boxes = int(row["small_box_count"])
        except ValueError:
            issues.append(f"Non-integer object metadata for {image_id}")
            continue
        split_small_boxes[split] += small_boxes
        actual_objects = 0
        if label_path.is_file():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                values = line.split()
                if len(values) != 5:
                    issues.append(f"{row['label_path']} line {line_number} does not contain five values")
                    continue
                try:
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except ValueError:
                    issues.append(f"{row['label_path']} line {line_number} contains non-numeric values")
                    continue
                if class_id != 0:
                    issues.append(f"{row['label_path']} line {line_number} has class ID {class_id}, expected 0")
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    issues.append(f"{row['label_path']} line {line_number} has coordinate outside 0-1")
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    issues.append(f"{row['label_path']} line {line_number} has non-positive width or height")
                actual_objects += 1
        if actual_objects != expected_objects:
            issues.append(f"{image_id} manifest object count {expected_objects} differs from labels {actual_objects}")
        split_objects[split] += actual_objects
        converted_objects += actual_objects
    actual_images = {
        path.relative_to(dataset_root).as_posix()
        for split in ("train", "val", "test")
        for path in (dataset_root / "images" / split).glob("*") if path.is_file()
    }
    actual_labels = {
        path.relative_to(dataset_root).as_posix()
        for split in ("train", "val", "test")
        for path in (dataset_root / "labels" / split).glob("*.txt")
    }
    if actual_images != image_paths:
        issues.append("Processed images do not exactly match the manifest.")
    if actual_labels != label_paths:
        issues.append("Processed labels do not exactly match the manifest.")
    for image_id, splits in seen_ids.items():
        if len(splits) > 1:
            issues.append(f"Image ID appears in multiple splits: {image_id}")
    crossings = sorted(group for group, splits in duplicate_splits.items() if len(splits) > 1)
    if crossings:
        issues.append(f"Duplicate groups cross splits: {', '.join(crossings)}")
    conversion_report_path = Path("outputs/reports/acne04_conversion_report.json")
    source_objects: int | None = None
    if conversion_report_path.is_file():
        conversion_report = json.loads(conversion_report_path.read_text(encoding="utf-8"))
        source_objects = int(conversion_report["source_objects"])
        if source_objects != converted_objects:
            issues.append(f"Source object total {source_objects} differs from converted total {converted_objects}")
    for output_path in (dataset_root / "split_manifest.csv", dataset_root / "acne04.yaml", dataset_root / "class_map.json"):
        if output_path.is_file() and any(is_absolute_value(token) for token in output_path.read_text(encoding="utf-8").split()):
            issues.append(f"Absolute machine path found in {output_path.relative_to(dataset_root)}")
    total_images = len(rows)
    split_summary = [
        {"split": split, "images": split_images[split], "ratio": split_images[split] / total_images,
         "objects": split_objects[split], "very_small_boxes": split_small_boxes[split]}
        for split in ("train", "val", "test")
    ]
    return {
        "valid": not issues,
        "issues": issues,
        "images": total_images,
        "source_objects": source_objects,
        "converted_objects": converted_objects,
        "object_totals_match": source_objects == converted_objects if source_objects is not None else False,
        "duplicate_groups_crossing_splits": crossings,
        "split_summary": split_summary,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    """Write machine-readable and human-readable validation reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acne04_yolo_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# ACNE04 YOLO Validation", "", f"- Valid: {report['valid']}", f"- Images: {report['images']}",
             f"- Source objects: {report['source_objects']}", f"- Converted objects: {report['converted_objects']}",
             f"- Object totals match: {report['object_totals_match']}",
             f"- Duplicate groups crossing splits: {len(report['duplicate_groups_crossing_splits'])}", "", "## Splits", ""]
    for row in report["split_summary"]:
        lines.append(f"- {row['split']}: {row['images']} images ({row['ratio']:.2%}), {row['objects']} objects, {row['very_small_boxes']} very small boxes")
    if report["issues"]:
        lines.extend(("", "## Issues", ""))
        lines.extend(f"- {issue}" for issue in report["issues"])
    (output_dir / "acne04_yolo_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = validate(args.dataset_root)
    write_report(report, args.output_dir)
    print(f"YOLO dataset valid: {report['valid']}")
    if not report["valid"]:
        raise RuntimeError(f"YOLO validation failed with {len(report['issues'])} issue(s).")


if __name__ == "__main__":
    main()
