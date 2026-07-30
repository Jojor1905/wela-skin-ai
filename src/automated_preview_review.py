"""Review YOLO preview conversion integrity against Pascal VOC source annotations.

This read-only review checks geometry and dataset split metadata.  It does not
assess whether any annotated region has a medically correct label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image


OUTPUT_COLUMNS = [
    "image_id", "split", "preview_path", "sampling_reason",
    "source_object_count", "converted_object_count", "max_coordinate_delta_px",
    "minimum_matched_iou", "mean_matched_iou", "unmatched_source_boxes",
    "unmatched_converted_boxes", "very_small_box_count", "duplicate_group_id",
    "duplicate_crosses_split", "automated_status", "automated_reasons",
    "review_status", "reviewer_notes",
]


@dataclass(frozen=True)
class Box:
    """One pixel-coordinate bounding box, without coordinate repair."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--yolo-dataset", type=Path, required=True)
    parser.add_argument("--preview-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def parse_voc(xml_path: Path) -> tuple[tuple[int, int], list[Box]]:
    """Read source XML dimensions and boxes without changing coordinates."""
    root = ElementTree.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("Source XML has no <size> element.")
    try:
        dimensions = (int(size.findtext("width", "")), int(size.findtext("height", "")))
    except ValueError as error:
        raise ValueError("Source XML has non-integer dimensions.") from error
    if min(dimensions) <= 0:
        raise ValueError("Source XML dimensions must be positive.")
    boxes: list[Box] = []
    for number, object_element in enumerate(root.findall("object"), start=1):
        bndbox = object_element.find("bndbox")
        if bndbox is None:
            raise ValueError(f"Source XML object {number} has no <bndbox>.")
        try:
            coordinates = [float(bndbox.findtext(name, "")) for name in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError as error:
            raise ValueError(f"Source XML object {number} has non-numeric coordinates.") from error
        boxes.append(Box(*coordinates))
    return dimensions, boxes


def parse_yolo(label_path: Path, dimensions: tuple[int, int]) -> tuple[list[Box], list[str]]:
    """Parse YOLO labels and reconstruct boxes; invalid rows are reported, not fixed."""
    width, height = dimensions
    boxes: list[Box] = []
    errors: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(f"label_line_{line_number}_does_not_have_five_values")
            continue
        try:
            class_id = int(values[0])
            xc, yc, box_width, box_height = (float(value) for value in values[1:])
        except ValueError:
            errors.append(f"label_line_{line_number}_has_non_numeric_values")
            continue
        if class_id != 0:
            errors.append(f"invalid_class_id_line_{line_number}")
        coordinates = (xc, yc, box_width, box_height)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
            errors.append(f"invalid_normalized_coordinate_line_{line_number}")
        if box_width <= 0.0 or box_height <= 0.0:
            errors.append(f"non_positive_box_size_line_{line_number}")
        box = Box((xc - box_width / 2) * width, (yc - box_height / 2) * height,
                  (xc + box_width / 2) * width, (yc + box_height / 2) * height)
        if not all(math.isfinite(value) for value in (box.xmin, box.ymin, box.xmax, box.ymax)) or (
            box.xmin < 0 or box.ymin < 0 or box.xmax > width or box.ymax > height
            or box.xmax <= box.xmin or box.ymax <= box.ymin
        ):
            errors.append(f"reconstructed_box_outside_image_line_{line_number}")
        boxes.append(box)
    return boxes, errors


def iou(first: Box, second: Box) -> float:
    intersection_width = max(0.0, min(first.xmax, second.xmax) - max(first.xmin, second.xmin))
    intersection_height = max(0.0, min(first.ymax, second.ymax) - max(first.ymin, second.ymin))
    intersection = intersection_width * intersection_height
    union = ((first.xmax - first.xmin) * (first.ymax - first.ymin)
             + (second.xmax - second.xmin) * (second.ymax - second.ymin) - intersection)
    return intersection / union if union > 0 else 0.0


def match_boxes(source_boxes: list[Box], converted_boxes: list[Box]) -> tuple[list[tuple[Box, Box, float]], int, int]:
    """Greedily pair highest-IoU boxes, leaving non-overlapping boxes unmatched."""
    candidates = sorted(
        ((iou(source, converted), source_index, converted_index)
         for source_index, source in enumerate(source_boxes)
         for converted_index, converted in enumerate(converted_boxes)),
        reverse=True,
    )
    used_source: set[int] = set()
    used_converted: set[int] = set()
    matches: list[tuple[Box, Box, float]] = []
    for overlap, source_index, converted_index in candidates:
        if overlap <= 0 or source_index in used_source or converted_index in used_converted:
            continue
        used_source.add(source_index)
        used_converted.add(converted_index)
        matches.append((source_boxes[source_index], converted_boxes[converted_index], overlap))
    return matches, len(source_boxes) - len(used_source), len(converted_boxes) - len(used_converted)


def coordinate_delta(first: Box, second: Box) -> float:
    return max(abs(left - right) for left, right in zip(
        (first.xmin, first.ymin, first.xmax, first.ymax),
        (second.xmin, second.ymin, second.xmax, second.ymax),
    ))


def resolve_source_image(source_images: Path, image_id: str, xml_path: Path) -> Path | None:
    root = ElementTree.parse(xml_path).getroot()
    filename = root.findtext("filename")
    candidates = [source_images / Path(filename).name] if filename else []
    candidates.extend(source_images / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png"))
    return next((path for path in candidates if path.is_file()), None)


def load_human_fields(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return {
        row["image_id"]: {
            "review_status": row.get("review_status", ""),
            "reviewer_notes": row.get("reviewer_notes", ""),
        }
        for row in read_csv(path) if row.get("image_id")
    }


def manifest_checks(rows: list[dict[str, str]]) -> tuple[set[str], set[str], dict[str, list[dict[str, str]]], dict[str, bool], dict[str, bool]]:
    """Evaluate dataset-wide split and duplicate metadata for preview rows."""
    ids_to_splits: dict[str, set[str]] = defaultdict(set)
    filenames_to_splits: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ids_to_splits[row["image_id"]].add(row["split"])
        filenames_to_splits[Path(row["image_path"]).name].add(row["split"])
        if row["duplicate_group_id"]:
            groups[row["duplicate_group_id"]].append(row)
    duplicate_crosses = {group: len({item["split"] for item in members}) > 1 for group, members in groups.items()}
    annotation_differences = {
        group: len({item["object_count"] for item in members}) > 1 for group, members in groups.items()
    }
    multi_split_ids = {image_id for image_id, splits in ids_to_splits.items() if len(splits) > 1}
    multi_split_names = {name for name, splits in filenames_to_splits.items() if len(splits) > 1}
    return multi_split_ids, multi_split_names, groups, duplicate_crosses, annotation_differences


def review_preview(
    preview: dict[str, str], manifest: dict[str, str], source_annotations: Path,
    source_images: Path, yolo_dataset: Path, duplicate_crosses_split: bool,
    duplicate_annotation_difference: bool, filename_multiple_splits: bool, id_multiple_splits: bool,
) -> dict[str, str]:
    """Run all integrity checks for one preview and assign an automated status."""
    image_id = preview["image_id"]
    reasons: list[str] = []
    reject_reasons: list[str] = []
    review_reasons: list[str] = []
    xml_path = source_annotations / f"{image_id}.xml"
    label_path = yolo_dataset / manifest["label_path"]
    converted_image_path = yolo_dataset / manifest["image_path"]
    source_count = converted_count = 0
    max_delta: float | None = None
    minimum_iou: float | None = None
    mean_iou: float | None = None
    unmatched_source = unmatched_converted = 0
    if not xml_path.is_file():
        reject_reasons.append("missing_source_xml")
    if not label_path.is_file():
        reject_reasons.append("missing_converted_label")
    if not converted_image_path.is_file():
        reject_reasons.append("missing_converted_image")
    source_image_path: Path | None = None
    xml_dimensions: tuple[int, int] | None = None
    source_boxes: list[Box] = []
    if xml_path.is_file():
        try:
            xml_dimensions, source_boxes = parse_voc(xml_path)
            source_count = len(source_boxes)
            source_image_path = resolve_source_image(source_images, image_id, xml_path)
            if source_image_path is None:
                reject_reasons.append("missing_source_image")
        except (ElementTree.ParseError, OSError, ValueError) as error:
            reject_reasons.append(f"invalid_source_xml:{error}")
    dimensions: tuple[int, int] | None = None
    if converted_image_path.is_file():
        try:
            with Image.open(converted_image_path) as image:
                dimensions = image.size
        except OSError as error:
            reject_reasons.append(f"unreadable_converted_image:{error}")
    if source_image_path is not None:
        try:
            with Image.open(source_image_path) as image:
                if dimensions is not None and image.size != dimensions:
                    reject_reasons.append("source_and_converted_dimensions_conflict")
                if xml_dimensions is not None and image.size != xml_dimensions:
                    reject_reasons.append("source_image_and_xml_dimensions_conflict")
        except OSError as error:
            reject_reasons.append(f"unreadable_source_image:{error}")
    if xml_dimensions is not None and dimensions is not None and xml_dimensions != dimensions:
        reject_reasons.append("source_xml_and_converted_dimensions_conflict")
    converted_boxes: list[Box] = []
    if label_path.is_file() and dimensions is not None:
        try:
            converted_boxes, label_errors = parse_yolo(label_path, dimensions)
            converted_count = len(converted_boxes)
            reject_reasons.extend(label_errors)
        except OSError as error:
            reject_reasons.append(f"unreadable_converted_label:{error}")
    if source_count != converted_count:
        reject_reasons.append("source_and_converted_object_counts_differ")
    if source_boxes and converted_boxes:
        matches, unmatched_source, unmatched_converted = match_boxes(source_boxes, converted_boxes)
        if matches:
            overlaps = [match[2] for match in matches]
            deltas = [coordinate_delta(match[0], match[1]) for match in matches]
            minimum_iou = min(overlaps)
            mean_iou = sum(overlaps) / len(overlaps)
            max_delta = max(deltas)
        if unmatched_source or unmatched_converted:
            reject_reasons.append("unmatched_source_or_converted_boxes")
    elif source_count or converted_count:
        unmatched_source, unmatched_converted = source_count, converted_count
        reject_reasons.append("unmatched_source_or_converted_boxes")
    if id_multiple_splits or filename_multiple_splits:
        reject_reasons.append("filename_occurs_in_multiple_splits")
    if duplicate_crosses_split:
        reject_reasons.append("duplicate_group_crosses_splits")
    if max_delta is not None and max_delta > 1.0:
        review_reasons.append("maximum_coordinate_difference_exceeds_1px")
    if minimum_iou is not None and minimum_iou < 0.99:
        review_reasons.append("minimum_matched_iou_below_0.99")
    if duplicate_annotation_difference:
        review_reasons.append("duplicate_group_annotation_counts_differ")
    very_small_box_count = int(manifest["small_box_count"])
    if very_small_box_count:
        review_reasons.append("contains_very_small_box")
    if reject_reasons:
        status = "Reject"
        reasons = reject_reasons + review_reasons
    elif review_reasons:
        status = "Needs review"
        reasons = review_reasons
    else:
        status = "Pass"
    return {
        "image_id": image_id, "split": preview["split"], "preview_path": preview["preview_path"],
        "sampling_reason": preview["sampling_reason"], "source_object_count": str(source_count),
        "converted_object_count": str(converted_count),
        "max_coordinate_delta_px": "" if max_delta is None else f"{max_delta:.6f}",
        "minimum_matched_iou": "" if minimum_iou is None else f"{minimum_iou:.6f}",
        "mean_matched_iou": "" if mean_iou is None else f"{mean_iou:.6f}",
        "unmatched_source_boxes": str(unmatched_source), "unmatched_converted_boxes": str(unmatched_converted),
        "very_small_box_count": str(very_small_box_count), "duplicate_group_id": manifest["duplicate_group_id"],
        "duplicate_crosses_split": str(duplicate_crosses_split).lower(), "automated_status": status,
        "automated_reasons": ";".join(reasons), "review_status": "", "reviewer_notes": "",
    }


def run_review(source_annotations: Path, source_images: Path, yolo_dataset: Path,
               preview_index: Path, split_manifest: Path, output_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Create review artifacts, preserving pre-existing human-review entries."""
    manifest_rows = read_csv(split_manifest)
    preview_rows = read_csv(preview_index)
    required_manifest = {"image_id", "split", "image_path", "label_path", "object_count", "small_box_count", "duplicate_group_id"}
    required_preview = {"image_id", "split", "preview_path", "sampling_reason"}
    if not manifest_rows or not required_manifest.issubset(manifest_rows[0]):
        raise ValueError("Split manifest is empty or lacks required columns.")
    if not preview_rows or not required_preview.issubset(preview_rows[0]):
        raise ValueError("Preview index is empty or lacks required columns.")
    manifest_by_id = {row["image_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Split manifest contains duplicate image IDs.")
    multi_ids, multi_names, groups, duplicate_crossings, annotation_differences = manifest_checks(manifest_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    preserved_human_fields = load_human_fields(output_dir / "automated_review.csv")
    records: list[dict[str, str]] = []
    for preview in preview_rows:
        image_id = preview["image_id"]
        if image_id not in manifest_by_id:
            record = {column: "" for column in OUTPUT_COLUMNS}
            record.update({"image_id": image_id, "split": preview.get("split", ""), "preview_path": preview.get("preview_path", ""),
                           "sampling_reason": preview.get("sampling_reason", ""), "automated_status": "Reject",
                           "automated_reasons": "preview_image_id_missing_from_split_manifest"})
        else:
            manifest = manifest_by_id[image_id]
            record = review_preview(
                preview, manifest, source_annotations, source_images, yolo_dataset,
                duplicate_crossings.get(manifest["duplicate_group_id"], False),
                annotation_differences.get(manifest["duplicate_group_id"], False),
                Path(manifest["image_path"]).name in multi_names, image_id in multi_ids,
            )
            if preview["split"] != manifest["split"]:
                record["automated_status"] = "Reject"
                record["automated_reasons"] = ";".join(filter(None, [record["automated_reasons"], "preview_and_manifest_splits_differ"]))
        record.update(preserved_human_fields.get(image_id, {}))
        records.append(record)
    records.sort(key=lambda row: row["image_id"])
    status_counts = Counter(row["automated_status"] for row in records)
    difference_groups = sorted(group for group, differs in annotation_differences.items() if differs)
    crossing_groups = sorted(group for group, crosses in duplicate_crossings.items() if crosses)
    summary: dict[str, Any] = {
        "previews_reviewed": len(records), "automated_status_counts": dict(sorted(status_counts.items())),
        "images_requiring_human_review": sum(row["automated_status"] == "Needs review" for row in records),
        "conversion_mismatch_found": any(row["automated_status"] == "Reject" for row in records),
        "duplicate_groups_crossing_splits": crossing_groups,
        "duplicate_groups_with_annotation_count_differences": difference_groups,
        "scope": "Conversion-integrity checks only; no medical or dermatological judgement is made.",
    }
    write_outputs(records, summary, output_dir)
    return records, summary


def write_outputs(records: list[dict[str, str]], summary: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "automated_review.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    needs_review = [row for row in records if row["automated_status"] == "Needs review"]
    with (output_dir / "needs_human_review.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(needs_review)
    (output_dir / "automated_review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Automated YOLO Preview Review", "", "This report checks Pascal VOC-to-YOLO conversion integrity only. It does not determine whether any marked region is acne or otherwise medically valid.", "",
             f"- Previews reviewed: {summary['previews_reviewed']}",
             f"- Pass: {summary['automated_status_counts'].get('Pass', 0)}",
             f"- Needs review: {summary['automated_status_counts'].get('Needs review', 0)}",
             f"- Reject: {summary['automated_status_counts'].get('Reject', 0)}",
             f"- Duplicate groups crossing splits: {len(summary['duplicate_groups_crossing_splits'])}",
             f"- Duplicate groups with annotation-count differences: {len(summary['duplicate_groups_with_annotation_count_differences'])}", "",
             "Human reviewers should inspect images listed in `needs_human_review.csv` and any rejected records. The `review_status` and `reviewer_notes` fields are reserved for human input and are preserved on rerun."]
    (output_dir / "AUTOMATED_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for description, path in (("Source annotations", args.source_annotations), ("Source images", args.source_images),
                              ("YOLO dataset", args.yolo_dataset)):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} directory does not exist: {path}")
    for description, path in (("Preview index", args.preview_index), ("Split manifest", args.split_manifest)):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    _, summary = run_review(args.source_annotations, args.source_images, args.yolo_dataset,
                            args.preview_index, args.split_manifest, args.output_dir)
    print(f"Pass: {summary['automated_status_counts'].get('Pass', 0)}")
    print(f"Needs review: {summary['automated_status_counts'].get('Needs review', 0)}")
    print(f"Reject: {summary['automated_status_counts'].get('Reject', 0)}")


if __name__ == "__main__":
    main()
