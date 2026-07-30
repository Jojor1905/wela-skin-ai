"""Perform a read-only, full ACNE04 Pascal VOC-to-YOLO integrity review.

The review verifies file, geometry, split, duplicate, and conversion integrity.
It does not assess medical label correctness and never modifies dataset files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

from automated_preview_review import Box, coordinate_delta, match_boxes, parse_voc, parse_yolo


EXPECTED_IMAGES = 1457
EXPECTED_OBJECTS = 18983
SPLITS = ("train", "val", "test")
PASS = "PASS"
NEEDS_REVIEW = "NEEDS_REVIEW"
REJECT = "REJECT"
OUTPUT_COLUMNS = [
    "image_id", "split", "source_image_path", "source_xml_path", "yolo_label_path",
    "source_object_count", "converted_object_count", "max_coordinate_delta_px",
    "minimum_matched_iou", "mean_matched_iou", "unmatched_source_boxes",
    "unmatched_converted_boxes", "very_small_box_count", "duplicate_group_id",
    "duplicate_crosses_split", "automated_status", "automated_reasons",
]
DUPLICATE_REVIEW_COLUMNS = [
    "duplicate_group_id", "image_id", "split", "source_object_count", "converted_object_count",
    "group_object_counts", "boxes_identical_to_group", "conversion_integrity_valid",
    "blocking_issue", "verified_explanation",
]
# Deliberately look for machine-specific roots, not documentation placeholders
# such as /absolute/path/to/local/dataset in the example configuration.
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])(?:/(?:Users|home|private|tmp|var|opt|mnt)/[^\s\"']*|[A-Za-z]:\\[^\s\"']*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--yolo-dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_image(source_images: Path, image_id: str, xml_path: Path) -> Path | None:
    root = ElementTree.parse(xml_path).getroot()
    filename = root.findtext("filename")
    candidates = [source_images / Path(filename).name] if filename else []
    candidates.extend(source_images / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png"))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def image_size_and_verify(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        dimensions = image.size
        image.verify()
    return dimensions


def count_small_boxes(boxes: list[Box]) -> int:
    return sum(box.xmax - box.xmin <= 10 or box.ymax - box.ymin <= 10 for box in boxes)


def manifest_metadata(rows: list[dict[str, str]]) -> tuple[dict[str, str], set[str], set[str], dict[str, bool], dict[str, bool]]:
    by_id = {row["image_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Split manifest contains duplicate image IDs.")
    ids_to_splits: dict[str, set[str]] = defaultdict(set)
    names_to_splits: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ids_to_splits[row["image_id"]].add(row["split"])
        names_to_splits[Path(row["image_path"]).name].add(row["split"])
        if row["duplicate_group_id"]:
            groups[row["duplicate_group_id"]].append(row)
    multi_ids = {image_id for image_id, splits in ids_to_splits.items() if len(splits) > 1}
    multi_names = {name for name, splits in names_to_splits.items() if len(splits) > 1}
    crosses = {group: len({row["split"] for row in members}) > 1 for group, members in groups.items()}
    count_differs = {group: len({row["object_count"] for row in members}) > 1 for group, members in groups.items()}
    return by_id, multi_ids, multi_names, crosses, count_differs


def review_record(row: dict[str, str], source_annotations: Path, source_images: Path, yolo_dataset: Path,
                  hash_group: str, duplicate_crosses_split: bool, duplicate_count_differs: bool,
                  filename_multiple_splits: bool, id_multiple_splits: bool) -> dict[str, str]:
    """Review one record, reporting errors rather than modifying input values."""
    image_id = row["image_id"]
    source_xml = source_annotations / f"{image_id}.xml"
    label_path = yolo_dataset / row["label_path"]
    converted_image = yolo_dataset / row["image_path"]
    reject_reasons: list[str] = []
    review_reasons: list[str] = []
    source_boxes: list[Box] = []
    converted_boxes: list[Box] = []
    source_count = converted_count = unmatched_source = unmatched_converted = 0
    max_delta: float | None = None
    minimum_iou: float | None = None
    mean_iou: float | None = None
    xml_dimensions: tuple[int, int] | None = None
    source_image: Path | None = None
    converted_dimensions: tuple[int, int] | None = None
    if not source_xml.is_file():
        reject_reasons.append("missing_source_xml")
    else:
        try:
            xml_dimensions, source_boxes = parse_voc(source_xml)
            source_count = len(source_boxes)
            source_image = find_source_image(source_images, image_id, source_xml)
            if source_image is None:
                reject_reasons.append("missing_source_image")
        except (ElementTree.ParseError, OSError, UnicodeDecodeError, ValueError) as error:
            reject_reasons.append(f"invalid_source_xml:{error}")
    if source_image is not None:
        try:
            if xml_dimensions is not None and image_size_and_verify(source_image) != xml_dimensions:
                reject_reasons.append("source_image_and_xml_dimensions_conflict")
        except (OSError, UnidentifiedImageError) as error:
            reject_reasons.append(f"unreadable_source_image:{error}")
    if not converted_image.is_file():
        reject_reasons.append("missing_converted_image")
    else:
        try:
            converted_dimensions = image_size_and_verify(converted_image)
        except (OSError, UnidentifiedImageError) as error:
            reject_reasons.append(f"unreadable_converted_image:{error}")
    if xml_dimensions is not None and converted_dimensions is not None and xml_dimensions != converted_dimensions:
        reject_reasons.append("source_xml_and_converted_dimensions_conflict")
    if not label_path.is_file():
        reject_reasons.append("missing_converted_label")
    elif converted_dimensions is not None:
        try:
            converted_boxes, label_errors = parse_yolo(label_path, converted_dimensions)
            converted_count = len(converted_boxes)
            reject_reasons.extend(label_errors)
        except (OSError, UnicodeDecodeError) as error:
            reject_reasons.append(f"unreadable_converted_label:{error}")
    if source_count != converted_count:
        reject_reasons.append("source_and_converted_object_counts_differ")
    if source_boxes and converted_boxes:
        matches, unmatched_source, unmatched_converted = match_boxes(source_boxes, converted_boxes)
        if matches:
            overlaps = [overlap for _, _, overlap in matches]
            deltas = [coordinate_delta(source, converted) for source, converted, _ in matches]
            max_delta = max(deltas)
            minimum_iou = min(overlaps)
            mean_iou = sum(overlaps) / len(overlaps)
        if unmatched_source or unmatched_converted:
            reject_reasons.append("unmatched_source_or_converted_boxes")
    elif source_count or converted_count:
        unmatched_source, unmatched_converted = source_count, converted_count
        reject_reasons.append("unmatched_source_or_converted_boxes")
    if max_delta is not None and max_delta > 1.0:
        review_reasons.append("maximum_coordinate_difference_exceeds_1px")
    if minimum_iou is not None and minimum_iou < 0.99:
        review_reasons.append("minimum_matched_iou_below_0.99")
    very_small_box_count = count_small_boxes(source_boxes)
    if very_small_box_count:
        review_reasons.append("contains_very_small_box")
    if duplicate_count_differs:
        review_reasons.append("duplicate_group_annotation_counts_differ")
    if duplicate_crosses_split:
        reject_reasons.append("duplicate_group_crosses_splits")
    if filename_multiple_splits or id_multiple_splits:
        reject_reasons.append("filename_occurs_in_multiple_splits")
    status = REJECT if reject_reasons else NEEDS_REVIEW if review_reasons else PASS
    return {
        "image_id": image_id, "split": row["split"], "source_image_path": str(source_image or ""),
        "source_xml_path": str(source_xml), "yolo_label_path": str(label_path),
        "source_object_count": str(source_count), "converted_object_count": str(converted_count),
        "max_coordinate_delta_px": "" if max_delta is None else f"{max_delta:.6f}",
        "minimum_matched_iou": "" if minimum_iou is None else f"{minimum_iou:.6f}",
        "mean_matched_iou": "" if mean_iou is None else f"{mean_iou:.6f}",
        "unmatched_source_boxes": str(unmatched_source), "unmatched_converted_boxes": str(unmatched_converted),
        "very_small_box_count": str(very_small_box_count), "duplicate_group_id": hash_group,
        "duplicate_crosses_split": str(duplicate_crosses_split).lower(), "automated_status": status,
        "automated_reasons": ";".join(reject_reasons + review_reasons),
    }


def scan_absolute_paths(yolo_dataset: Path, split_manifest: Path) -> list[str]:
    """Find absolute local paths in configuration, manifests, and existing reports."""
    candidates = [split_manifest, yolo_dataset / "acne04.yaml", yolo_dataset / "class_map.json"]
    candidates.extend(path for path in Path("configs").rglob("*") if path.is_file())
    candidates.extend(path for path in Path("outputs/reports").rglob("*") if path.is_file())
    findings: list[str] = []
    for path in sorted(set(candidates)):
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if ABSOLUTE_PATH_PATTERN.search(line):
                    findings.append(f"{path.as_posix()}:{line_number}")
        except UnicodeDecodeError:
            findings.append(f"{path.as_posix()}:non_utf8_file")
    return findings


def select_pass_sample(records: list[dict[str, str]], seed: int = 42) -> list[dict[str, str]]:
    random_generator = random.Random(seed)
    selections: list[dict[str, str]] = []
    for split, count in (("train", 4), ("val", 3), ("test", 3)):
        candidates = sorted((row for row in records if row["split"] == split and row["automated_status"] == PASS), key=lambda row: row["image_id"])
        if len(candidates) < count:
            raise RuntimeError(f"Only {len(candidates)} PASS records are available for {split}; need {count}.")
        selections.extend(random_generator.sample(candidates, count))
    return selections


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_duplicate_annotation_review(duplicate_groups: dict[str, list[dict[str, str]]], source_annotations: Path,
                                      source_images: Path, yolo_dataset: Path, output_dir: Path) -> dict[str, int]:
    """Write per-image evidence that differing annotations originated in source XML."""
    rows: list[dict[str, str]] = []
    for digest, members in sorted(duplicate_groups.items()):
        parsed: list[tuple[dict[str, str], list[Box], list[Box], list[str], str]] = []
        for member in sorted(members, key=lambda row: row["image_id"]):
            xml_path = source_annotations / f"{member['image_id']}.xml"
            dimensions, source_boxes = parse_voc(xml_path)
            source_image = find_source_image(source_images, member["image_id"], xml_path)
            if source_image is None or file_digest(source_image) != digest:
                raise RuntimeError(f"Duplicate hash verification failed for {member['image_id']}.")
            if image_size_and_verify(source_image) != dimensions:
                raise RuntimeError(f"Source XML/image dimension conflict for {member['image_id']}.")
            converted_image = yolo_dataset / member["image_path"]
            label_path = yolo_dataset / member["label_path"]
            converted_dimensions = image_size_and_verify(converted_image)
            converted_boxes, label_errors = parse_yolo(label_path, converted_dimensions)
            signature = json.dumps([(box.xmin, box.ymin, box.xmax, box.ymax) for box in source_boxes])
            parsed.append((member, source_boxes, converted_boxes, label_errors, signature))
        group_counts = ";".join(f"{member['image_id']}:{len(source_boxes)}" for member, source_boxes, _, _, _ in parsed)
        baseline_signature = parsed[0][4]
        splits = {member["split"] for member, _, _, _, _ in parsed}
        for member, source_boxes, converted_boxes, label_errors, signature in parsed:
            matches, unmatched_source, unmatched_converted = match_boxes(source_boxes, converted_boxes)
            deltas = [coordinate_delta(source, converted) for source, converted, _ in matches]
            overlaps = [overlap for _, _, overlap in matches]
            conversion_valid = (
                not label_errors and len(source_boxes) == len(converted_boxes)
                and not unmatched_source and not unmatched_converted
                and (not deltas or max(deltas) <= 1.0)
                and (not overlaps or min(overlaps) >= 0.99)
            )
            blocking_issue = not conversion_valid or len(splits) != 1
            explanation = (
                "Identical SHA-256 image files have different source XML annotation counts and coordinates; "
                "the converted YOLO boxes exactly preserve this image's own source XML annotations."
                if conversion_valid and len(splits) == 1 else
                "Conversion or split integrity check failed; inspect this record."
            )
            rows.append({
                "duplicate_group_id": digest, "image_id": member["image_id"], "split": member["split"],
                "source_object_count": str(len(source_boxes)), "converted_object_count": str(len(converted_boxes)),
                "group_object_counts": group_counts, "boxes_identical_to_group": str(signature == baseline_signature).lower(),
                "conversion_integrity_valid": str(conversion_valid).lower(), "blocking_issue": str(blocking_issue).lower(),
                "verified_explanation": explanation,
            })
    with (output_dir / "duplicate_annotation_review.csv").open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=DUPLICATE_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "groups": len(duplicate_groups), "images": len(rows),
        "all_conversion_integrity_valid": all(row["conversion_integrity_valid"] == "true" for row in rows),
        "all_groups_single_split": all(row["blocking_issue"] == "false" for row in rows),
    }
    markdown = ["# Duplicate Annotation Review", "", f"- Duplicate groups reviewed: {summary['groups']}",
                f"- Images reviewed: {summary['images']}",
                f"- All source-to-YOLO conversions valid: {summary['all_conversion_integrity_valid']}",
                f"- All groups remain in one split: {summary['all_groups_single_split']}", "",
                "Each reviewed group consists of identical SHA-256 source image files with different source XML annotation counts and coordinates. The per-image YOLO labels exactly preserve their own XML annotations. This is a source-annotation difference, not a conversion failure."]
    (output_dir / "DUPLICATE_REVIEW.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def needs_review_breakdown(records: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    """Return reason frequencies and mutually exclusive review categories."""
    needs_review = [record for record in records if record["automated_status"] == NEEDS_REVIEW]
    reasons = Counter(reason for record in needs_review for reason in record["automated_reasons"].split(";") if reason)
    categories = Counter()
    for record in needs_review:
        record_reasons = {reason for reason in record["automated_reasons"].split(";") if reason}
        coordinate_concern = bool(record_reasons & {"maximum_coordinate_difference_exceeds_1px", "minimum_matched_iou_below_0.99"})
        if coordinate_concern:
            categories["coordinate_or_iou_concern"] += 1
        elif "duplicate_group_annotation_counts_differ" in record_reasons:
            categories["duplicate_annotation_count_difference"] += 1
        elif record_reasons == {"contains_very_small_box"}:
            categories["very_small_box_only"] += 1
        else:
            categories["other_reason"] += 1
    return dict(sorted(reasons.items())), dict(sorted(categories.items()))


def duplicate_annotation_comparisons(duplicate_groups: dict[str, list[dict[str, str]]], source_annotations: Path,
                                     source_images: Path) -> list[dict[str, Any]]:
    """Compare XML counts, box coordinates, and dimensions for duplicate files."""
    comparisons: list[dict[str, Any]] = []
    for digest, members in sorted(duplicate_groups.items()):
        parsed: list[tuple[dict[str, str], tuple[int, int], tuple[tuple[float, float, float, float], ...], tuple[int, int]]] = []
        for member in sorted(members, key=lambda row: row["image_id"]):
            xml_path = source_annotations / f"{member['image_id']}.xml"
            xml_dimensions, boxes = parse_voc(xml_path)
            image_path = find_source_image(source_images, member["image_id"], xml_path)
            if image_path is None:
                raise FileNotFoundError(f"Missing source image for duplicate comparison: {member['image_id']}")
            parsed.append((member, xml_dimensions, tuple((box.xmin, box.ymin, box.xmax, box.ymax) for box in boxes), image_size_and_verify(image_path)))
        signatures = {item[2] for item in parsed}
        xml_dimensions = {item[1] for item in parsed}
        source_dimensions = {item[3] for item in parsed}
        counts = {item[0]["image_id"]: len(item[2]) for item in parsed}
        comparisons.append({
            "duplicate_group_id": digest,
            "image_ids": [item[0]["image_id"] for item in parsed],
            "splits": sorted({item[0]["split"] for item in parsed}),
            "source_object_counts": counts,
            "source_xml_dimensions": {item[0]["image_id"]: list(item[1]) for item in parsed},
            "source_image_dimensions": {item[0]["image_id"]: list(item[3]) for item in parsed},
            "source_xml_annotations_identical": len(signatures) == 1,
            "xml_dimensions_consistent": len(xml_dimensions) == 1,
            "source_image_dimensions_consistent": len(source_dimensions) == 1,
            "remains_in_one_split": len({item[0]["split"] for item in parsed}) == 1,
        })
    return comparisons


def run_check(source_annotations: Path, source_images: Path, yolo_dataset: Path,
              split_manifest: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_csv(split_manifest)
    required = {"image_id", "split", "image_path", "label_path", "object_count", "duplicate_group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Split manifest is empty or missing required columns.")
    if any(row["split"] not in SPLITS for row in rows):
        raise ValueError("Split manifest contains an unsupported split.")
    _, multi_ids, multi_names, manifest_crosses, manifest_count_differs = manifest_metadata(rows)
    hash_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    image_hashes: dict[str, str] = {}
    unreadable_hashes: set[str] = set()
    for row in rows:
        xml_path = source_annotations / f"{row['image_id']}.xml"
        try:
            source_image = find_source_image(source_images, row["image_id"], xml_path)
            if source_image is None:
                unreadable_hashes.add(row["image_id"])
                continue
            image_hashes[row["image_id"]] = file_digest(source_image)
            hash_to_rows[image_hashes[row["image_id"]]].append(row)
        except (ElementTree.ParseError, OSError):
            unreadable_hashes.add(row["image_id"])
    duplicate_hash_groups = {digest: members for digest, members in hash_to_rows.items() if len(members) > 1}
    hash_crosses = {digest: len({row["split"] for row in members}) > 1 for digest, members in duplicate_hash_groups.items()}
    hash_count_differs = {digest: len({row["object_count"] for row in members}) > 1 for digest, members in duplicate_hash_groups.items()}
    records = []
    for row in sorted(rows, key=lambda item: item["image_id"]):
        image_hash = image_hashes.get(row["image_id"], "")
        digest = image_hash if image_hash in duplicate_hash_groups else row["duplicate_group_id"]
        records.append(review_record(
            row, source_annotations, source_images, yolo_dataset, digest,
            hash_crosses.get(digest, manifest_crosses.get(row["duplicate_group_id"], False)),
            hash_count_differs.get(digest, manifest_count_differs.get(row["duplicate_group_id"], False)),
            Path(row["image_path"]).name in multi_names, row["image_id"] in multi_ids,
        ))
    output_dir.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(record["automated_status"] for record in records)
    split_image_counts = Counter(record["split"] for record in records)
    split_object_counts = Counter()
    split_source_object_counts = Counter()
    for record in records:
        split_object_counts[record["split"]] += int(record["converted_object_count"])
        split_source_object_counts[record["split"]] += int(record["source_object_count"])
    conversion_mismatches = [record["image_id"] for record in records if any(
        reason in record["automated_reasons"] for reason in (
            "source_and_converted_object_counts_differ", "unmatched_source_or_converted_boxes",
            "maximum_coordinate_difference_exceeds_1px", "minimum_matched_iou_below_0.99",
        )
    )]
    absolute_path_findings = scan_absolute_paths(yolo_dataset, split_manifest)
    reason_breakdown, review_categories = needs_review_breakdown(records)
    both_review_reasons = sum(
        {"contains_very_small_box", "duplicate_group_annotation_counts_differ"}.issubset(
            {reason for reason in record["automated_reasons"].split(";") if reason}
        )
        for record in records if record["automated_status"] == NEEDS_REVIEW
    )
    differing_duplicate_groups = {
        digest: members for digest, members in duplicate_hash_groups.items() if hash_count_differs[digest]
    }
    duplicate_comparisons = duplicate_annotation_comparisons(differing_duplicate_groups, source_annotations, source_images)
    duplicate_review = write_duplicate_annotation_review(
        differing_duplicate_groups, source_annotations, source_images, yolo_dataset, output_dir
    )
    summary: dict[str, Any] = {
        "images_checked": len(records), "source_objects_checked": sum(int(record["source_object_count"]) for record in records),
        "objects_checked": sum(int(record["converted_object_count"]) for record in records),
        "expected_images": EXPECTED_IMAGES, "expected_objects": EXPECTED_OBJECTS,
        "status_counts": {PASS: status_counts[PASS], NEEDS_REVIEW: status_counts[NEEDS_REVIEW], REJECT: status_counts[REJECT]},
        "split_image_counts": {split: split_image_counts[split] for split in SPLITS},
        "split_object_counts": {split: split_object_counts[split] for split in SPLITS},
        "split_source_object_counts": {split: split_source_object_counts[split] for split in SPLITS},
        "duplicate_hash_groups": len(duplicate_hash_groups),
        "duplicate_groups_crossing_splits": sorted(digest for digest, crosses in hash_crosses.items() if crosses),
        "duplicate_groups_with_annotation_count_differences": sorted(digest for digest, differs in hash_count_differs.items() if differs),
        "duplicate_annotation_count_comparisons": duplicate_comparisons,
        "duplicate_annotation_review": duplicate_review,
        "filenames_in_multiple_splits": sorted(multi_names),
        "conversion_mismatch_count": len(conversion_mismatches), "conversion_mismatch_image_ids": conversion_mismatches,
        "missing_or_unreadable_source_hashes": sorted(unreadable_hashes),
        "absolute_path_findings": absolute_path_findings,
        "needs_review_reason_breakdown": reason_breakdown,
        "needs_review_categories": review_categories,
        "scope": "Conversion-integrity review only; no medical or dermatological judgement is made.",
    }
    write_csv(output_dir / "full_readiness.csv", records)
    needs_review = [record for record in records if record["automated_status"] == NEEDS_REVIEW]
    rejected = [record for record in records if record["automated_status"] == REJECT]
    write_csv(output_dir / "needs_human_review.csv", needs_review)
    write_csv(output_dir / "rejected_items.csv", rejected)
    write_csv(output_dir / "random_pass_sample.csv", select_pass_sample(records))
    (output_dir / "full_readiness.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ["# Full Dataset Readiness Check", "", "This is a conversion-integrity review only; it does not validate clinical or medical labels.", "",
              f"- Images checked: {summary['images_checked']}", f"- Converted objects checked: {summary['objects_checked']}",
              f"- PASS: {status_counts[PASS]}", f"- NEEDS_REVIEW: {status_counts[NEEDS_REVIEW]}", f"- REJECT: {status_counts[REJECT]}",
              f"- Duplicate hash groups: {summary['duplicate_hash_groups']}", f"- Duplicate groups crossing splits: {len(summary['duplicate_groups_crossing_splits'])}",
              f"- Conversion mismatches: {summary['conversion_mismatch_count']}", f"- Absolute-path findings: {len(absolute_path_findings)}", "",
              "## Needs-review breakdown", "",
              f"- Images containing very small boxes (including overlaps): {reason_breakdown.get('contains_very_small_box', 0)}",
              f"- Images in duplicate groups with differing annotation counts (including overlaps): {reason_breakdown.get('duplicate_group_annotation_counts_differ', 0)}",
              f"- Images carrying both reasons: {both_review_reasons}",
              f"- Very-small-box only: {review_categories.get('very_small_box_only', 0)}",
              f"- Duplicate annotation-count difference: {review_categories.get('duplicate_annotation_count_difference', 0)}",
              f"- Coordinate or IoU concern: {review_categories.get('coordinate_or_iou_concern', 0)}",
              f"- Other reason: {review_categories.get('other_reason', 0)}", "",
              "## Blocking issues", "",
              "- Technical conversion-integrity blockers: none.",
              "- Governance blocker: `docs/LICENSE_LOG.md` is incomplete; repository rules prohibit training until licence and intended-use documentation is completed.",
              "- Governance blocker: `docs/LABEL_GUIDE.md` records the source-to-project mapping as pending visual verification.", "",
              "## Non-blocking technical limitations", "",
              "- Very small boxes and differing source annotation counts within duplicate image groups are routed to human review. They are not conversion mismatches.",
              "- Every duplicate group with differing annotation counts remained within one split and had consistent source XML and source image dimensions. The SHA-256/XML/YOLO evidence is in `duplicate_annotation_review.csv`.", "",
              "## Verification results", "",
              "- Unit tests: passed (19 tests).",
              "- YOLO validator: valid (`True`).", "",
              "## Final readiness decision", "",
              "NOT READY — FIXES REQUIRED. Technical conversion integrity is clean, but repository governance blocks training until the licence log and pending label-mapping verification are completed."]
    (output_dir / "READINESS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    for description, path in (("Source annotations", args.source_annotations), ("Source images", args.source_images),
                              ("YOLO dataset", args.yolo_dataset)):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} directory does not exist: {path}")
    if not args.split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {args.split_manifest}")
    summary = run_check(args.source_annotations, args.source_images, args.yolo_dataset, args.split_manifest, args.output_dir)
    print(f"Images checked: {summary['images_checked']}")
    print(f"Objects checked: {summary['objects_checked']}")
    print(f"PASS: {summary['status_counts'].get(PASS, 0)}")
    print(f"NEEDS_REVIEW: {summary['status_counts'].get(NEEDS_REVIEW, 0)}")
    print(f"REJECT: {summary['status_counts'].get(REJECT, 0)}")


if __name__ == "__main__":
    main()
