"""Convert audited ACNE04 Pascal VOC annotations into a duplicate-aware YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image

from split_dataset import SplitItem, assign_splits, validate_ratios


SOURCE_CLASS = "fore"
PROJECT_CLASS = "acne_lesion"
CLASS_ID = 0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DUPLICATE_GROUP_PATTERN = re.compile(r"Duplicate SHA-256 group: ([0-9a-f]{64})\.")


@dataclass(frozen=True)
class ConvertedBox:
    """A valid source box expressed as a YOLO normalized label row."""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ConversionRecord:
    """Validated source metadata ready to be copied and converted."""

    image_id: str
    image_path: Path
    xml_path: Path
    image_width: int
    image_height: int
    boxes: tuple[ConvertedBox, ...]
    small_box_count: int
    duplicate_group_id: str


def parse_args() -> argparse.Namespace:
    """Parse the conversion-and-splitting command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--test-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing processed dataset root.")
    return parser.parse_args()


def project_relative(path: Path) -> str:
    """Return a portable path relative to the current repository directory."""
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def xml_text(element: ElementTree.Element, tag: str) -> str | None:
    """Read stripped XML text while preserving absent fields."""
    value = element.findtext(tag)
    return value.strip() if value is not None else None


def convert_box(xmin: float, ymin: float, xmax: float, ymax: float, image_width: int, image_height: int) -> ConvertedBox:
    """Convert one valid VOC box to a normalized YOLO box without clamping."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("Bounding box has negative coordinates or non-positive size.")
    if xmax > image_width or ymax > image_height:
        raise ValueError("Bounding box exceeds actual image dimensions.")
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height
    values = (x_center, y_center, width, height)
    if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
        raise ValueError("YOLO normalized coordinates are outside the valid range.")
    return ConvertedBox(x_center, y_center, width, height)


def resolve_image(root: ElementTree.Element, images_by_name: dict[str, Path], images_by_stem: dict[str, Path]) -> Path | None:
    """Resolve XML filename, XML path basename, then a supported-extension stem."""
    candidates = [xml_text(root, "filename"), xml_text(root, "path")]
    for candidate in candidates:
        if candidate:
            image_path = images_by_name.get(Path(candidate).name)
            if image_path is not None:
                return image_path
    for candidate in candidates:
        if candidate and Path(candidate).suffix.lower() in IMAGE_SUFFIXES:
            image_path = images_by_stem.get(Path(candidate).stem)
            if image_path is not None:
                return image_path
    return None


def duplicate_groups_from_audit(audit_json: Path, image_id_by_name: dict[str, str]) -> dict[str, str]:
    """Map image IDs to the SHA-256 duplicate group reported by the audit."""
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    duplicate_groups: dict[str, str] = {}
    for finding in report.get("findings", {}).get("duplicate_images", []):
        match = DUPLICATE_GROUP_PATTERN.fullmatch(finding.get("detail", ""))
        if match is None:
            raise ValueError("Audit duplicate finding has an unexpected format.")
        image_id = image_id_by_name.get(Path(finding["path"]).name)
        if image_id is None:
            raise ValueError(f"Audit duplicate finding cannot resolve image: {finding['path']}")
        duplicate_groups[image_id] = match.group(1)
    return duplicate_groups


def parse_record(xml_path: Path, image_path: Path, duplicate_group_id: str) -> ConversionRecord:
    """Validate one XML against its actual image and convert all its boxes in memory."""
    root = ElementTree.parse(xml_path).getroot()
    with Image.open(image_path) as image:
        image_width, image_height = image.size
        image.verify()
    boxes: list[ConvertedBox] = []
    small_box_count = 0
    for index, object_element in enumerate(root.findall("object"), start=1):
        class_name = xml_text(object_element, "name")
        if class_name != SOURCE_CLASS:
            raise ValueError(f"Object {index} has unsupported source class {class_name!r}; expected {SOURCE_CLASS!r}.")
        bndbox = object_element.find("bndbox")
        if bndbox is None:
            raise ValueError(f"Object {index} has no <bndbox>.")
        try:
            xmin, ymin, xmax, ymax = (
                float(xml_text(bndbox, coordinate) or "")
                for coordinate in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError as error:
            raise ValueError(f"Object {index} contains a non-numeric bounding-box coordinate.") from error
        converted = convert_box(xmin, ymin, xmax, ymax, image_width, image_height)
        boxes.append(converted)
        if xmax - xmin <= 10 or ymax - ymin <= 10:
            small_box_count += 1
    return ConversionRecord(
        image_id=xml_path.stem,
        image_path=image_path,
        xml_path=xml_path,
        image_width=image_width,
        image_height=image_height,
        boxes=tuple(boxes),
        small_box_count=small_box_count,
        duplicate_group_id=duplicate_group_id,
    )


def load_records(annotations_dir: Path, images_dir: Path, audit_json: Path) -> tuple[list[ConversionRecord], list[dict[str, str]]]:
    """Preflight every source file so no partial processed dataset is written on failure."""
    image_paths = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    images_by_name = {path.name: path for path in image_paths}
    images_by_stem = {path.stem: path for path in image_paths}
    xml_paths = sorted(annotations_dir.glob("*.xml"))
    image_id_by_name = {xml_path.with_suffix(".jpg").name: xml_path.stem for xml_path in xml_paths}
    for image_path in image_paths:
        image_id_by_name[image_path.name] = image_path.stem
    duplicate_map = duplicate_groups_from_audit(audit_json, image_id_by_name)
    records: list[ConversionRecord] = []
    failures: list[dict[str, str]] = []
    for xml_path in xml_paths:
        try:
            root = ElementTree.parse(xml_path).getroot()
            image_path = resolve_image(root, images_by_name, images_by_stem)
            if image_path is None:
                raise FileNotFoundError("No image resolves from XML <filename>, <path>, or supported stem.")
            record = parse_record(xml_path, image_path, duplicate_map.get(xml_path.stem, ""))
            records.append(record)
        except (ElementTree.ParseError, OSError, ValueError) as error:
            failures.append({"relative_xml_path": project_relative(xml_path), "detail": str(error)})
    if len({record.image_id for record in records}) != len(records):
        failures.append({"relative_xml_path": "", "detail": "Duplicate XML image IDs are not supported."})
    return records, failures


def write_failures(failures: list[dict[str, str]]) -> Path:
    """Write the required relative-path conversion-failure worksheet."""
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "acne04_conversion_failures.csv"
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["relative_xml_path", "detail"])
        writer.writeheader()
        writer.writerows(failures)
    return path


def write_label(path: Path, boxes: tuple[ConvertedBox, ...]) -> None:
    """Write one exact YOLO label file, including an empty file for no objects."""
    with path.open("w", encoding="utf-8") as file_handle:
        for box in boxes:
            file_handle.write(f"{CLASS_ID} {box.x_center:.17g} {box.y_center:.17g} {box.width:.17g} {box.height:.17g}\n")


def write_processed_dataset(
    records: list[ConversionRecord], assignments: dict[str, str], output_root: Path,
    ratios: dict[str, float], seed: int,
) -> list[dict[str, Any]]:
    """Copy images and write labels/manifests only after the preflight has succeeded."""
    manifest_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=False)
    for record in sorted(records, key=lambda item: item.image_id):
        split = assignments[record.image_id]
        output_image = output_root / "images" / split / f"{record.image_id}{record.image_path.suffix.lower()}"
        output_label = output_root / "labels" / split / f"{record.image_id}.txt"
        shutil.copy2(record.image_path, output_image)
        write_label(output_label, record.boxes)
        manifest_rows.append({
            "image_id": record.image_id,
            "split": split,
            "image_path": output_image.relative_to(output_root).as_posix(),
            "label_path": output_label.relative_to(output_root).as_posix(),
            "source_image_path": project_relative(record.image_path),
            "source_xml_path": project_relative(record.xml_path),
            "image_width": record.image_width,
            "image_height": record.image_height,
            "object_count": len(record.boxes),
            "small_box_count": record.small_box_count,
            "duplicate_group_id": record.duplicate_group_id,
        })
    fieldnames = list(manifest_rows[0]) if manifest_rows else []
    with (output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "class_map.json").write_text(
        json.dumps({"source_class": SOURCE_CLASS, "project_class": PROJECT_CLASS, "yolo_class_id": CLASS_ID}, indent=2) + "\n",
        encoding="utf-8",
    )
    yaml_text = "\n".join((
        "path: .", "train: images/train", "val: images/val", "test: images/test",
        "names:", f"  {CLASS_ID}: {PROJECT_CLASS}",
        "",
    ))
    (output_root / "acne04.yaml").write_text(yaml_text, encoding="utf-8")
    return manifest_rows


def split_summary(rows: list[dict[str, Any]], ratios: dict[str, float]) -> list[dict[str, Any]]:
    """Create deterministic, per-split image/object/small-box summary rows."""
    totals = Counter(row["split"] for row in rows)
    objects = Counter()
    small_boxes = Counter()
    for row in rows:
        objects[row["split"]] += int(row["object_count"])
        small_boxes[row["split"]] += int(row["small_box_count"])
    return [
        {"split": split, "images": totals[split], "image_ratio": f"{totals[split] / len(rows):.6f}",
         "target_ratio": f"{ratios[split]:.6f}", "objects": objects[split], "very_small_boxes": small_boxes[split]}
        for split in ("train", "val", "test")
    ]


def write_reports(rows: list[dict[str, Any]], failures: list[dict[str, str]], ratios: dict[str, float], seed: int) -> None:
    """Write required conversion JSON, Markdown, and split-summary reports."""
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = split_summary(rows, ratios)
    with (report_dir / "acne04_split_summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    duplicate_crossings = {
        group_id: {row["split"] for row in rows if row["duplicate_group_id"] == group_id}
        for group_id in {row["duplicate_group_id"] for row in rows if row["duplicate_group_id"]}
    }
    report = {
        "source_images": len(rows), "source_objects": sum(int(row["object_count"]) for row in rows),
        "converted_images": len(rows), "converted_objects": sum(int(row["object_count"]) for row in rows),
        "source_class": SOURCE_CLASS, "project_class": PROJECT_CLASS, "yolo_class_id": CLASS_ID,
        "seed": seed, "split_targets": ratios, "splits": summary_rows,
        "conversion_failures": len(failures),
        "duplicate_groups_crossing_splits": sum(len(splits) > 1 for splits in duplicate_crossings.values()),
        "person_level_separation": "cannot be guaranteed because verified subject IDs are unavailable",
    }
    (report_dir / "acne04_conversion_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = "# ACNE04 Pascal VOC to YOLO Conversion Report\n\n" + "\n".join(
        (f"- Source images: {report['source_images']}", f"- Source objects: {report['source_objects']}",
         f"- Converted images: {report['converted_images']}", f"- Converted objects: {report['converted_objects']}",
         f"- Conversion failures: {report['conversion_failures']}",
         f"- Duplicate groups crossing splits: {report['duplicate_groups_crossing_splits']}",
         f"- Seed: {seed}", "- Person-level separation cannot be guaranteed because verified subject IDs are unavailable.", "")
    )
    (report_dir / "acne04_conversion_report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    """Preflight, convert, copy, and split ACNE04 without altering raw sources."""
    args = parse_args()
    for name, path in (("Annotations directory", args.annotations_dir), ("Images directory", args.images_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not args.audit_json.is_file():
        raise FileNotFoundError(f"Audit JSON does not exist: {args.audit_json}")
    ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    records, failures = load_records(args.annotations_dir, args.images_dir, args.audit_json)
    failures_path = write_failures(failures)
    if failures:
        raise RuntimeError(f"Conversion preflight failed for {len(failures)} files; see {failures_path}.")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.output_root}. Re-run with --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    assignments = assign_splits(
        [SplitItem(record.image_id, len(record.boxes), record.small_box_count, record.duplicate_group_id) for record in records],
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
    )
    rows = write_processed_dataset(records, assignments, args.output_root, ratios, args.seed)
    write_reports(rows, failures, ratios, args.seed)
    print(f"Converted images: {len(rows)}")
    print(f"Converted objects: {sum(int(row['object_count']) for row in rows)}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
