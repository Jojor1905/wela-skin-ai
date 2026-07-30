"""Create a read-only audit report for a local Pascal VOC-style dataset.

The audit never changes dataset files.  It writes JSON and CSV reports to the
selected output directory so annotation issues can be reviewed before any
conversion, split, or model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REFERENCE_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class Finding:
    """A single audit finding, with a path relative to the dataset root."""

    path: str
    detail: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a read-only dataset audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Pascal VOC-style dataset directory.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Optional image directory. Defaults to dataset-root/JPEGImages.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated audit reports.")
    parser.add_argument(
        "--small-box-max-side",
        type=int,
        default=10,
        help="Report boxes with width or height at most this many pixels (default: 10).",
    )
    return parser.parse_args()


def relative_path(path: Path, dataset_root: Path) -> str:
    """Return a portable path relative to the VOC root, including sibling directories."""
    return Path(os.path.relpath(path, dataset_root)).as_posix()


def find_images(images_dir: Path) -> list[Path]:
    """Find supported images below the explicitly selected image directory."""
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_image_reference(
    root: ElementTree.Element,
    images_by_name: dict[str, list[Path]],
    images_by_stem: dict[str, list[Path]],
) -> tuple[Path | None, str | None, str | None]:
    """Resolve a VOC image reference and identify an extension-only mismatch.

    Resolution order is XML ``filename``, the basename of XML ``path``, then
    either reference stem with supported JPEG/PNG extensions.  The returned
    mismatch description is populated only when stem fallback was required.
    """
    filename = text_value(root, "filename")
    xml_path = text_value(root, "path")
    candidates = [
        ("<filename>", Path(filename).name) for filename in [filename] if filename
    ]
    if xml_path:
        candidates.append(("<path>", Path(xml_path).name))

    for _, candidate in candidates:
        exact_matches = images_by_name.get(candidate, [])
        if exact_matches:
            return exact_matches[0], None, None

    for source, candidate in candidates:
        suffix = Path(candidate).suffix.lower()
        stem = Path(candidate).stem
        if suffix not in REFERENCE_IMAGE_SUFFIXES or not stem:
            continue
        stem_matches = images_by_stem.get(stem, [])
        if stem_matches:
            resolved = stem_matches[0]
            return (
                resolved,
                f"{source} requests {candidate}, resolved by stem to {resolved.name}.",
                None,
            )

    descriptions = [f"{source}={candidate}" for source, candidate in candidates]
    return None, None, "; ".join(descriptions) if descriptions else None


def text_value(parent: ElementTree.Element, tag: str) -> str | None:
    """Read a required-looking XML text field without coercing missing values."""
    value = parent.findtext(tag)
    return value.strip() if value is not None else None


def parse_number(value: str | None) -> float | None:
    """Parse a finite coordinate value, retaining invalid source values as None."""
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_positive_integer(value: str | None) -> int | None:
    """Parse a positive integer dimension from the VOC XML metadata."""
    number = parse_number(value)
    if number is None or not number.is_integer() or number <= 0:
        return None
    return int(number)


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest while reading the file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[Finding]) -> None:
    """Write a deterministic, relative-path CSV findings report."""
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["path", "detail"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def audit(dataset_root: Path, images_dir: Path, small_box_max_side: int) -> dict[str, Any]:
    """Inspect VOC XML and image files without modifying the dataset."""
    annotation_dir = dataset_root / "Annotations"
    xml_paths = sorted(annotation_dir.glob("*.xml")) if annotation_dir.is_dir() else []
    image_paths = find_images(images_dir)
    images_by_name: dict[str, list[Path]] = defaultdict(list)
    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_paths:
        images_by_name[image_path.name].append(image_path)
        if image_path.suffix.lower() in REFERENCE_IMAGE_SUFFIXES:
            images_by_stem[image_path.stem].append(image_path)

    invalid_annotations: list[Finding] = []
    invalid_boxes: list[Finding] = []
    small_boxes: list[Finding] = []
    missing_images: list[Finding] = []
    extension_mismatches: list[Finding] = []
    referenced_images: set[Path] = set()
    class_counts: Counter[str] = Counter()
    annotated_objects = 0

    for xml_path in xml_paths:
        xml_relative = relative_path(xml_path, dataset_root)
        try:
            root = ElementTree.parse(xml_path).getroot()
        except (ElementTree.ParseError, OSError) as error:
            invalid_annotations.append(Finding(xml_relative, f"XML cannot be parsed: {error}"))
            continue

        resolved_image, extension_mismatch, reference_description = resolve_image_reference(
            root, images_by_name, images_by_stem
        )
        if resolved_image is None:
            detail = "No XML <filename> or <path> filename is available."
            if reference_description:
                detail = f"Referenced image is missing: {reference_description}."
            else:
                invalid_annotations.append(Finding(xml_relative, detail))
            missing_images.append(Finding(xml_relative, detail))
        else:
            referenced_images.add(resolved_image)
            if extension_mismatch:
                extension_mismatches.append(Finding(xml_relative, extension_mismatch))

        size = root.find("size")
        width = parse_positive_integer(text_value(size, "width") if size is not None else None)
        height = parse_positive_integer(text_value(size, "height") if size is not None else None)
        if width is None or height is None:
            invalid_annotations.append(Finding(xml_relative, "Missing or invalid positive <size>/<width> or <height>."))

        for index, object_element in enumerate(root.findall("object"), start=1):
            annotated_objects += 1
            class_name = text_value(object_element, "name")
            if not class_name:
                invalid_annotations.append(Finding(xml_relative, f"Object {index} has no valid <name>."))
            else:
                class_counts[class_name] += 1

            bndbox = object_element.find("bndbox")
            coordinates = {
                coordinate: parse_number(text_value(bndbox, coordinate) if bndbox is not None else None)
                for coordinate in ("xmin", "ymin", "xmax", "ymax")
            }
            if any(value is None for value in coordinates.values()):
                invalid_boxes.append(Finding(xml_relative, f"Object {index} has missing, non-numeric, or non-finite coordinates."))
                continue

            xmin = coordinates["xmin"]
            ymin = coordinates["ymin"]
            xmax = coordinates["xmax"]
            ymax = coordinates["ymax"]
            assert xmin is not None and ymin is not None and xmax is not None and ymax is not None
            problems: list[str] = []
            if xmin < 0 or ymin < 0:
                problems.append("minimum coordinate is negative")
            if xmax <= xmin or ymax <= ymin:
                problems.append("box has non-positive width or height")
            if width is not None and xmax > width:
                problems.append(f"xmax {xmax:g} exceeds XML width {width}")
            if height is not None and ymax > height:
                problems.append(f"ymax {ymax:g} exceeds XML height {height}")
            if problems:
                invalid_boxes.append(Finding(xml_relative, f"Object {index}: {'; '.join(problems)}."))
                continue

            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= small_box_max_side or box_height <= small_box_max_side:
                small_boxes.append(
                    Finding(
                        xml_relative,
                        f"Object {index}: width={box_width:g}, height={box_height:g}; threshold={small_box_max_side}px.",
                    )
                )

    missing_xml = [
        Finding(relative_path(image_path, dataset_root), "No XML annotation references this image.")
        for image_path in image_paths
        if image_path not in referenced_images
    ]

    corrupt_images: list[Finding] = []
    digest_paths: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_paths:
        image_relative = relative_path(image_path, dataset_root)
        try:
            with Image.open(image_path) as image:
                image.verify()
            digest_paths[file_digest(image_path)].append(image_path)
        except (OSError, UnidentifiedImageError) as error:
            corrupt_images.append(Finding(image_relative, f"Image cannot be opened and verified: {error}"))

    duplicate_images = [
        Finding(relative_path(path, dataset_root), f"Duplicate SHA-256 group: {digest}.")
        for digest, paths in sorted(digest_paths.items())
        if len(paths) > 1
        for path in paths
    ]

    return {
        "summary": {
            "total_images": len(image_paths),
            "total_xml_annotations": len(xml_paths),
            "total_annotated_objects": annotated_objects,
            "unique_object_classes": sorted(class_counts),
            "object_class_counts": dict(sorted(class_counts.items())),
            "invalid_annotations": len(invalid_annotations),
            "invalid_bounding_boxes": len(invalid_boxes),
            "missing_images_for_xml": len(missing_images),
            "missing_xml_for_images": len(missing_xml),
            "filename_extension_mismatches": len(extension_mismatches),
            "corrupt_images": len(corrupt_images),
            "duplicate_images": len(duplicate_images),
            "very_small_bounding_boxes": len(small_boxes),
            "very_small_box_definition": f"width or height <= {small_box_max_side} pixels",
        },
        "findings": {
            "invalid_annotations": [asdict(finding) for finding in invalid_annotations],
            "invalid_bounding_boxes": [asdict(finding) for finding in invalid_boxes],
            "missing_images_for_xml": [asdict(finding) for finding in missing_images],
            "missing_xml_for_images": [asdict(finding) for finding in missing_xml],
            "filename_extension_mismatches": [asdict(finding) for finding in extension_mismatches],
            "corrupt_images": [asdict(finding) for finding in corrupt_images],
            "duplicate_images": [asdict(finding) for finding in duplicate_images],
            "very_small_bounding_boxes": [asdict(finding) for finding in small_boxes],
        },
    }


def main() -> None:
    """Run the requested audit and write reports outside the dataset directory."""
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")
    images_dir = args.images_dir if args.images_dir is not None else args.dataset_root / "JPEGImages"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if args.small_box_max_side < 0:
        raise ValueError("--small-box-max-side must be zero or greater.")

    report = audit(args.dataset_root, images_dir, args.small_box_max_side)
    report["image_directory"] = relative_path(images_dir, args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "acne04_dataset_audit.json"
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")

    findings = report["findings"]
    for finding_name, finding_rows in findings.items():
        write_csv(
            args.output_dir / f"acne04_{finding_name}.csv",
            [Finding(**row) for row in finding_rows],
        )

    summary = report["summary"]
    print("Dataset audit complete.")
    for field, value in summary.items():
        print(f"{field}: {value}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
