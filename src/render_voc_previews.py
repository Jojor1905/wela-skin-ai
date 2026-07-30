"""Render deterministic, read-only visual previews of Pascal VOC annotations.

This script reads local annotation and image files and writes annotated copies
to an output directory.  It does not change raw data, labels, splits, or
duplicate images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DUPLICATE_GROUP_PATTERN = re.compile(r"Duplicate SHA-256 group: ([0-9a-f]{64})\.")


@dataclass(frozen=True)
class Box:
    """One valid Pascal VOC object box."""

    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return max(self.width / self.height, self.height / self.width)


@dataclass
class ImageRecord:
    """An image and its source XML information for preview generation."""

    image_id: str
    image_path: Path
    xml_path: Path
    boxes: list[Box]
    split_lists: list[str] = field(default_factory=list)
    duplicate_group_id: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def object_count(self) -> int:
        return len(self.boxes)

    @property
    def small_box_count(self) -> int:
        return sum(box.width <= 10 or box.height <= 10 for box in self.boxes)


def parse_args() -> argparse.Namespace:
    """Parse preview-rendering command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def project_relative(path: Path) -> str:
    """Return a portable path relative to the current project directory."""
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def xml_text(element: ElementTree.Element, tag: str) -> str | None:
    """Read stripped XML text without treating absent fields as empty strings."""
    value = element.findtext(tag)
    return value.strip() if value is not None else None


def resolve_image(root: ElementTree.Element, images_by_name: dict[str, Path], images_by_stem: dict[str, Path]) -> Path | None:
    """Resolve a VOC image from filename, path basename, then supported stem."""
    candidates = [xml_text(root, "filename"), xml_text(root, "path")]
    for candidate in candidates:
        if candidate:
            exact = images_by_name.get(Path(candidate).name)
            if exact is not None:
                return exact
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.suffix.lower() in IMAGE_SUFFIXES:
            stem_match = images_by_stem.get(candidate_path.stem)
            if stem_match is not None:
                return stem_match
    return None


def parse_boxes(root: ElementTree.Element, xml_path: Path) -> list[Box]:
    """Read valid boxes from an XML file; invalid values halt preview creation."""
    boxes: list[Box] = []
    for index, object_element in enumerate(root.findall("object"), start=1):
        class_name = xml_text(object_element, "name")
        bndbox = object_element.find("bndbox")
        if not class_name or bndbox is None:
            raise ValueError(f"Invalid object {index} in {xml_path}: missing class name or bounding box.")
        try:
            xmin, ymin, xmax, ymax = (
                float(xml_text(bndbox, coordinate) or "")
                for coordinate in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError as error:
            raise ValueError(f"Invalid coordinates in object {index} of {xml_path}.") from error
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Non-positive bounding box in object {index} of {xml_path}.")
        boxes.append(Box(class_name, xmin, ymin, xmax, ymax))
    return boxes


def load_records(annotations_dir: Path, images_dir: Path) -> dict[str, ImageRecord]:
    """Load all XMLs and resolve their images without modifying either directory."""
    image_paths = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    images_by_name = {path.name: path for path in image_paths}
    images_by_stem = {path.stem: path for path in image_paths}
    records: dict[str, ImageRecord] = {}
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        root = ElementTree.parse(xml_path).getroot()
        image_path = resolve_image(root, images_by_name, images_by_stem)
        if image_path is None:
            raise FileNotFoundError(f"No source image resolves for annotation: {xml_path}")
        image_id = xml_path.stem
        if image_id in records:
            raise ValueError(f"Duplicate XML image ID: {image_id}")
        records[image_id] = ImageRecord(image_id, image_path, xml_path, parse_boxes(root, xml_path))
    return records


def load_duplicate_groups(audit_json: Path, records: dict[str, ImageRecord]) -> dict[str, list[str]]:
    """Read audit SHA-256 duplicate groups and attach them to image records."""
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = defaultdict(list)
    image_id_by_name = {record.image_path.name: image_id for image_id, record in records.items()}
    for finding in audit.get("findings", {}).get("duplicate_images", []):
        match = DUPLICATE_GROUP_PATTERN.fullmatch(finding.get("detail", ""))
        if match is None:
            raise ValueError("Audit duplicate finding does not contain the expected SHA-256 group ID.")
        image_id = image_id_by_name.get(Path(finding["path"]).name)
        if image_id is None:
            raise ValueError(f"Duplicate audit finding cannot be resolved to an image: {finding['path']}")
        group_id = match.group(1)
        groups[group_id].append(image_id)
        records[image_id].duplicate_group_id = group_id
    return {group_id: sorted(image_ids) for group_id, image_ids in sorted(groups.items())}


def attach_split_lists(annotations_dir: Path, records: dict[str, ImageRecord]) -> None:
    """Attach ImageSets/Main membership where split-list files are available."""
    split_dir = annotations_dir.parent / "ImageSets" / "Main"
    if not split_dir.is_dir():
        return
    for split_path in sorted(split_dir.glob("*.txt")):
        for line in split_path.read_text(encoding="utf-8").splitlines():
            image_reference = line.strip().split()[0] if line.strip() else ""
            image_id = Path(image_reference).stem
            if image_id in records:
                records[image_id].split_lists.append(split_path.name)


def add_selection(selected: dict[str, ImageRecord], record: ImageRecord, reason: str, sample_size: int) -> None:
    """Add a required record or explain why the requested sample is too small."""
    if record.image_id not in selected and len(selected) >= sample_size:
        raise ValueError(
            f"--sample-size {sample_size} cannot cover all required sampling categories; increase it and rerun."
        )
    selected[record.image_id] = record
    if reason not in record.reasons:
        record.reasons.append(reason)


def select_records(records: dict[str, ImageRecord], duplicate_groups: dict[str, list[str]], sample_size: int, seed: int) -> list[ImageRecord]:
    """Select duplicate, risk-focused, split-aware, and seeded-random previews."""
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    selected: dict[str, ImageRecord] = {}

    # Include every member, not merely one representative, to permit visual comparison.
    for group_id, image_ids in duplicate_groups.items():
        for image_id in image_ids:
            add_selection(selected, records[image_id], f"duplicate_group:{group_id[:12]}", sample_size)

    by_object_count = sorted(records.values(), key=lambda record: (-record.object_count, record.image_id))
    for record in by_object_count[:10]:
        add_selection(selected, record, "highest_object_count", sample_size)

    by_small_boxes = sorted(records.values(), key=lambda record: (-record.small_box_count, record.image_id))
    for record in by_small_boxes[:10]:
        if record.small_box_count:
            add_selection(selected, record, "contains_very_small_box", sample_size)

    by_largest_box = sorted(
        records.values(), key=lambda record: (-max(box.area for box in record.boxes), record.image_id)
    )
    for record in by_largest_box[:5]:
        add_selection(selected, record, "unusual_large_box", sample_size)
    by_aspect_ratio = sorted(
        records.values(), key=lambda record: (-max(box.aspect_ratio for box in record.boxes), record.image_id)
    )
    for record in by_aspect_ratio[:5]:
        add_selection(selected, record, "unusual_aspect_ratio", sample_size)

    split_names = sorted({split_name for record in records.values() for split_name in record.split_lists})
    for split_name in split_names:
        candidates = sorted(
            (record for record in records.values() if split_name in record.split_lists),
            key=lambda record: (record.image_id in selected, record.image_id),
        )
        if candidates:
            add_selection(selected, candidates[0], f"split_list:{split_name}", sample_size)

    remaining = [record for image_id, record in sorted(records.items()) if image_id not in selected]
    random_generator = random.Random(seed)
    random_generator.shuffle(remaining)
    if remaining and len(selected) < sample_size:
        add_selection(selected, remaining.pop(0), "seeded_random", sample_size)
    while remaining and len(selected) < sample_size:
        add_selection(selected, remaining.pop(0), "seeded_random", sample_size)

    if not any("seeded_random" in record.reasons for record in selected.values()):
        raise ValueError("--sample-size leaves no room for the required seeded-random sample; increase it and rerun.")
    return sorted(selected.values(), key=lambda record: record.image_id)


def load_font(image_size: tuple[int, int]) -> ImageFont.ImageFont:
    """Use a scalable common font when available, with Pillow's fallback otherwise."""
    font_size = max(16, min(36, min(image_size) // 45))
    for font_path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def render_preview(record: ImageRecord, preview_path: Path) -> None:
    """Draw all source boxes on an unscaled copy with a separate metadata legend."""
    with Image.open(record.image_path) as source_image:
        preview = source_image.convert("RGB")
    draw = ImageDraw.Draw(preview)
    font = load_font(preview.size)
    outline_width = max(2, min(8, min(preview.size) // 450))
    for box in record.boxes:
        coordinates = (box.xmin, box.ymin, box.xmax, box.ymax)
        draw.rectangle(coordinates, outline=(255, 40, 40), width=outline_width)
        if box.width <= 10 or box.height <= 10:
            center_x = (box.xmin + box.xmax) / 2
            center_y = (box.ymin + box.ymax) / 2
            radius = max(5, outline_width + 2)
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=(255, 255, 0), width=outline_width)

    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 6
    legend_width = max(280, min(520, preview.width // 2))
    legend_height = max(preview.height, 44 + line_height * (record.object_count + 1))
    rendered = Image.new("RGB", (preview.width + legend_width, legend_height), "white")
    rendered.paste(preview, (0, 0))
    legend = ImageDraw.Draw(rendered)
    legend.text((preview.width + 14, 14), f"objects: {record.object_count}", fill="black", font=font)
    for index, box in enumerate(record.boxes, start=1):
        legend.text(
            (preview.width + 14, 26 + line_height * index),
            f"#{index}  {box.class_name}  {box.width:g} x {box.height:g}",
            fill="black",
            font=font,
        )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(preview_path, format="JPEG", quality=95, subsampling=0)


def annotation_fingerprint(record: ImageRecord) -> str:
    """Hash ordered class and box coordinates for duplicate-group comparison."""
    source = "|".join(
        f"{box.class_name}:{box.xmin:g},{box.ymin:g},{box.xmax:g},{box.ymax:g}" for box in record.boxes
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def write_outputs(records: list[ImageRecord], duplicate_groups: dict[str, list[str]], output_dir: Path) -> None:
    """Render previews and write the review index, guide, and duplicate worksheet."""
    images_output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "preview_index.csv"
    if index_path.is_file():
        with index_path.open(newline="", encoding="utf-8") as previous_index:
            for row in csv.DictReader(previous_index):
                previous_preview = Path(row.get("preview_path", ""))
                if previous_preview.parent == Path("images") and previous_preview.suffix.lower() == ".jpg":
                    (images_output_dir / previous_preview.name).unlink(missing_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as index_file:
        fieldnames = [
            "image_id", "relative_source_image_path", "relative_xml_path", "preview_path",
            "object_count", "small_box_count", "duplicate_group_id", "sampling_reason",
            "review_status", "reviewer_notes",
        ]
        writer = csv.DictWriter(index_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            preview_path = images_output_dir / f"{record.image_id}.jpg"
            render_preview(record, preview_path)
            writer.writerow({
                "image_id": record.image_id,
                "relative_source_image_path": project_relative(record.image_path),
                "relative_xml_path": project_relative(record.xml_path),
                "preview_path": (Path("images") / preview_path.name).as_posix(),
                "object_count": record.object_count,
                "small_box_count": record.small_box_count,
                "duplicate_group_id": record.duplicate_group_id,
                "sampling_reason": ";".join(record.reasons),
                "review_status": "Pending",
                "reviewer_notes": "",
            })

    records_by_id = {record.image_id: record for record in records}
    duplicate_path = output_dir / "duplicate_review.csv"
    with duplicate_path.open("w", newline="", encoding="utf-8") as duplicate_file:
        fieldnames = [
            "duplicate_group_id", "image_id", "relative_source_image_path", "relative_xml_path",
            "preview_path", "annotation_fingerprint", "group_fingerprints_identical", "review_status", "reviewer_notes",
        ]
        writer = csv.DictWriter(duplicate_file, fieldnames=fieldnames)
        writer.writeheader()
        for group_id, image_ids in duplicate_groups.items():
            group_records = [records_by_id[image_id] for image_id in image_ids]
            fingerprints = {annotation_fingerprint(record) for record in group_records}
            for record in group_records:
                writer.writerow({
                    "duplicate_group_id": group_id,
                    "image_id": record.image_id,
                    "relative_source_image_path": project_relative(record.image_path),
                    "relative_xml_path": project_relative(record.xml_path),
                    "preview_path": (Path("images") / f"{record.image_id}.jpg").as_posix(),
                    "annotation_fingerprint": annotation_fingerprint(record),
                    "group_fingerprints_identical": "yes" if len(fingerprints) == 1 else "no",
                    "review_status": "Pending",
                    "reviewer_notes": "",
                })

    (output_dir / "REVIEW_GUIDE.md").write_text(
        """# Visual Annotation Review Guide

These previews are a read-only review aid. They do not validate a medical diagnosis, change source data, or approve a label mapping.

## Review each preview

1. Confirm each red box covers a visible acne lesion. Inspect the yellow centre marker used for boxes whose width or height is 10 pixels or less; the preview remains at the source image dimensions so small lesions are not resized away.
2. Flag boxes that instead point to pores, moles, shadows, scars, highlights, hair, or other non-acne areas.
3. Check the displayed `fore` label and dimensions against the visible target. The source label has not been medically validated.
4. For each row in `duplicate_review.csv`, compare all previews in its duplicate group and determine whether visually identical images have identical annotations. The fingerprint is an aid for exact source-box comparison, not a substitute for visual review.

## Record a decision

Use `preview_index.csv` and `duplicate_review.csv` to enter one of these statuses:

- `Pass`: boxes and source-label usage appear consistent with the intended visible-lesion research task.
- `Needs review`: there is ambiguity, a difficult image, a very small target, or a discrepancy requiring a second reviewer.
- `Reject`: boxes clearly do not support the intended visible-acne-lesion label or the annotation is materially unusable.

Add concise evidence in `reviewer_notes`; do not add identity information or reproduce source images outside the approved local workspace.
""",
        encoding="utf-8",
    )


def main() -> None:
    """Run the deterministic, read-only visual annotation review generation."""
    args = parse_args()
    for label, directory in (("annotations", args.annotations_dir), ("images", args.images_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label.capitalize()} directory does not exist: {directory}")
    if not args.audit_json.is_file():
        raise FileNotFoundError(f"Audit JSON does not exist: {args.audit_json}")
    records = load_records(args.annotations_dir, args.images_dir)
    duplicate_groups = load_duplicate_groups(args.audit_json, records)
    attach_split_lists(args.annotations_dir, records)
    selected = select_records(records, duplicate_groups, args.sample_size, args.seed)
    write_outputs(selected, duplicate_groups, args.output_dir)
    print(f"Previews generated: {len(selected)}")
    print(f"Duplicate groups included: {len(duplicate_groups)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
