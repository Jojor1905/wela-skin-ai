"""Render read-only previews from converted YOLO TXT labels, not source XML."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(dataset_root: Path) -> list[dict[str, str]]:
    with (dataset_root / "split_manifest.csv").open(newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def add(selected: dict[str, dict[str, str]], row: dict[str, str], reason: str, limit: int) -> None:
    if row["image_id"] not in selected and len(selected) >= limit:
        return
    selected[row["image_id"]] = row
    row.setdefault("_reasons", "")
    reasons = row["_reasons"].split(";") if row["_reasons"] else []
    if reason not in reasons:
        reasons.append(reason)
    row["_reasons"] = ";".join(reasons)


def select(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    selected: dict[str, dict[str, str]] = {}
    duplicates = [row for row in rows if row["duplicate_group_id"]]
    for row in sorted(duplicates, key=lambda item: (item["duplicate_group_id"], item["image_id"]))[:12]:
        add(selected, row, "duplicate_group", sample_size)
    for row in sorted(rows, key=lambda item: (-int(item["object_count"]), item["image_id"]))[:10]:
        add(selected, row, "many_objects", sample_size)
    for row in sorted(rows, key=lambda item: (-int(item["small_box_count"]), item["image_id"]))[:10]:
        if int(row["small_box_count"]):
            add(selected, row, "contains_very_small_box", sample_size)
    random_generator = random.Random(seed)
    for split in ("train", "val", "test"):
        candidates = [row for row in rows if row["split"] == split and row["image_id"] not in selected]
        random_generator.shuffle(candidates)
        for row in candidates[:6]:
            add(selected, row, f"random_{split}", sample_size)
    remaining = [row for row in rows if row["image_id"] not in selected]
    random_generator.shuffle(remaining)
    for row in remaining:
        if len(selected) >= sample_size:
            break
        add(selected, row, "seeded_random", sample_size)
    return sorted(selected.values(), key=lambda item: item["image_id"])


def render(row: dict[str, str], dataset_root: Path, output_path: Path) -> str:
    """Reconstruct pixel boxes from TXT labels and return an optional mismatch detail."""
    image_path = dataset_root / row["image_path"]
    label_path = dataset_root / row["label_path"]
    source_path = Path(row["source_image_path"])
    with Image.open(image_path) as image:
        preview = image.convert("RGB")
        width, height = preview.size
    mismatch = ""
    if source_path.is_file():
        with Image.open(source_path) as source_image:
            if source_image.size != (width, height):
                mismatch = f"Source dimensions {source_image.size} differ from converted image dimensions {(width, height)}."
    draw = ImageDraw.Draw(preview)
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        class_id, x_center, y_center, box_width, box_height = line.split()
        if class_id != "0":
            mismatch = mismatch or f"Unexpected class ID on line {line_number}: {class_id}."
        xc, yc, bw, bh = (float(value) for value in (x_center, y_center, box_width, box_height))
        xmin, ymin = (xc - bw / 2) * width, (yc - bh / 2) * height
        xmax, ymax = (xc + bw / 2) * width, (yc + bh / 2) * height
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
            mismatch = mismatch or f"Reconstructed YOLO box on line {line_number} is outside image dimensions."
        draw.rectangle((xmin, ymin, xmax, ymax), outline=(30, 220, 60), width=max(2, min(width, height) // 450))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, format="JPEG", quality=95, subsampling=0)
    return mismatch


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_root)
    selected = select(rows, args.sample_size, args.seed)
    if len(selected) != args.sample_size:
        raise RuntimeError(f"Only selected {len(selected)} previews; requested {args.sample_size}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_index = args.output_dir / "yolo_preview_index.csv"
    if previous_index.is_file():
        with previous_index.open(newline="", encoding="utf-8") as index_file:
            for row in csv.DictReader(index_file):
                preview_path = Path(row.get("preview_path", ""))
                if preview_path.parent == Path("images") and preview_path.suffix.lower() == ".jpg":
                    (args.output_dir / "images" / preview_path.name).unlink(missing_ok=True)
    output_rows: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    for row in selected:
        preview_path = args.output_dir / "images" / f"{row['image_id']}.jpg"
        mismatch = render(row, args.dataset_root, preview_path)
        output_rows.append({"image_id": row["image_id"], "split": row["split"], "preview_path": (Path("images") / preview_path.name).as_posix(), "sampling_reason": row["_reasons"], "dimension_or_box_mismatch": mismatch})
        if mismatch:
            mismatches.append(output_rows[-1])
    for file_name, fieldnames, contents in (
        ("yolo_preview_index.csv", list(output_rows[0]), output_rows),
        ("yolo_preview_mismatches.csv", list(output_rows[0]), mismatches),
    ):
        with (args.output_dir / file_name).open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(contents)
    if mismatches:
        raise RuntimeError(f"YOLO preview dimension or box mismatch count: {len(mismatches)}")
    print(f"YOLO previews generated: {len(output_rows)}")
    print("YOLO preview mismatches: 0")


if __name__ == "__main__":
    main()
