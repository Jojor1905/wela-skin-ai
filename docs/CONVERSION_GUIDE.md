# ACNE04 Pascal VOC to YOLO Conversion

## Scope and safeguards

This is a local academic-prototype conversion only. It copies audited images into `data/processed/acne04_yolo/` and never modifies `data/raw/`. The mapping is `fore` to class ID `0`, `acne_lesion`; it is not a medical classification and creates no acne subtypes.

## Source paths

- Annotations: `data/raw/acne04/Detection/VOC2007/Annotations`
- Images: `data/raw/acne04/Classification/JPEGImages`
- Duplicate audit: `outputs/reports/acne04_dataset_audit.json`

## Conversion formula

For a Pascal VOC box `(xmin, ymin, xmax, ymax)` and actual Pillow image dimensions `(W, H)`, the YOLO row is:

`0 ((xmin + xmax) / 2W) ((ymin + ymax) / 2H) ((xmax - xmin) / W) ((ymax - ymin) / H)`

Coordinates are validated against actual image dimensions and are never clamped or repaired.

## Reproduce conversion and split

```bash
python src/convert_voc_to_yolo.py \
  --annotations-dir data/raw/acne04/Detection/VOC2007/Annotations \
  --images-dir data/raw/acne04/Classification/JPEGImages \
  --audit-json outputs/reports/acne04_dataset_audit.json \
  --output-root data/processed/acne04_yolo \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

The split is deterministic and keeps every SHA-256 duplicate group together. It balances image, object, and very-small-box totals heuristically. It does not use `ImageSets/Main` as a final split; those files are source metadata only. Person-level separation cannot be guaranteed because verified subject IDs are unavailable.

## Validation and visual review

```bash
python src/validate_yolo_dataset.py \
  --dataset-root data/processed/acne04_yolo \
  --output-dir outputs/reports

python src/render_yolo_previews.py \
  --dataset-root data/processed/acne04_yolo \
  --output-dir outputs/yolo_previews \
  --sample-size 60 --seed 42
```

Validation must pass before any later research step. Neither command trains a model.
