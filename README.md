# Wela Skin AI: Acne Lesion Detection Prototype

## Objective

This academic feasibility prototype explores whether a pretrained YOLO object-detection model can locate visible acne lesions in a single facial photograph. The MVP has one class, `acne_lesion`, and is intended to return bounding boxes, confidence values, and a lesion count.

It is research-only: it is not medical diagnosis, does not claim clinical accuracy, and is not ready for commercial or production use. The project must not perform face recognition, identity matching, or identity inference.

## Planned Workflow

1. Record dataset provenance, licensing, intended use, and limitations in `docs/`.
2. Audit locally supplied data and review label guidance.
3. Convert annotations and create person-level splits where subject information exists.
4. Validate YOLO labels, then define a reproducible baseline experiment.
5. Evaluate detection metrics and document limitations honestly.

Do not download data, install packages, or train a model until the dataset audit and licensing documentation are complete.

## Local Data and Privacy

Place authorised data only on the local machine under `data/raw/`; derived files may use `data/interim/` and `data/processed/`. These directories, annotations, model weights, credentials, and generated model artifacts are excluded from Git. Never add facial images or personally identifying information to issues, pull requests, or documentation.

## Structure

- `configs/` — example dataset configuration.
- `docs/` — dataset, licence, labelling, experiment, and model documentation.
- `notebooks/` — planning notebooks with no results or data.
- `src/` — command-line script placeholders for the planned pipeline.
- `tests/` — future automated tests.
- `outputs/` — local metrics, predictions, and reports.

## Current Status

No dataset has been downloaded, no dependencies have been installed, and no model has been trained. The next manual action is to identify an appropriate dataset and complete `docs/LICENSE_LOG.md` and `docs/DATASET_CARD.md` before placing any authorised data locally.

## Dataset Audit

After local dataset access and governance approval, run the read-only VOC audit with:

```bash
python src/audit_dataset.py \
  --dataset-root data/raw/acne04/Detection/VOC2007 \
  --images-dir data/raw/acne04/Classification/JPEGImages \
  --output-dir outputs/reports
```

`--images-dir` is optional; when it is omitted, the audit checks `JPEGImages` inside the VOC root. The audit requires Pillow and writes only JSON and CSV reports beneath the output directory. It does not alter raw data, convert annotations, create splits, or train a model.

## Visual Annotation Review

After a clean audit, generate deterministic, read-only annotation previews with:

```bash
python src/render_voc_previews.py \
  --annotations-dir data/raw/acne04/Detection/VOC2007/Annotations \
  --images-dir data/raw/acne04/Classification/JPEGImages \
  --audit-json outputs/reports/acne04_dataset_audit.json \
  --output-dir outputs/previews \
  --sample-size 120 \
  --seed 42
```

The generated previews and CSV worksheets remain local, and support human review before any label conversion or training.

## YOLO Conversion and Validation

After approval of the `fore` to `acne_lesion` research-prototype mapping, use the commands in [CONVERSION_GUIDE.md](docs/CONVERSION_GUIDE.md). Conversion copies local source images into the ignored `data/processed/` directory, creates duplicate-aware deterministic splits, and must be followed by validation. It does not train a model.

## Automated YOLO Preview Review

Run the read-only conversion-integrity review after rendering YOLO previews:

```bash
python src/automated_preview_review.py \
  --source-annotations data/raw/acne04/Detection/VOC2007/Annotations \
  --source-images data/raw/acne04/Classification/JPEGImages \
  --yolo-dataset data/processed/acne04_yolo \
  --preview-index outputs/yolo_previews/yolo_preview_index.csv \
  --split-manifest data/processed/acne04_yolo/split_manifest.csv \
  --output-dir outputs/yolo_previews
```

It compares source Pascal VOC boxes with reconstructed YOLO boxes and writes local review worksheets. It checks conversion integrity and split metadata only; it does not assess medical label accuracy, alter source or converted data, or train a model.

## Full Dataset Readiness Check

Run the complete read-only integrity check across every source XML/image and converted YOLO pair:

```bash
python src/full_dataset_readiness_check.py \
  --source-annotations data/raw/acne04/Detection/VOC2007/Annotations \
  --source-images data/raw/acne04/Classification/JPEGImages \
  --yolo-dataset data/processed/acne04_yolo \
  --split-manifest data/processed/acne04_yolo/split_manifest.csv \
  --output-dir outputs/readiness
```

It confirms conversion and split integrity, including duplicate hashes, without changing source or converted data. It is not clinical label validation and does not train a model.
