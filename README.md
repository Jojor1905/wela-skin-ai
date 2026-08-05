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

The repository contains the existing `models/acne-yolo-best.pt` weights for academic-prototype inference. This deployment work does not retrain the model or change weights, datasets, annotations, splits, or evaluation results. Dataset provenance and limitations remain documented in `docs/DATASET_CARD.md` and `docs/LICENSE_LOG.md`.

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

## Local Apple Silicon Pilot

The guarded local MPS pilot workflow is documented in [LOCAL_PILOT_GUIDE.md](docs/LOCAL_PILOT_GUIDE.md). It deliberately uses only training and validation data, keeps artifacts under ignored `outputs/experiments/`, and must not be used for external-cloud training without explicit permission.

## Local Prototype API

The local FastAPI inference service and exact installation, run, curl, unit-test, and import-check commands are documented in [LOCAL_API_GUIDE.md](docs/LOCAL_API_GUIDE.md). It loads the existing one-class model once, does not retain uploaded images by default, and returns non-medical UI-prototype output only.

Install the API dependencies and start it from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

The API decodes each uploaded JPEG, PNG, or WEBP image in memory, runs inference on that request's image, and closes it after inference. It does not permanently save uploaded user images. Logs contain a request ID, byte count, short SHA-256 prefix, dimensions, and detection counts; they do not contain image data or complete hashes.

## Render Web Service

The root `render.yaml` defines the public Python Web Service. Its commands are:

```text
Build: python -m pip install --upgrade pip && pip install -r requirements.txt
Start: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

Configure these non-secret environment variables (the same values are declared in `render.yaml`):

| Variable | Value | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/acne-yolo-best.pt` | Repository-relative trained weights path |
| `YOLO_DEVICE` | `cpu` | CPU-only inference on Render |
| `ALLOWED_ORIGINS` | `https://wela-liff-prototype.vercel.app,http://localhost:3000,http://localhost:3001` | Comma-separated browser origins |
| `LOG_LEVEL` | `INFO` | Privacy-safe application logging level |

After Render assigns the service hostname, the public endpoints are:

- Health: `https://<render-service-host>/health`
- OpenAPI documentation: `https://<render-service-host>/docs`

A healthy service returns `{"status":"ok","model_loaded":true}` from `/health`. A missing or unloadable model returns an unhealthy response and is never reported as loaded. Verify the checked-in model, CPU loading, import, and health endpoint without modifying model or dataset files:

```bash
python scripts/verify_render_deployment.py
```

Ultralytics declares PyTorch, Torchvision, NumPy, and `opencv-python` as transitive runtime dependencies. Therefore `requirements.txt` does not separately install `opencv-python-headless`, which would cause both OpenCV distributions to be installed. PyTorch and Ultralytics are comparatively large and may approach free-tier build-time, memory, or cold-start limits; CPU inference can also be slow.

## Research and Medical Disclaimer

This public service remains an academic feasibility prototype. It is not a medical device, does not diagnose acne or any health condition, must not guide treatment or clinical decisions, and makes no clinical-accuracy or production-readiness claim. It does not perform face recognition, identity matching, or identity inference.
