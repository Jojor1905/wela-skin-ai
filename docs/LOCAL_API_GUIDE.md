# Local Inference API Guide

## Scope and safety

This FastAPI service is a UI-integration prototype for the repository's single `acne_lesion` detection class. It is not a medical device, diagnosis, treatment tool, or production system. Uploaded images are decoded in memory, corrected using EXIF orientation, passed directly to the model, and closed when inference ends. The API does not retain uploads.

The detector does not identify dark circles, acne scars, pigmentation, pores, wrinkles, dryness, oiliness, or sensitivity. Those self-reported concerns can only influence cosmetic product-category suggestions.

## Install

Use the repository's Python 3.11 environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default model location is:

```text
models/acne-yolo-best.pt
```

Override runtime settings when needed:

```bash
export MODEL_PATH=models/acne-yolo-best.pt
export YOLO_DEVICE=cpu
export ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000
```

`MODEL_PATH` is resolved relative to the repository root unless it is absolute, and it is never returned by an endpoint. `YOLO_DEVICE` defaults to `cpu`. `ALLOWED_ORIGINS` is a comma-separated list whose defaults are the three development origins above.

## Run

From the repository root:

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Startup loads the model once. If the configured weights are missing or do not report exactly one class, startup fails with an actionable error.

## Exact local API test steps

Keep the API running in the first terminal. In a second terminal, run:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/model-info
```

Test a local authorised JPEG, PNG, or WEBP file (10 MB maximum):

```bash
curl -sS -X POST http://127.0.0.1:8000/predict \
  -F 'image=@/absolute/path/to/authorised-face-image.jpg;type=image/jpeg' \
  -F 'gender=prefer not to say' \
  -F 'ageRange=25-34' \
  -F 'skinType=combination' \
  -F 'concerns=["breakouts","oiliness"]' \
  -F 'goal=build a simple balanced routine'
```

The `concerns` value may instead be comma-separated:

```bash
-F 'concerns=breakouts,oiliness'
```

OpenAPI documentation is available locally at `http://127.0.0.1:8000/docs`.

Run all unit tests without invoking training or locked-test evaluation:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run import and syntax checks:

```bash
python -m compileall -q src tests
python -c 'from src.api.app import app; print(app.title)'
```

## Response interpretation

`/predict` returns request provenance (request ID, a short SHA-256 prefix, and raw/post-threshold detection counts), oriented image dimensions, class-0 boxes and confidence values, normalized coordinates, approximate image-relative region counts, a count-based prototype breakout label, a UI-only prototype skin score, conservative insights, and cosmetic product categories. Region labels are coarse coordinate zones; they are not facial-landmark detections. Questionnaire answers do not create additional visual detections. Prediction responses use `Cache-Control: no-store`.

For a three-image endpoint/direct-model comparison against the unchanged configured weights, run the API and then execute:

```bash
.venv/bin/python scripts/verify_prediction_pipeline.py
```

Every prediction includes this notice:

> Experimental visual analysis for prototype demonstration only. Results may be incomplete or inaccurate and are not a medical diagnosis.
