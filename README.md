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
