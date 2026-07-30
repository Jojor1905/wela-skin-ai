# Repository Guidelines

## Purpose and Scope

This repository supports a research feasibility prototype for detecting visible acne lesions in facial photographs. It is not a medical device, diagnostic tool, production system, or face-recognition project. The MVP uses one detection class: `acne_lesion`.

## Non-Negotiable Data Rules

- Never commit facial images, annotations, model weights, or credentials.
- Before changing data-processing code, read `docs/DATASET_CARD.md`, `docs/LICENSE_LOG.md`, and `docs/LABEL_GUIDE.md`.
- Do not invent labels, metrics, licence terms, dataset properties, or medical claims.
- Do not train on data until its licence and intended use are documented in `docs/LICENSE_LOG.md` and `docs/DATASET_CARD.md`.
- Do not implement face recognition, identity matching, or identity inference.
- Keep subject information only where authorised and necessary; use person-level train/validation/test splits when subject information is available.

## Project Layout

Place Python code in `src/`, tests in `tests/`, documentation in `docs/`, and example configuration in `configs/`. Local datasets belong in the ignored `data/raw/`, `data/interim/`, and `data/processed/` directories. Generated metrics, predictions, and reports belong in `outputs/`.

## Coding and Documentation

Use English for code, file names, documentation, and comments. Prefer deterministic, reproducible scripts with type hints, command-line arguments, clear validation, and actionable error messages. Use four spaces for Python indentation and `snake_case` for modules, functions, and variables. Do not add dependencies or train models without explicit approval and documented data governance.

## Validation and Contributions

No dependency set or test runner is configured yet. When tooling is introduced, document exact commands in `README.md` and pin dependencies in `requirements.txt`. Keep commits focused and imperative (for example, `Add label validation checks`). Pull requests must describe the change, tests run, data-governance impact, and any limitations; include screenshots only if they contain no sensitive images.
