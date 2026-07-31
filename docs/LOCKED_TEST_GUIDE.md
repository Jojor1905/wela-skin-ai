# One-Time Locked ACNE04 Test Evaluation

## Purpose

This workflow evaluates the final local 640-pixel baseline once against the
held-out ACNE04 test split. It is an evaluation-only action: it never trains,
does not use train or validation metrics for model selection, and must not be
used to tune a later model. It makes no medical, diagnostic, or
production-readiness claim.

The available ACNE04 permission supports the documented academic prototype but
does not explicitly authorise external-cloud upload. Keep the data, weights,
predictions, and reports on authorised local storage.

## Preconditions

- Run from the normal Apple Silicon Python 3.11 environment. The Codex
  sandbox cannot reliably access MPS.
- The final `best.pt` must already exist locally.
- The source dataset YAML must define `test: images/test` (or an equivalent
  test path), and the split manifest must be present beneath the dataset root.
- The test set must remain untouched and must not have been used for model
  selection or tuning.
- Use a new run name. The evaluator refuses an existing output directory and
  has no overwrite option.

## Manual command

Do not run this command from the sandbox. The explicit confirmation flag is
required and records that this is the one-time locked evaluation.

```bash
python src/evaluate_locked_test.py \
  --model outputs/experiments/yolo26n_final_640_30e_batch1_manual/weights/best.pt \
  --data data/processed/acne04_yolo/acne04.yaml \
  --imgsz 640 \
  --batch 1 \
  --workers 0 \
  --device mps \
  --project outputs/locked_test \
  --run-name yolo26n_final_640_locked_test \
  --confirm-locked-test
```

## Guardrails and outputs

Before loading the model, the script verifies 216 test images, 216 test
labels, and exact image-to-label stem pairing. It resolves the dataset root
from the repository and writes an absolute-path runtime YAML without using
the global Ultralytics `datasets_dir`. The runtime file is written as
`outputs/runtime/locked_test_<timestamp>.yaml` and includes path, train, val,
test, and names. It verifies MPS when requested and
never falls back silently to CPU.

The run directory is created by Ultralytics only when the requested name is
unused. It must contain:

- `LOCKED_TEST_RESULTS.json`
- `LOCKED_TEST_REPORT.md`
- `LOCKED_TEST_COMPLETE`
- Ultralytics validation plots, including the confusion matrix and
  precision-recall curves

The JSON/report records test image and instance counts, precision, recall,
mAP50, mAP50-95, preprocessing/inference/postprocessing milliseconds per
image, elapsed time, model/data/manifest SHA-256 hashes, library versions,
Python version, device, and `split: test`.

Do not delete or rewrite a completed run. Do not use the test metrics to
recommend hyperparameter changes. Stop after recording the one-time result.
