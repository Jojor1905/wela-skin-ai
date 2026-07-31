# Local Apple Silicon Pilot Guide

## Scope

This is a local academic feasibility pilot only. It uses Apple Silicon MPS, train and validation splits only, and never uploads images or uses the test split for training decisions. It does not provide medical, diagnostic, or production results.

## Prerequisites

- Python 3.11.
- An approved local environment containing PyTorch with MPS support and a compatible Ultralytics installation. No packages are installed by this repository workflow.
- `outputs/readiness/READINESS_REPORT.md` must contain the local technical pilot approval.
- Do not upload ACNE04 images, labels, or processed copies to Kaggle or any other cloud service. The available source permission does not explicitly authorise external upload.

## Smoke Test

After confirming the local environment, run this explicit one-epoch smoke test. It is the first command that starts training and therefore requires user approval:

```bash
python3.11 src/train_local_pilot.py \
  --data configs/local_pilot.yaml \
  --model yolo26n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 2 \
  --workers 0 \
  --device mps \
  --project outputs/experiments \
  --run-name local_pilot_smoke \
  --seed 42
```

The script verifies MPS before starting and does not silently fall back to CPU. If MPS memory is exhausted, lower `--batch` or `--imgsz` and rerun explicitly.

## Pilot Defaults

The default run is one epoch for smoke testing, with image size 640, batch size 2, zero workers, MPS, seed 42, patience 3, and cache disabled. Outputs stay under `outputs/experiments/` and include Ultralytics weights, plots, `results.csv`, and `local_pilot_summary.json`. Generated weights and prediction artifacts must remain local and uncommitted.
