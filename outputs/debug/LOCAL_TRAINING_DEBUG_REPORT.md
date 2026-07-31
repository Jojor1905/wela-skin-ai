# Local Training Debug Report

## Final Status

LOCAL ONE-EPOCH SMOKE TEST PASSED

The one-epoch MPS smoke test was executed manually in the normal terminal
process because the Codex sandbox cannot expose MPS to its Python process.

## MPS Verification

- Normal terminal: MPS available and functional on Apple M5.
- Manual run device: `mps`.
- Codex sandbox: MPS availability could not be reproduced (`is_built=True`,
  `is_available=False`).
- No CPU fallback was used.

## Experiment

- Exact experiment directory: `/Users/jojorandpleng/Desktop/wela-skin-ai/outputs/experiments/yolo26n_smoke_640_manual`
- One epoch completed: yes (`results.csv` contains exactly one row, `epoch=1`).
- Validation completed: yes (`val: true`, `split: val`, and validation metrics are present).
- Elapsed time: 99.805 seconds.
- Device: `mps`.

## Smoke-Test Metrics

| Metric | Value |
| --- | ---: |
| Precision | 0.20299 |
| Recall | 0.19502 |
| mAP50 | 0.09944 |
| mAP50-95 | 0.02211 |

No NaN values were present in the epoch metrics. These are pilot-training
metrics only and are not medical or production-readiness claims.

## Required Artifacts

The manual run produced:

- `results.csv`
- `weights/best.pt`
- `weights/last.pt`
- `local_pilot_summary.json`
- Ultralytics plots, including `results.png`, precision/recall curves,
  confusion matrices, and train/validation batch visualizations.

The runtime configuration used an absolute dataset root and contained only
train and validation paths. The test split was not used.

## Exact Smoke-Test Command

```bash
python src/train_local_pilot.py \
  --data configs/local_pilot.yaml \
  --model yolo26n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 2 \
  --workers 0 \
  --device mps \
  --project outputs/experiments \
  --run-name yolo26n_smoke_640 \
  --seed 42
```

The manual run used the safe resolved directory
`yolo26n_smoke_640_manual` because prior requested run names existed.

## Validation and Safety Checks

- Unit tests: 29 passed.
- YOLO validator: `True`.
- Dataset counts remained 1,457 images and 18,983 objects.
- Train pairs: 1,023 images / 1,023 labels.
- Validation pairs: 218 images / 218 labels.
- Test pairs: 216 images / 216 labels; not used for pilot decisions.
- No raw or processed dataset files were modified.
- No cloud upload occurred.

## Remaining Warnings

- The Codex sandbox still cannot reproduce MPS availability; the successful
  result is from the normal terminal process.
- The source permission does not explicitly address external-cloud upload or
  redistribution; keep all weights and image artifacts local and uncommitted.
- This one-epoch pilot is a workflow smoke test, not a performance claim.
