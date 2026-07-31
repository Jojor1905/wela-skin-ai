# Final Local 640-Pixel Baseline

## Scope

This is a clean, local academic feasibility baseline for the ACNE04 YOLO
conversion. It uses the train and validation splits only. The test split is
not passed to Ultralytics and must not be used for training decisions. The
workflow is not a medical, diagnostic, or production evaluation.

The run must start from the official local `yolo26n.pt` weight file. It must
not continue from the prior smoke, 15-epoch, or 960-pixel experiment weights.
The available dataset permission does not explicitly authorise external cloud
upload, so keep all images, labels, weights, and predictions local.

## Configuration

`configs/final_baseline_640.yaml` is a portable dataset YAML:

```yaml
path: data/processed/acne04_yolo
train: images/train
val: images/val
names:
  0: acne_lesion
```

The intended training settings are:

| Setting | Value |
|---|---|
| Model | `yolo26n.pt` |
| Epochs | `30` |
| Image size | `640` |
| Batch | `2` |
| Workers | `0` |
| Device | `mps` |
| Seed | `42` |
| Patience | `8` |
| Cache | `false` |
| Pretrained | `true` |

The script creates an absolute-path runtime YAML under `outputs/runtime/`,
checks train/validation image-label counts, rejects any `test` key, verifies
MPS, and chooses a unique output directory rather than overwriting an existing
run. It never silently falls back to CPU.

## Manual command

Run this only from the normal Apple Silicon terminal after reviewing the
preflight output. The Codex sandbox cannot access MPS reliably.

```bash
python src/train_local_pilot.py \
  --data configs/final_baseline_640.yaml \
  --model yolo26n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 2 \
  --workers 0 \
  --device mps \
  --project outputs/experiments \
  --run-name yolo26n_final_baseline_640 \
  --seed 42 \
  --patience 8 \
  --no-cache \
  --pretrained
```

The command starts training and is intentionally not run as part of repository
preparation. If the requested run name already exists, the script reports the
generated numeric suffix. Review `local_pilot_summary.json`, `results.csv`,
and the validation plots after completion. Do not evaluate the test split.

## Safety notes

- Do not modify `data/raw/` or `data/processed/acne04_yolo/`.
- Do not delete previous experiment folders.
- Do not upload the dataset or generated predictions.
- If MPS memory is exhausted, adjust the command explicitly; no CPU fallback
  is performed.
- Generated weights and facial-image outputs must remain uncommitted.
