# Final ACNE04 640-Pixel Baseline Analysis

## Decision

**READY FOR LOCKED TEST EVALUATION**

This decision authorises one locked evaluation of the selected validation
checkpoint against the held-out test split. It does not authorise tuning on the
test results, medical interpretation, or production use. The test split was
not evaluated during this analysis.

## Completion and integrity

The clean run `outputs/experiments/yolo26n_final_640_30e_manual` contains 30
rows in `results.csv`, epochs 1 through 30. `args.yaml` records a clean start
from local `yolo26n.pt`, `epochs: 30`, `patience: 8`, `imgsz: 640`, `batch: 2`,
`device: mps`, `seed: 42`, `cache: false`, `pretrained: true`, and
`split: val`.

All 30 epochs completed; early stopping did not occur. Validation metrics are
present for every epoch. All numeric result values are finite. The required
artifacts are present and non-empty: `results.csv`, `args.yaml`, plots,
confusion matrices, validation label/prediction montages,
`weights/best.pt`, and `weights/last.pt`. There is no evidence of an
interrupted or incomplete training run.

The best validation mAP50-95 is at **epoch 29**:

- Precision: **0.35594**
- Recall: **0.39292**
- mAP50: **0.29662**
- mAP50-95: **0.07957**

The final epoch is slightly lower on recall and mAP (0.38895, 0.29289, and
0.07873 mAP50-95), so epoch 29 is the checkpoint to lock for evaluation.

## Comparison with previous validation runs

Values below are each experiment's best row selected by validation mAP50-95.

| Experiment | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Elapsed seconds |
|---|---:|---:|---:|---:|---:|---:|
| Clean 640 baseline | 29 | 0.35594 | 0.39292 | 0.29662 | 0.07957 | 2469.761 |
| 640 pilot, 5 epochs | 5 | 0.25703 | 0.34561 | 0.19547 | 0.05089 | 416.549 |
| 640 continuation, requested 15 / recorded 10 | 7 | 0.30448 | 0.38750 | 0.26160 | 0.06833 | 852.396 |
| 960 pilot, 5 epochs | 2 | 0.29389 | 0.36006 | 0.22613 | 0.05944 | 779.402 |

Compared with the 5-epoch 640 pilot, the clean baseline improves precision by
0.09891 (+38.48%), recall by 0.04731 (+13.69%), mAP50 by 0.10115 (+51.75%),
and mAP50-95 by 0.02868 (+56.36%).

Compared with the previous 640 continuation, improvements are precision
0.05146 (+16.90%), recall 0.00542 (+1.40%), mAP50 0.03502 (+13.39%), and
mAP50-95 0.01124 (+16.45%).

Compared with the 960 pilot, improvements are precision 0.06205 (+21.11%),
recall 0.03286 (+9.13%), mAP50 0.07049 (+31.17%), and mAP50-95 0.02013
(+33.87%). Elapsed times are not a resolution-only comparison because the
runs have different epoch counts, starting checkpoints, and batch sizes.

## Loss and metric trends

Training losses decrease overall across the clean run:

- Train box loss: 2.78986 to 2.35395.
- Train classification loss: 4.00697 to 2.09205.
- Train DFL/localisation loss (`train/l1_loss`): 0.00662 to 0.00564.

Validation classification loss decreases strongly from 3.73970 to 2.12197.
Validation box loss is noisy but ends below its first-epoch value (2.31796
versus 2.37312). Validation DFL/localisation loss fluctuates around 0.0060
and ends at 0.00603. Precision, recall, mAP50, and mAP50-95 trend upward with
epoch-to-epoch variation and reach their best mAP50-95 at epoch 29.

There is no clear sustained overfitting pattern: training losses improve,
validation classification loss improves, and validation metrics improve late
in training. The small epoch-30 decline and validation-loss fluctuations are
normal instability signals to record, not sufficient evidence of overfitting
after this run.

## Validation prediction review

The label montages contain many very small and densely packed source boxes.
The prediction montages show more class-0 boxes than the short pilot outputs,
but still miss many boxes in dense groups. Small-box recall therefore remains
a material limitation.

The confusion matrix contains a substantial background-associated component
(2,373 in the class-0/ class-0 cell, 45,790 in the predicted class-0/true
background cell, and 396 in the predicted-background/true class-0 cell under
the displayed axes). This indicates that false-positive control and missed
boxes require continued review; raw matrix counts should not be treated as a
clinical error rate.

From the supplied prediction montages:

- Missed small or densely packed boxes are clearly visible.
- False-positive behaviour is suggested by the confusion matrix and low
  confidence overlays, but individual false positives cannot be classified
  reliably from montages alone.
- No definitive duplicate-prediction pattern is established; close boxes in
  dense regions need targeted review.
- No obvious oversized prediction boxes are visible.
- No prediction can be confidently identified from these montages as being on
  an eye, lip, hair, shadow, mole, or other artefact. This absence is not proof
  that such errors never occur.

These observations concern technical class-0 detection behaviour only. They
do not determine what any marked region medically represents.

## Locked-evaluation rationale and limits

The clean run is suitable for a one-time locked test evaluation because it
started from `yolo26n.pt`, used train/validation only, completed all requested
epochs, produced complete finite artifacts, and improved over the prior 640
and 960 validation runs. The selected checkpoint is epoch 29 (`best.pt`).

The evaluation remains limited by the documented very small boxes, differing
source annotations in duplicate image files, unavailable person-level
separation guarantees, and lack of an external Thai/mobile-selfie validation
set. The project label `acne_lesion` is a research-prototype semantic mapping,
not a clinical diagnosis or dermatologist-verified classification.

## Files reviewed

Clean run:

- `results.csv`
- `args.yaml`
- `local_pilot_summary.json`
- `results.png`
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png`
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `val_batch0_labels.jpg`, `val_batch0_pred.jpg`
- `val_batch1_labels.jpg`, `val_batch1_pred.jpg`
- `val_batch2_labels.jpg`, `val_batch2_pred.jpg`
- `weights/best.pt`, `weights/last.pt` (existence and size checks)

Prior runs and project context:

- `outputs/experiments/yolo26n_pilot_640_manual/results.csv`
- `outputs/experiments/yolo26n_baseline_640_15e_manual/results.csv`
- `outputs/experiments/yolo26n_pilot_960_5e_manual/results.csv`
- `docs/RESOLUTION_COMPARISON.md`
- `docs/DATASET_CARD.md`
- `docs/LABEL_GUIDE.md`

No test evaluation was run, no training was repeated, and no raw or processed
dataset files were modified.
