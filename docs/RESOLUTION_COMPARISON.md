# ACNE04 Resolution Comparison

## Scope

This is a read-only comparison of the completed local validation runs:

- **Experiment A:** `yolo26n_baseline_640_15e_manual`
- **Experiment B:** `yolo26n_pilot_960_5e_manual`

Only the train/validation workflow recorded in each run was considered. The
test split was not evaluated or used for any conclusion. The class name in the
rendered outputs is the project class label; this report makes no medical
judgement about any marked region.

## Completion and provenance

Both run directories contain `results.csv`, validation plots, confusion
matrices, prediction/label montages, and `weights/best.pt` and
`weights/last.pt`. All numeric values in both CSV files are finite.

Experiment A requested 15 epochs but records 10 epochs (1–10), consistent
with early stopping (`patience: 3`). Its final elapsed value in the run summary
is 852.396 seconds. Experiment B records all 5 requested epochs and reports
779.402 seconds. Validation was enabled in both runs with `split: val`.

These runs are not a resolution-only controlled experiment: A starts from the
previous 640 pilot checkpoint, while B starts from A's best checkpoint. B also
uses batch size 1 versus A's batch size 2 and runs for five additional epochs
after the A checkpoint. The results are therefore directional evidence, not a
causal estimate of resolution alone.

## Best validation results

The best epoch is selected by validation mAP50-95.

| Metric | 640 (best epoch 7) | 960 (best epoch 2) | 960 − 640 | Relative change |
|---|---:|---:|---:|---:|
| Precision | 0.30448 | 0.29389 | -0.01059 | -3.48% |
| Recall | 0.38750 | 0.36006 | -0.02744 | -7.08% |
| mAP50 | 0.26160 | 0.22613 | -0.03547 | -13.56% |
| mAP50-95 | 0.06833 | 0.05944 | -0.00889 | -13.01% |
| Validation box loss | 2.38016 | 2.24976 | -0.13040 | -5.48% |
| Validation classification loss | 2.16347 | 2.49448 | +0.33101 | +15.30% |
| Validation DFL/localisation loss | 0.00626 | 0.00593 | -0.00033 | -5.27% |

At the recorded best validation point, 960 is lower on all four detection
metrics. Its validation box and DFL/localisation losses are lower, but its
classification loss is higher.

For reference, the final recorded rows are:

| Metric | 640 final epoch 10 | 960 final epoch 5 |
|---|---:|---:|
| Precision | 0.32094 | 0.26074 |
| Recall | 0.36981 | 0.37667 |
| mAP50 | 0.25784 | 0.21353 |
| mAP50-95 | 0.06770 | 0.05747 |
| Validation box loss | 2.38277 | 2.28599 |
| Validation classification loss | 2.22128 | 2.40250 |
| Validation DFL/localisation loss | 0.00633 | 0.00599 |

## Time and speed

The run summaries report 852.396 seconds for A and 779.402 seconds for B, so
the 960 run is 72.994 seconds shorter in wall-clock time (-8.56%). This total
is not a fair speed comparison because A records 10 epochs and B records 5,
and their batch sizes differ. Using the CSV's cumulative epoch-time field as a
rough per-recorded-epoch indicator gives approximately 85.24 s/epoch for 640
and 155.88 s/epoch for 960; the 960 run is therefore about 82.9% slower per
epoch under these settings.

Neither `results.csv`, `args.yaml`, nor `local_pilot_summary.json` records an
inference-throughput or per-image validation-speed measurement. No inference
speed number is invented here; a valid FPS or ms/image comparison would
require a separate, controlled benchmark and is outside this review.

## Visual and error-pattern review

The PR curves and F1 curves for both runs show a low-confidence operating
region and a rapid precision decline as recall is increased. The 640 PR curve
has the higher area/mAP50 label (0.262 versus 0.226), consistent with the CSV.

The validation label montages contain many small and densely packed source
boxes. In the corresponding prediction montages:

- Both resolutions visibly miss many of the small/dense marked boxes.
- The sampled 960 prediction montages do not show a clear small-box recall
  improvement over 640; several dense examples remain unboxed or sparsely
  boxed.
- No convincing oversized predicted boxes are visible in the sampled
  montages.
- No clear duplicate predicted boxes can be confirmed from these montages;
  dense overlapping rectangles in label montages are source annotations, not
  model predictions.
- The montages do not establish a defensible example of a prediction on an
  eye, lip, hair region, mole, shadow, or other artefact. Absence from these
  samples is not evidence that such errors never occur.

Both confusion matrices contain a large background-associated component
(640: 38,717 in the predicted-class/background column; 960: 50,323). The
raw counts are not directly comparable because the resolution and inference
configuration change the candidate/grid counts, but the shared pattern means
false-positive control remains an explicit review item. The matrices do not
support a clinical interpretation.

## Resolution decision

**Recommended next full baseline: 640.**

In these runs, 640 has the better best precision, recall, mAP50, and mAP50-95,
while requiring fewer seconds per recorded epoch and using batch size 2. The
960 setting costs substantially more compute per epoch and did not visibly
recover more small/dense boxes in the supplied prediction montages. The 960
resolution may still be useful for a later controlled small-object study, but
it should not replace the 640 baseline based on these confounded runs.

This recommendation is a workflow and validation choice, not a statement of
medical or production suitability.

## Files reviewed

For **Experiment A** and **Experiment B**:

- `results.csv`
- `args.yaml`
- `local_pilot_summary.json`
- `results.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `BoxF1_curve.png`
- `BoxPR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `val_batch0_labels.jpg`, `val_batch0_pred.jpg`
- `val_batch1_labels.jpg`, `val_batch1_pred.jpg`
- `val_batch2_labels.jpg`, `val_batch2_pred.jpg`
- `weights/best.pt`, `weights/last.pt` (existence checks only)

No training was run or repeated, no test data was evaluated, and no dataset
files were modified.
