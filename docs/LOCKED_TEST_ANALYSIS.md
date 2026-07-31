# Locked ACNE04 Test Evaluation Analysis

## Decision

**SUFFICIENT FOR UI PROTOTYPE INTEGRATION**

This decision is limited to a controlled, non-medical research UI prototype.
It is not approval for diagnosis, treatment, production deployment, public
distribution, or clinical use.

## Completion and provenance

`outputs/locked_test/yolo26n_final_640_locked_test/LOCKED_TEST_COMPLETE`
exists. The recorded evaluation used `split: test` with 216 test images and
2,766 test instances. The result JSON records these hashes:

- Model SHA-256: `1c787a4c232f2687293f039975ec130648ecf4982eabdd5fe8f220f23d7c37a4`
- Source dataset YAML SHA-256: `f7a8a50cd435dd4bf274da4b7610364006a685e8f16f48af9b4de9a11d408182`
- Split manifest SHA-256: `6b3383e4d254d9b8101ca3c78d1819223138b3027934bbc452525c3c5e49a551`
- Runtime YAML SHA-256: `8e0108067ae6a183dbdc0bf0d5908f505f7ea9ed5d776f7643c5ea077d4e1ab7`

The run used MPS, PyTorch 2.13.0, Ultralytics 8.4.112, and Python 3.11.15.
Recorded inference time was approximately 17.49 ms per image (preprocess
1.16 ms and postprocess 0.29 ms per image).

## Validation versus locked test

| Metric | Best validation | Locked test | Absolute test − validation | Relative gap |
|---|---:|---:|---:|---:|
| Precision | 0.30448 | 0.3051066360 | +0.0006266360 | +0.21% |
| Recall | 0.38750 | 0.3477946493 | -0.0397053507 | -10.25% |
| mAP50 | 0.26160 | 0.2428234490 | -0.0187765510 | -7.18% |
| mAP50-95 | 0.06833 | 0.0605141441 | -0.0078158559 | -11.44% |

Precision is effectively unchanged. Recall and both mAP measures are lower on
the locked test set, with the largest relative gap at the stricter mAP50-95
threshold. Overall this is **moderately lower than validation**, not a
substantial collapse or an unexpected failure of generalisation within this
dataset's split and domain.

## Technical limitations

- The documented very small and densely packed boxes remain likely sources of
  missed detections. The validation prediction montages showed sparse output
  over several dense annotation groups; the lower test recall is consistent
  with that limitation.
- Validation confusion matrices showed a substantial background-associated
  prediction component. This indicates false-positive control remains a
  limitation, but it is not a medical error rate and was not re-estimated from
  the locked result here.
- The gap between mAP50 and mAP50-95 indicates that stricter box localisation
  is materially harder than coarse overlap. This is a localisation limitation,
  not evidence about the medical meaning of any region.
- The source contains duplicate image files with differing source annotation
  counts, very small boxes, and no guaranteed person-level separation.
- There is no validated external Thai or mobile-selfie test set. Performance
  outside this dataset and capture domain is unestablished.
- The one-class `acne_lesion` label is a project-level semantic normalisation
  of the source `fore` label. It is not a clinical diagnosis, subtype, or
  dermatologist-verified classification.

No retraining or hyperparameter recommendation is made from the locked test
results. The test result is preserved as the one-time generalisation record.

## Safe LINE LIFF prototype demonstration

The model can be integrated into a restricted LIFF research demonstration if
the UI is framed as technical visualization only:

1. Show a persistent notice such as “Research prototype only — not a medical
   diagnosis or treatment recommendation.” Do not show severity, diagnosis,
   triage, or treatment advice.
2. Render class-0 boxes and confidence values as model output, using neutral
   wording such as “model-marked region.” Do not claim that a region is
   clinically acne.
3. Keep ACNE04 images, labels, weights, and predictions in authorised local
   storage. Do not bundle them into a public LIFF deployment or send them to
   LINE, Kaggle, or another external cloud service without explicit permission.
4. For a hosted LIFF mock-up, use synthetic, public, or separately authorised
   demonstration images, or connect only to a private authorised backend.
   Avoid retaining uploaded photos and disable unnecessary analytics.
5. Restrict access to approved reviewers, record consent where user-provided
   photos are used, and do not implement identity recognition or inference.
6. Keep the locked test metrics out of user-facing copy; they document a
   research evaluation and are not a product guarantee.

This allows interface and interaction work while preserving the dataset
licence boundary and the project's non-medical scope.

## Reviewed files

- `outputs/locked_test/yolo26n_final_640_locked_test/LOCKED_TEST_RESULTS.json`
- `outputs/locked_test/yolo26n_final_640_locked_test/LOCKED_TEST_REPORT.md`
- `outputs/locked_test/yolo26n_final_640_locked_test/LOCKED_TEST_COMPLETE`
- `docs/FINAL_BASELINE_ANALYSIS.md`
- `docs/DATASET_CARD.md`
- `docs/LABEL_GUIDE.md`

No model, validation, prediction, or test evaluation was rerun, and the locked
result files were not modified.
