# Full Dataset Readiness Check

This is a conversion-integrity review only; it does not validate clinical or medical labels.

- Images checked: 1457
- Converted objects checked: 18983
- PASS: 1317
- NEEDS_REVIEW: 140
- REJECT: 0
- Duplicate hash groups: 38
- Duplicate groups crossing splits: 0
- Conversion mismatches: 0
- Absolute-path findings: 0

## Needs-review breakdown

- Images containing very small boxes (including overlaps): 88
- Images in duplicate groups with differing annotation counts (including overlaps): 54
- Images carrying both reasons: 2
- Very-small-box only: 86
- Duplicate annotation-count difference: 54
- Coordinate or IoU concern: 0
- Other reason: 0

## Blocking issues

- Technical conversion-integrity blockers: none.
- External-cloud permission blocker: the source statement does not explicitly address private Kaggle or other cloud upload, public redistribution, or sharing trained weights. Request clarification before any external upload.

## Non-blocking technical limitations

- Very small boxes and differing source annotation counts within duplicate image groups are routed to human review. They are not conversion mismatches.
- Every duplicate group with differing annotation counts remained within one split and had consistent source XML and source image dimensions. The SHA-256/XML/YOLO evidence is in `duplicate_annotation_review.csv`.

## Verification results

- Unit tests: passed (19 tests).
- YOLO validator: valid (`True`).

## Readiness decisions

### Technical readiness

READY FOR PILOT TRAINING

Conversion integrity is clean, and the `fore` to `acne_lesion` mapping is verified for the local academic research prototype. This decision does not authorise medical claims, production use, or external upload.

### External-cloud readiness

NOT READY FOR KAGGLE UPLOAD — LICENCE CLARIFICATION REQUIRED

The available source statement permits academic usage but does not explicitly address cloud upload or redistribution. See `docs/LICENSE_LOG.md` and `docs/ACNE04_PERMISSION_REQUEST.md`.
