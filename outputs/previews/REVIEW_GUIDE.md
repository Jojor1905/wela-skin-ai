# Visual Annotation Review Guide

These previews are a read-only review aid. They do not validate a medical diagnosis, change source data, or approve a label mapping.

## Review each preview

1. Confirm each red box covers a visible acne lesion. Inspect the yellow centre marker used for boxes whose width or height is 10 pixels or less; the preview remains at the source image dimensions so small lesions are not resized away.
2. Flag boxes that instead point to pores, moles, shadows, scars, highlights, hair, or other non-acne areas.
3. Check the displayed `fore` label and dimensions against the visible target. The source label has not been medically validated.
4. For each row in `duplicate_review.csv`, compare all previews in its duplicate group and determine whether visually identical images have identical annotations. The fingerprint is an aid for exact source-box comparison, not a substitute for visual review.

## Record a decision

Use `preview_index.csv` and `duplicate_review.csv` to enter one of these statuses:

- `Pass`: boxes and source-label usage appear consistent with the intended visible-lesion research task.
- `Needs review`: there is ambiguity, a difficult image, a very small target, or a discrepancy requiring a second reviewer.
- `Reject`: boxes clearly do not support the intended visible-acne-lesion label or the annotation is materially unusable.

Add concise evidence in `reviewer_notes`; do not add identity information or reproduce source images outside the approved local workspace.
