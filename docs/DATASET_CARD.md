# Dataset Card: ACNE04

## Identification

- Dataset name: ACNE04.
- Official source: https://github.com/xpwu95/LDL
- Source work: *Joint Acne Image Grading and Counting via Label Distribution Learning*.
- Access date: 2026-07-30.

## Purpose and Intended Use

- Intended use in this repository: local academic feasibility prototype.
- Current task: one-class visible acne-lesion object detection using project class `acne_lesion` (YOLO ID `0`).
- Not intended for diagnosis, medical treatment, clinical decision-making, or production deployment.
- The prototype does not perform face recognition, identity matching, or identity inference.

## Data Description

- Source images and annotations: 1,457 images and 18,983 source objects.
- Converted structure: `data/processed/acne04_yolo/` contains copied images, YOLO labels, a class map, and a split manifest; it does not modify source data.
- Deterministic split: seed 42; train 1,023 images / 13,448 objects, validation 218 / 2,769, test 216 / 2,766.
- Verified subject identifiers are unavailable. Person-level separation therefore cannot be guaranteed.

## Privacy, Permission, and Storage

- The available source statement supports academic use. It does not explicitly authorise public redistribution, commercial use, or external-cloud upload.
- Keep the dataset local to authorised storage. The dataset must not be published in this repository.
- Request author clarification before uploading images or processed copies to Kaggle or another external cloud service. See `docs/LICENSE_LOG.md` and `docs/ACNE04_PERMISSION_REQUEST.md`.

## Quality and Limitations

- Technical conversion integrity review: 1,457 images and 18,983 objects were verified; no conversion mismatch, rejected image, or duplicate split crossing was found.
- Very small lesions/boxes remain a documented limitation.
- Some identical duplicate image files have inconsistent source annotations. Those annotation differences originate in the source XML and are preserved per image in conversion; they are not a conversion failure.
- No validated Thai or mobile-selfie external test set is available. Generalisation to Thai users, mobile selfies, or other populations and capture conditions has not been established.
- This dataset card does not make medical, dermatological, or performance claims.
