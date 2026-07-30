# Dataset Card

## Identification

- Dataset name:
- Version and source URL:
- Maintainer or publisher:
- Access date:

## Purpose and Intended Use

- Original dataset purpose:
- Permitted use for this research prototype: Academic feasibility prototype for visible-lesion object detection only; not a medical device, diagnostic system, or face-recognition system.
- Prohibited or restricted uses:

## Data Description

- Number of images and subjects: 1,457 source images. Verified subject identifiers are unavailable, so the subject count and person-level separation cannot be guaranteed.
- Image sources and collection context:
- Available demographics and representation limits:
- Subject identifiers available for splitting: yes/no; details:

## Privacy and Governance

- Consent, de-identification, and privacy notes:
- Storage location and access controls:
- Known risks:

## Quality and Limitations

- Annotation source and quality checks:
- Converted dataset structure: `data/processed/acne04_yolo/` contains copied images and YOLO labels in train, validation, and test directories, plus a class map and split manifest. It contains no source-data modifications.
- Deterministic split: seed 42; target ratios train 70%, validation 15%, test 15%; known SHA-256 duplicate groups are kept within one split.
- Split counts: train 1,023 images / 13,448 objects / 566 very small boxes; validation 218 images / 2,769 objects / 131 very small boxes; test 216 images / 2,766 objects / 136 very small boxes. See `outputs/reports/acne04_split_summary.csv`.
- Known biases, gaps, and failure modes: Person-level separation cannot be guaranteed because verified subject IDs are unavailable. Results may not generalise to mobile selfies, Thai users, or other populations and capture conditions; this is a known domain gap, not a measured performance claim.
- Review date and reviewer:
