# Label Guide

## Scope

The initial task contains exactly one object-detection class: `acne_lesion`. This guide must be reviewed and approved against the selected dataset before labels are created, converted, or changed.

## Annotation Rules To Define Before Use

- Lesion inclusion and exclusion criteria:
- Bounding-box placement convention:
- Occlusion, blur, lighting, and makeup handling:
- Ambiguous-case review process:
- Annotator training and agreement checks:

## YOLO Representation

Use class ID `0` for `acne_lesion` only after the dataset audit confirms this mapping. A YOLO label row uses `class_id x_center y_center width height`, with coordinates normalized to the image dimensions. Do not fabricate labels or convert annotations until the source schema and approved rules are documented.
