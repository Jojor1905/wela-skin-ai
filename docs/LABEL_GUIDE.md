# Label Guide

## Scope

The academic feasibility prototype has one object-detection class: `acne_lesion`. It is not a medical device, clinical diagnosis, severity-grade system, or face-recognition system.

## Verified Source-to-Project Mapping

| Field | Value |
| --- | --- |
| Source label | `fore` |
| Source label count | One unique object class |
| Source annotated objects | 18,983 |
| Project label | `acne_lesion` |
| YOLO class ID | `0` |
| Mapping status | Verified for academic object-detection prototype use. |

Verification basis:

- The local Pascal VOC annotations contain only the source label `fore`.
- The source dataset is documented as an acne image grading and lesion-counting dataset.
- Bounding boxes represent the source dataset’s local lesion annotations.
- `acne_lesion` is a project-level semantic normalisation for object-detection and user-interface wording.

This mapping is not an acne subtype, clinical diagnosis, severity grade, or dermatologist-verified medical classification. It does not establish medical accuracy. Do not rename or modify the original XML annotations.

## YOLO Representation

Each converted row uses `0 x_center y_center width height`, with normalized coordinates. The conversion integrity review verified all 18,983 source objects against their converted YOLO boxes without clamping or repair.

## Limits for Review

- Very small boxes and differing source annotations on duplicate image files remain documented dataset limitations.
- The mapping supports this local academic prototype only; it does not authorise clinical claims or uses outside the documented dataset permission.
