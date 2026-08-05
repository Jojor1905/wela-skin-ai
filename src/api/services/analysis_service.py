"""Non-medical, image-relative summaries of one-class detector output."""

from __future__ import annotations

from dataclasses import dataclass

from src.api.schemas import BoundingBox, Detection, RegionCounts
from src.api.services.model_service import RawDetection


REGION_ORDER = ("forehead", "left_cheek", "right_cheek", "nose", "chin")


@dataclass(frozen=True)
class PrototypeAnalysis:
    detections: list[Detection]
    region_counts: RegionCounts
    dominant_region: str
    breakout_level: str
    skin_score: int
    mean_confidence: float
    insights: list[str]


def approximate_region(center_x: float, center_y: float) -> str:
    """Map a normalized box centre to a coarse image-relative facial zone."""
    if center_y < 0.30:
        return "forehead"
    if center_y >= 0.72:
        return "chin"
    if 0.38 <= center_x <= 0.62:
        return "nose"
    return "left_cheek" if center_x < 0.50 else "right_cheek"


def _breakout_level(count: int) -> str:
    if count == 0:
        return "none_marked"
    if count <= 5:
        return "low"
    if count <= 15:
        return "moderate"
    return "high"


def analyse_detections(
    raw_detections: list[RawDetection], image_width: int, image_height: int
) -> PrototypeAnalysis:
    """Build a UI-only score and region summary; no clinical meaning is assigned."""
    detections: list[Detection] = []
    counts = {region: 0 for region in REGION_ORDER}
    for raw in raw_detections:
        x1 = min(max(raw.x1, 0.0), float(image_width))
        y1 = min(max(raw.y1, 0.0), float(image_height))
        x2 = min(max(raw.x2, 0.0), float(image_width))
        y2 = min(max(raw.y2, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized = BoundingBox(
            x1=x1 / image_width,
            y1=y1 / image_height,
            x2=x2 / image_width,
            y2=y2 / image_height,
        )
        region = approximate_region(
            (normalized.x1 + normalized.x2) / 2,
            (normalized.y1 + normalized.y2) / 2,
        )
        counts[region] += 1
        detections.append(
            Detection(
                confidence=min(max(raw.confidence, 0.0), 1.0),
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                normalized_box=normalized,
                approximate_region=region,
            )
        )

    count = len(detections)
    mean_confidence = sum(item.confidence for item in detections) / count if count else 0.0
    dominant_region = "none"
    if count:
        dominant_region = max(REGION_ORDER, key=lambda region: counts[region])
    skin_score = max(0, round(100 - min(count, 20) * 3 - mean_confidence * 10))
    insights = [
        f"The one-class model marked {count} image region{'s' if count != 1 else ''} as acne_lesion candidates.",
        "Face-region counts are approximate image-coordinate zones, not detected facial anatomy.",
        "The prototype skin score is a UI-only index derived from model count and confidence; it is not a health score.",
    ]
    if count:
        insights.insert(
            1,
            f"The largest approximate concentration is in the {dominant_region.replace('_', ' ')} zone.",
        )
    else:
        insights.insert(1, "No candidate region passed the configured model confidence threshold.")
    return PrototypeAnalysis(
        detections=detections,
        region_counts=RegionCounts(**counts),
        dominant_region=dominant_region,
        breakout_level=_breakout_level(count),
        skin_score=skin_score,
        mean_confidence=mean_confidence,
        insights=insights,
    )

