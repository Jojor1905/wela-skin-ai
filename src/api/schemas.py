"""Pydantic request-domain and response schemas for the local API."""

from __future__ import annotations

from pydantic import BaseModel, Field


DISCLAIMER = (
    "Experimental visual analysis for prototype demonstration only. "
    "Results may be incomplete or inaccurate and are not a medical diagnosis."
)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str = "acne_lesion"
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox
    normalized_box: BoundingBox
    approximate_region: str


class RegionCounts(BaseModel):
    forehead: int = 0
    left_cheek: int = 0
    right_cheek: int = 0
    nose: int = 0
    chin: int = 0


class ProductRecommendation(BaseModel):
    category: str
    focus: str
    rationale: str


class Questionnaire(BaseModel):
    gender: str = Field(min_length=1, max_length=80)
    age_range: str = Field(min_length=1, max_length=80)
    skin_type: str = Field(min_length=1, max_length=80)
    concerns: list[str] = Field(default_factory=list, max_length=20)
    goal: str = Field(min_length=1, max_length=300)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    project_class: str
    class_count: int
    scope: str
    intended_use: str
    limitations: list[str]
    model_loaded: bool


class PredictResponse(BaseModel):
    request_id: str
    input_sha256_prefix: str = Field(min_length=12, max_length=12)
    inference_executed: bool
    raw_detection_count: int = Field(ge=0)
    post_threshold_detection_count: int = Field(ge=0)
    image_width: int
    image_height: int
    total_detection_count: int
    mean_detection_confidence: float
    detections: list[Detection]
    approximate_face_region_counts: RegionCounts
    dominant_region: str
    prototype_breakout_level: str
    prototype_skin_score: int = Field(ge=0, le=100)
    insights: list[str]
    product_recommendations: list[ProductRecommendation]
    disclaimer: str = DISCLAIMER
