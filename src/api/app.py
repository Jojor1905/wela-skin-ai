"""FastAPI application for non-medical prototype inference."""

from __future__ import annotations

import json
import hashlib
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, Protocol
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from src.api.config import Settings
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    Questionnaire,
)
from src.api.services.analysis_service import analyse_detections
from src.api.services.model_service import InferenceResult, ModelService
from src.api.services.recommendation_service import build_recommendations


ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
LOGGER = logging.getLogger("uvicorn.error")


class InferenceService(Protocol):
    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def predict(self, image: Image.Image) -> InferenceResult: ...


def log_prediction_event(
    *,
    request_id: str,
    byte_count: int,
    sha256_prefix: str,
    width: int,
    height: int,
    inference_executed: bool,
    raw_detection_count: int,
    filtered_detection_count: int,
) -> None:
    """Log non-reversible request provenance without retaining image content."""
    LOGGER.info(
        "prediction_pipeline %s",
        json.dumps(
            {
                "request_id": request_id,
                "uploaded_byte_count": byte_count,
                "input_sha256_prefix": sha256_prefix,
                "decoded_width": width,
                "decoded_height": height,
                "inference_executed": inference_executed,
                "raw_detection_count": raw_detection_count,
                "filtered_detection_count": filtered_detection_count,
            },
            sort_keys=True,
        ),
    )


def parse_concerns(value: str) -> list[str]:
    """Accept a JSON string list or a comma-separated form value."""
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError("concerns must be a JSON array or a comma-separated string.") from error
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("concerns JSON must be an array of strings.")
        values = parsed
    else:
        values = stripped.split(",")
    normalized = [item.strip() for item in values if item.strip()]
    if len(normalized) > 20:
        raise ValueError("concerns may contain at most 20 values.")
    if any(len(item) > 100 for item in normalized):
        raise ValueError("each concern must contain at most 100 characters.")
    return normalized


def decode_image(payload: bytes) -> tuple[Image.Image, int, int]:
    """Verify, decode, and apply EXIF orientation without retaining source bytes."""
    try:
        with Image.open(BytesIO(payload)) as verification_image:
            detected_format = verification_image.format
            if detected_format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError("Only JPEG, PNG, and WEBP images are supported.")
            verification_image.verify()
        with Image.open(BytesIO(payload)) as source_image:
            oriented_image = ImageOps.exif_transpose(source_image)
            oriented_image.load()
            decoded = oriented_image.convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a decodable JPEG, PNG, or WEBP image.") from error
    width, height = decoded.size
    if width <= 0 or height <= 0:
        decoded.close()
        raise ValueError("The uploaded image has invalid dimensions.")
    return decoded, width, height


def create_app(
    settings: Settings | None = None,
    model_service: InferenceService | None = None,
) -> FastAPI:
    """Create an application with injectable settings and inference for unit tests."""
    runtime_settings = settings or Settings.from_environment()
    LOGGER.setLevel(getattr(logging, runtime_settings.log_level))
    inference_service = model_service or ModelService(
        runtime_settings.model_path,
        confidence_threshold=runtime_settings.confidence_threshold,
        device=runtime_settings.yolo_device,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model_service = inference_service
        try:
            await run_in_threadpool(inference_service.load)
        except Exception:
            LOGGER.exception("Model startup load failed; health checks will report unavailable.")
        yield

    application = FastAPI(
        title="Wela Skin AI Prototype API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request, response: Response) -> HealthResponse:
        service: InferenceService = request.app.state.model_service
        if not service.is_loaded:
            response.status_code = 503
            return HealthResponse(status="unavailable", model_loaded=False)
        return HealthResponse(status="ok", model_loaded=True)

    @application.get("/model-info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        service: InferenceService = request.app.state.model_service
        return ModelInfoResponse(
            project_class="acne_lesion",
            class_count=1,
            scope="UI prototype integration only",
            intended_use="Local experimental visualization of one-class model-marked regions.",
            limitations=[
                "The detector identifies only the project acne_lesion class.",
                "It does not detect or assess dark circles, acne scars, pigmentation, pores, wrinkles, dryness, oiliness, or sensitivity.",
                "It is not a medical device, diagnosis, treatment tool, or production-ready system.",
            ],
            model_loaded=service.is_loaded,
        )

    @application.post("/predict", response_model=PredictResponse)
    async def predict(
        request: Request,
        response: Response,
        image: UploadFile = File(...),
        gender: str = Form(...),
        age_range: str = Form(..., alias="ageRange"),
        skin_type: str = Form(..., alias="skinType"),
        concerns: str = Form(...),
        goal: str = Form(...),
    ) -> PredictResponse:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        request_id = uuid4().hex
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            await image.close()
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WEBP image uploads are supported.")
        try:
            payload = await image.read(runtime_settings.maximum_upload_bytes + 1)
        finally:
            await image.close()
        if not payload:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(payload) > runtime_settings.maximum_upload_bytes:
            raise HTTPException(status_code=413, detail="The uploaded image exceeds the 10 MB limit.")
        input_sha256_prefix = hashlib.sha256(payload).hexdigest()[:12]
        try:
            parsed_concerns = parse_concerns(concerns)
            questionnaire = Questionnaire(
                gender=gender.strip(),
                age_range=age_range.strip(),
                skin_type=skin_type.strip(),
                concerns=parsed_concerns,
                goal=goal.strip(),
            )
        except (ValueError, ValidationError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            decoded_image, width, height = decode_image(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        try:
            service: InferenceService = request.app.state.model_service
            inference = await run_in_threadpool(service.predict, decoded_image)
        except Exception as error:
            log_prediction_event(
                request_id=request_id,
                byte_count=len(payload),
                sha256_prefix=input_sha256_prefix,
                width=width,
                height=height,
                inference_executed=False,
                raw_detection_count=0,
                filtered_detection_count=0,
            )
            raise HTTPException(status_code=500, detail="Local model inference failed.") from error
        finally:
            decoded_image.close()

        analysis = analyse_detections(inference.detections, width, height)
        log_prediction_event(
            request_id=request_id,
            byte_count=len(payload),
            sha256_prefix=input_sha256_prefix,
            width=width,
            height=height,
            inference_executed=True,
            raw_detection_count=inference.raw_detection_count,
            filtered_detection_count=len(inference.detections),
        )
        recommendations = build_recommendations(
            questionnaire,
            detection_count=len(analysis.detections),
            mean_confidence=analysis.mean_confidence,
            dominant_region=analysis.dominant_region,
        )
        return PredictResponse(
            request_id=request_id,
            input_sha256_prefix=input_sha256_prefix,
            inference_executed=True,
            raw_detection_count=inference.raw_detection_count,
            post_threshold_detection_count=len(inference.detections),
            image_width=width,
            image_height=height,
            total_detection_count=len(analysis.detections),
            mean_detection_confidence=round(analysis.mean_confidence, 6),
            detections=analysis.detections,
            approximate_face_region_counts=analysis.region_counts,
            dominant_region=analysis.dominant_region,
            prototype_breakout_level=analysis.breakout_level,
            prototype_skin_score=analysis.skin_score,
            insights=analysis.insights,
            product_recommendations=recommendations,
        )

    return application


app = create_app()
