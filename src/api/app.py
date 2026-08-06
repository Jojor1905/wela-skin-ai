"""FastAPI application for non-medical prototype inference."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from time import perf_counter
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

from .schemas import ProductQuery, ProductResponse
from .services.product_service import search_products

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_INFERENCE_DIMENSION = 1280
LOGGER = logging.getLogger("uvicorn.error")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


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
    decode_ms: float,
    inference_ms: float,
    postprocess_ms: float,
    total_ms: float,
    inference_executed: bool,
    raw_detection_count: int,
    filtered_detection_count: int,
    response_status: int,
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
                "decode_ms": round(decode_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "postprocess_ms": round(postprocess_ms, 3),
                "total_ms": round(total_ms, 3),
                "inference_executed": inference_executed,
                "raw_detection_count": raw_detection_count,
                "filtered_detection_count": filtered_detection_count,
                "response_status": response_status,
            },
            sort_keys=True,
        ),
)


def request_id_from_header(value: str | None) -> str:
    """Use a safe caller reference when supplied, otherwise create one."""
    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid4().hex


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
            oriented_image.thumbnail(
                (MAX_INFERENCE_DIMENSION, MAX_INFERENCE_DIMENSION),
                Image.Resampling.LANCZOS,
            )
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

    app = FastAPI(
        title="Wela Skin AI Prototype API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID", "Accept", "Origin"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_reference(request: Request, call_next):
        request_id = request_id_from_header(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    @app.post(
        "/product",
        response_model=ProductResponse,
        tags=["products"],
        summary="ค้นหาผลิตภัณฑ์ตามสภาพและปัญหาผิว",
    )
    async def get_products(payload: ProductQuery) -> ProductResponse:
        products = search_products(payload)
        return ProductResponse(
            count=len(products),
            matched_condition_ids=payload.condition_ids,
            items=products,
            disclaimer=(
                "คำแนะนำนี้จัดทำเพื่อการทดลองต้นแบบเชิงวิชาการ "
                "ไม่ใช่การวินิจฉัยหรือคำแนะนำทางการแพทย์ "
                "ควรทดสอบผลิตภัณฑ์บริเวณเล็ก ๆ ก่อนใช้งาน"
            ),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request, response: Response) -> HealthResponse:
        service: InferenceService = request.app.state.model_service
        if not service.is_loaded:
            response.status_code = 503
            return HealthResponse(status="unavailable", model_loaded=False)
        return HealthResponse(status="ok", model_loaded=True)

    @app.get("/model-info", response_model=ModelInfoResponse)
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

    @app.post("/predict", response_model=PredictResponse)
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
        request_id = request.state.request_id
        started_at = perf_counter()
        byte_count = 0
        sha256_prefix = ""
        width = 0
        height = 0
        decode_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0

        def log_attempt(
            *,
            response_status: int,
            inference_executed: bool = False,
            raw_detection_count: int = 0,
            filtered_detection_count: int = 0,
        ) -> None:
            log_prediction_event(
                request_id=request_id,
                byte_count=byte_count,
                sha256_prefix=sha256_prefix,
                width=width,
                height=height,
                decode_ms=decode_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
                total_ms=(perf_counter() - started_at) * 1000,
                inference_executed=inference_executed,
                raw_detection_count=raw_detection_count,
                filtered_detection_count=filtered_detection_count,
                response_status=response_status,
            )

        service: InferenceService = request.app.state.model_service
        if not service.is_loaded:
            log_attempt(response_status=503)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "model_not_ready",
                    "message": "The analysis model is not ready.",
                },
            )
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            await image.close()
            log_attempt(response_status=415)
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WEBP image uploads are supported.")
        try:
            payload = await image.read(runtime_settings.maximum_upload_bytes + 1)
        finally:
            await image.close()
        byte_count = len(payload)
        if not payload:
            log_attempt(response_status=400)
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(payload) > runtime_settings.maximum_upload_bytes:
            log_attempt(response_status=413)
            raise HTTPException(status_code=413, detail="The uploaded image exceeds the 10 MB limit.")
        input_sha256_prefix = hashlib.sha256(payload).hexdigest()[:12]
        sha256_prefix = input_sha256_prefix
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
            log_attempt(response_status=422)
            raise HTTPException(status_code=422, detail=str(error)) from error

        decode_started_at = perf_counter()
        try:
            decoded_image, width, height = decode_image(payload)
        except ValueError as error:
            decode_ms = (perf_counter() - decode_started_at) * 1000
            log_attempt(response_status=400)
            raise HTTPException(status_code=400, detail=str(error)) from error
        decode_ms = (perf_counter() - decode_started_at) * 1000

        inference_started_at = perf_counter()
        try:
            inference = await run_in_threadpool(service.predict, decoded_image)
        except Exception as error:
            inference_ms = (perf_counter() - inference_started_at) * 1000
            log_attempt(response_status=500)
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "prediction_failed",
                    "message": "Model inference failed.",
                },
            ) from error
        finally:
            decoded_image.close()
        inference_ms = (perf_counter() - inference_started_at) * 1000

        postprocess_started_at = perf_counter()
        try:
            analysis = analyse_detections(inference.detections, width, height)
            recommendations = build_recommendations(
                questionnaire,
                detection_count=len(analysis.detections),
                mean_confidence=analysis.mean_confidence,
                dominant_region=analysis.dominant_region,
            )
            result = PredictResponse(
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
        except Exception as error:
            postprocess_ms = (perf_counter() - postprocess_started_at) * 1000
            log_attempt(
                response_status=500,
                inference_executed=True,
                raw_detection_count=inference.raw_detection_count,
                filtered_detection_count=len(inference.detections),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "postprocess_failed",
                    "message": "Prediction post-processing failed.",
                },
            ) from error
        postprocess_ms = (perf_counter() - postprocess_started_at) * 1000
        log_attempt(
            response_status=200,
            inference_executed=True,
            raw_detection_count=inference.raw_detection_count,
            filtered_detection_count=len(inference.detections),
        )
        return result

    return app


app = create_app()
