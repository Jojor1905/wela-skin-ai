"""Single-load, thread-safe Ultralytics inference service."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RawDetection:
    """One raw class-0 model prediction in pixel coordinates."""

    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class InferenceResult:
    """Raw low-confidence candidates and validated configured-threshold detections."""

    raw_detection_count: int
    detections: list[RawDetection]


class ModelService:
    """Own one YOLO instance and serialize access to model prediction."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model: Any | None = None
        self._prediction_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load and validate the configured local one-class model once."""
        if self._model is not None:
            return
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Configured model weights do not exist: {self._model_path}. Set MODEL_PATH to a local weights file."
            )
        from ultralytics import YOLO

        model = YOLO(str(self._model_path))
        model.to(self._device)
        names = model.names
        class_names = list(names.values()) if isinstance(names, dict) else list(names)
        if len(class_names) != 1:
            raise RuntimeError(
                f"The local API requires exactly one detection class; the configured model reports {len(class_names)}."
            )
        self._model = model

    def predict(self, image: Image.Image) -> InferenceResult:
        """Run class-0 prediction directly on the current decoded upload."""
        if self._model is None:
            raise RuntimeError("The model has not been loaded.")
        with self._prediction_lock:
            results = self._model.predict(
                source=image,
                conf=self._confidence_threshold,
                device=self._device,
                verbose=False,
            )
        detections: list[RawDetection] = []
        if not results:
            return InferenceResult(raw_detection_count=0, detections=detections)
        boxes = results[0].boxes
        if boxes is None:
            return InferenceResult(raw_detection_count=0, detections=detections)
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        raw_detection_count = len(coordinates)
        for coordinate, confidence, class_id in zip(coordinates, confidences, classes):
            values = [float(value) for value in coordinate]
            numeric_confidence = float(confidence)
            if round(float(class_id)) != 0:
                continue
            if not all(math.isfinite(value) for value in values + [numeric_confidence]):
                continue
            if numeric_confidence < self._confidence_threshold:
                continue
            detections.append(RawDetection(numeric_confidence, *values))
        return InferenceResult(raw_detection_count=raw_detection_count, detections=detections)
