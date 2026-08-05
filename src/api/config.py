"""Environment-backed configuration for the inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_RELATIVE_PATH = Path("models/acne-yolo-best.pt")
DEFAULT_YOLO_DEVICE = "cpu"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
)


def repository_root() -> Path:
    """Return the repository root independently of the current directory."""
    return Path(__file__).resolve().parents[2]


def parse_allowed_origins(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ALLOWED_ORIGINS
    origins = tuple(origin.strip().rstrip("/") for origin in value.split(",") if origin.strip())
    if not origins:
        raise ValueError("ALLOWED_ORIGINS must contain at least one comma-separated origin.")
    return origins


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings that do not expose secrets or model bytes."""

    model_path: Path
    allowed_origins: tuple[str, ...]
    yolo_device: str = DEFAULT_YOLO_DEVICE
    log_level: str = DEFAULT_LOG_LEVEL
    maximum_upload_bytes: int = 10 * 1024 * 1024
    confidence_threshold: float = 0.25

    @classmethod
    def from_environment(cls) -> "Settings":
        root = repository_root()
        configured_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_RELATIVE_PATH)))
        model_path = configured_path if configured_path.is_absolute() else root / configured_path
        yolo_device = os.environ.get("YOLO_DEVICE", DEFAULT_YOLO_DEVICE).strip()
        if not yolo_device:
            raise ValueError("YOLO_DEVICE must not be empty.")
        log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
        if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be CRITICAL, ERROR, WARNING, INFO, or DEBUG.")
        return cls(
            model_path=model_path.resolve(),
            allowed_origins=parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS")),
            yolo_device=yolo_device,
            log_level=log_level,
        )
