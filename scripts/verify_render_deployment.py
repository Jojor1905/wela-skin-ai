#!/usr/bin/env python3
"""Verify the checked-in Render model and FastAPI health path without mutation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "models/acne-yolo-best.pt"
HASH_PREFIX_LENGTH = 12

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:HASH_PREFIX_LENGTH]


async def get_health(application: Any) -> tuple[int, dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 8000),
        "root_path": "",
    }
    async with application.router.lifespan_context(application):
        await application(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return int(start["status"]), json.loads(body)


def main() -> int:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Required deployment model is missing: {MODEL_PATH.relative_to(REPOSITORY_ROOT)}")

    print(f"Model path: {MODEL_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"Model size: {MODEL_PATH.stat().st_size} bytes")
    print(f"Model SHA-256 prefix: {sha256_prefix(MODEL_PATH)}")

    from src.api.app import create_app
    from src.api.config import Settings
    from src.api.services.model_service import ModelService

    print("Import: src.api.app OK")
    settings = Settings(
        model_path=MODEL_PATH,
        allowed_origins=("http://localhost:3000",),
        yolo_device="cpu",
    )
    model_service = ModelService(MODEL_PATH, device="cpu")
    model_service.load()
    if not model_service.is_loaded:
        raise RuntimeError("The model did not report loaded after CPU loading.")
    print("Model load: CPU OK")

    status_code, health = asyncio.run(get_health(create_app(settings, model_service)))
    expected = {"status": "ok", "model_loaded": True}
    if status_code != 200 or health != expected:
        raise RuntimeError(f"Unexpected /health response: HTTP {status_code} {health}")
    print(f"GET /health: HTTP {status_code} {json.dumps(health, sort_keys=True)}")
    print("Render deployment verification: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Render deployment verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
