#!/usr/bin/env python3
"""Verify that distinct multipart uploads reach the configured YOLO model."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.api.app import decode_image
from src.api.config import Settings, repository_root
from src.api.services.analysis_service import analyse_detections
from src.api.services.model_service import ModelService


@dataclass(frozen=True)
class VerificationImage:
    path: Path
    annotation_count: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def annotation_count(label_path: Path) -> int:
    if not label_path.is_file():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def select_images(root: Path) -> list[VerificationImage]:
    image_root = root / "data/processed/acne04_yolo/images/test"
    label_root = root / "data/processed/acne04_yolo/labels/test"
    candidates: list[VerificationImage] = []
    for path in sorted(image_root.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        payload = path.read_bytes()
        candidates.append(
            VerificationImage(
                path=path,
                annotation_count=annotation_count(label_root / f"{path.stem}.txt"),
                sha256=sha256_bytes(payload),
            )
        )

    selected: list[VerificationImage] = []
    seen_annotation_counts: set[int] = set()
    seen_hashes: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: (item.annotation_count, item.path.name)):
        if candidate.sha256 in seen_hashes or candidate.annotation_count in seen_annotation_counts:
            continue
        selected.append(candidate)
        seen_hashes.add(candidate.sha256)
        seen_annotation_counts.add(candidate.annotation_count)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        for candidate in candidates:
            if candidate.sha256 in seen_hashes:
                continue
            selected.append(candidate)
            seen_hashes.add(candidate.sha256)
            if len(selected) == 3:
                break
    if len(selected) < 3:
        raise RuntimeError("Could not locate three distinct existing test images.")
    return selected


def detection_signature(detections: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            item["class_name"],
            round(float(item["confidence"]), 6),
            round(float(item["normalized_box"]["x1"]), 6),
            round(float(item["normalized_box"]["y1"]), 6),
            round(float(item["normalized_box"]["x2"]), 6),
            round(float(item["normalized_box"]["y2"]), 6),
            item["approximate_region"],
        )
        for item in detections
    ]


def direct_prediction(model: ModelService, payload: bytes) -> dict[str, Any]:
    image, width, height = decode_image(payload)
    try:
        inference = model.predict(image)
    finally:
        image.close()
    analysis = analyse_detections(inference.detections, width, height)
    return {
        "raw_detection_count": inference.raw_detection_count,
        "post_threshold_detection_count": len(inference.detections),
        "total_detection_count": len(analysis.detections),
        "approximate_face_region_counts": analysis.region_counts.model_dump(),
        "detections": [item.model_dump() for item in analysis.detections],
    }


def endpoint_prediction(client: httpx.Client, image: VerificationImage, payload: bytes) -> dict[str, Any]:
    content_type = mimetypes.guess_type(image.path.name)[0] or "image/jpeg"
    response = client.post(
        "/predict",
        data={
            "gender": "woman",
            "ageRange": "30-39",
            "skinType": "combination",
            "concerns": "visible-breakouts",
            "goal": "calmer-looking-skin",
        },
        files={"image": (image.path.name, payload, content_type)},
    )
    print(f"  HTTP status: {response.status_code}")
    if response.status_code != 200:
        raise RuntimeError(f"Prediction failed for {image.path.name}: {response.text[:500]}")
    if "no-store" not in response.headers.get("cache-control", ""):
        raise RuntimeError("Prediction response is missing Cache-Control: no-store.")
    return response.json()


def main() -> int:
    args = parse_args()
    root = repository_root()
    settings = Settings.from_environment()
    images = select_images(root)
    if len({item.sha256 for item in images}) != 3:
        raise RuntimeError("Selected verification files do not have three distinct hashes.")

    print(f"Model: {settings.model_path}")
    model = ModelService(settings.model_path, confidence_threshold=settings.confidence_threshold)
    model.load()
    endpoint_results: list[dict[str, Any]] = []
    direct_results: list[dict[str, Any]] = []

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json() != {"status": "ok", "model_loaded": True}:
            raise RuntimeError(f"Backend health is not ready: {health.text}")

        for image in images:
            payload = image.path.read_bytes()
            print(f"\n{image.path.name}")
            print(f"  local SHA-256 prefix: {image.sha256[:12]}")
            print(f"  annotation objects: {image.annotation_count}")
            endpoint = endpoint_prediction(client, image, payload)
            direct = direct_prediction(model, payload)
            print(f"  response lesion count: {endpoint['total_detection_count']}")
            print(f"  regional counts: {json.dumps(endpoint['approximate_face_region_counts'], sort_keys=True)}")
            print(f"  returned detections: {len(endpoint['detections'])}")
            print(f"  raw/post-threshold: {endpoint['raw_detection_count']}/{endpoint['post_threshold_detection_count']}")

            if endpoint.get("input_sha256_prefix") != image.sha256[:12]:
                raise RuntimeError("The endpoint did not report the current uploaded-image hash.")
            if endpoint.get("inference_executed") is not True:
                raise RuntimeError("The endpoint reports that inference was not executed.")
            if endpoint["post_threshold_detection_count"] != len(endpoint["detections"]):
                raise RuntimeError("Post-threshold count does not match returned detections.")
            for field in ("raw_detection_count", "post_threshold_detection_count", "total_detection_count", "approximate_face_region_counts"):
                if endpoint[field] != direct[field]:
                    raise RuntimeError(f"Endpoint/direct inference mismatch for {image.path.name}: {field}")
            if detection_signature(endpoint["detections"]) != detection_signature(direct["detections"]):
                raise RuntimeError(f"Endpoint/direct detection mismatch for {image.path.name}.")
            endpoint_results.append(endpoint)
            direct_results.append(direct)

    reported_hashes = {result["input_sha256_prefix"] for result in endpoint_results}
    if len(reported_hashes) != 3:
        raise RuntimeError("Every request did not reach the backend as a distinct input hash.")

    endpoint_signatures = {
        json.dumps(
            {
                "count": item["total_detection_count"],
                "regions": item["approximate_face_region_counts"],
                "detections": detection_signature(item["detections"]),
            },
            sort_keys=True,
        )
        for item in endpoint_results
    }
    direct_signatures = {
        json.dumps(
            {
                "count": item["total_detection_count"],
                "regions": item["approximate_face_region_counts"],
                "detections": detection_signature(item["detections"]),
            },
            sort_keys=True,
        )
        for item in direct_results
    }
    if len(endpoint_signatures) == 1 and len(direct_signatures) > 1:
        raise RuntimeError("Endpoint responses are fixed even though direct model outputs differ.")
    if len(endpoint_signatures) == 1:
        print("\nAll three model outputs happened to match; hashes and direct inference still matched request-by-request.")
    print("\nPASS: three distinct uploads were inferred independently and matched direct model output.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
