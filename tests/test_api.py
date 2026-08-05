"""Unit tests for the local FastAPI prototype; no real model inference runs."""

from __future__ import annotations

import io
import hashlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app, decode_image
from src.api.config import DEFAULT_ALLOWED_ORIGINS, Settings, parse_allowed_origins, repository_root
from src.api.schemas import DISCLAIMER
from src.api.services.model_service import InferenceResult, RawDetection


class FakeModelService:
    def __init__(self) -> None:
        self.loaded = False
        self.seen_paths: list[Path] = []
        self.seen_sizes: list[tuple[int, int]] = []
        self.seen_hashes: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        self.loaded = True

    def predict(self, image: Image.Image) -> InferenceResult:
        self.seen_sizes.append(image.size)
        self.seen_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        return InferenceResult(
            raw_detection_count=3,
            detections=[
                RawDetection(confidence=0.8, x1=10, y1=5, x2=30, y2=15),
                RawDetection(confidence=0.6, x1=5, y1=45, x2=20, y2=65),
            ],
        )


class FailingModelService(FakeModelService):
    def predict(self, image: Image.Image) -> InferenceResult:
        raise RuntimeError("deliberate inference failure")


class MissingModelService(FakeModelService):
    def load(self) -> None:
        self.loaded = False
        raise FileNotFoundError("deliberately missing model")


def image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (100, 80),
    orientation: int | None = None,
    color: tuple[int, int, int] = (180, 140, 120),
) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color=color)
    if orientation is None:
        image.save(buffer, format=image_format)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(buffer, format=image_format, exif=exif)
    return buffer.getvalue()


class LocalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_model = FakeModelService()
        settings = Settings(
            model_path=Path("unused-by-fake.pt"),
            allowed_origins=("http://localhost:3000",),
            maximum_upload_bytes=10 * 1024 * 1024,
        )
        self.client_context = TestClient(create_app(settings, self.fake_model))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    @staticmethod
    def form_data(concerns: str = '["breakouts", "dryness"]') -> dict[str, str]:
        return {
            "gender": "prefer not to say",
            "ageRange": "25-34",
            "skinType": "combination",
            "concerns": concerns,
            "goal": "balanced routine",
        }

    def test_health_reports_loaded_model(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": True})

    def test_missing_model_is_not_reported_as_loaded(self) -> None:
        with self.assertLogs("uvicorn.error", level="ERROR"):
            settings = Settings(
                model_path=Path("missing.pt"),
                allowed_origins=("http://localhost:3000",),
            )
            with TestClient(create_app(settings, MissingModelService())) as client:
                response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unavailable", "model_loaded": False})

    def test_model_info_is_safe_and_does_not_expose_weights(self) -> None:
        response = self.client.get("/model-info")
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["project_class"], "acne_lesion")
        self.assertEqual(body["class_count"], 1)
        self.assertNotIn("model_path", body)
        self.assertNotIn("best.pt", response.text)

    def test_predict_returns_boxes_regions_recommendations_and_disclaimer(self) -> None:
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = self.client.post(
                "/predict",
                data=self.form_data(),
                files={"image": ("face.png", image_bytes(), "image/png")},
            )
        body = response.json()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((body["image_width"], body["image_height"]), (100, 80))
        self.assertEqual(body["total_detection_count"], 2)
        self.assertTrue(body["inference_executed"])
        self.assertEqual(body["raw_detection_count"], 3)
        self.assertEqual(body["post_threshold_detection_count"], 2)
        self.assertEqual(body["input_sha256_prefix"], hashlib.sha256(image_bytes()).hexdigest()[:12])
        self.assertAlmostEqual(body["mean_detection_confidence"], 0.7)
        self.assertEqual(body["detections"][0]["class_name"], "acne_lesion")
        self.assertAlmostEqual(body["detections"][0]["normalized_box"]["x1"], 0.1)
        self.assertEqual(body["approximate_face_region_counts"]["forehead"], 1)
        self.assertEqual(body["approximate_face_region_counts"]["left_cheek"], 1)
        self.assertEqual(body["dominant_region"], "forehead")
        self.assertEqual(body["disclaimer"], DISCLAIMER)
        categories = {item["category"] for item in body["product_recommendations"]}
        self.assertTrue({"cleanser", "serum", "moisturiser", "sunscreen", "optional spot care"}.issubset(categories))
        self.assertTrue(self.fake_model.seen_hashes)
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
        self.assertIn(body["request_id"], logs.output[-1])
        self.assertIn(body["input_sha256_prefix"], logs.output[-1])
        self.assertIn('"inference_executed": true', logs.output[-1])
        self.assertIn('"filtered_detection_count": 2', logs.output[-1])

    def test_comma_separated_concerns_are_accepted(self) -> None:
        response = self.client.post(
            "/predict",
            data=self.form_data("breakouts, oiliness"),
            files={"image": ("face.webp", image_bytes("WEBP"), "image/webp")},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_exif_orientation_is_applied_before_inference(self) -> None:
        response = self.client.post(
            "/predict",
            data=self.form_data(),
            files={"image": ("phone.jpg", image_bytes("JPEG", (40, 80), orientation=6), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.json()["image_width"], response.json()["image_height"]), (80, 40))
        self.assertEqual(self.fake_model.seen_sizes[-1], (80, 40))

    def test_distinct_upload_bytes_reach_inference_as_distinct_decoded_images(self) -> None:
        first = image_bytes(color=(10, 20, 30))
        second = image_bytes(color=(220, 210, 200))

        first_response = self.client.post(
            "/predict",
            data=self.form_data(),
            files={"image": ("first.png", first, "image/png")},
        )
        second_response = self.client.post(
            "/predict",
            data=self.form_data(),
            files={"image": ("second.png", second, "image/png")},
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertNotEqual(first_response.json()["input_sha256_prefix"], second_response.json()["input_sha256_prefix"])
        self.assertNotEqual(self.fake_model.seen_hashes[-2], self.fake_model.seen_hashes[-1])

    def test_inference_failure_is_non_200_and_has_no_mock_prediction(self) -> None:
        settings = Settings(
            model_path=Path("unused-by-fake.pt"),
            allowed_origins=("http://localhost:3000",),
        )
        with TestClient(create_app(settings, FailingModelService())) as client:
            response = client.post(
                "/predict",
                data=self.form_data(),
                files={"image": ("face.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Local model inference failed."})
        self.assertNotIn("detections", response.json())

    def test_prediction_does_not_create_a_permanent_upload_file(self) -> None:
        with patch.object(Path, "write_bytes", side_effect=AssertionError("upload must remain in memory")):
            response = self.client.post(
                "/predict",
                data=self.form_data(),
                files={"image": ("private-face.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)

    def test_current_upload_bytes_reach_image_decoding(self) -> None:
        upload = image_bytes(color=(91, 82, 73))
        with patch("src.api.app.decode_image", wraps=decode_image) as decoder:
            response = self.client.post(
                "/predict",
                data=self.form_data(),
                files={"image": ("current.png", upload, "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        decoder.assert_called_once_with(upload)

    def test_detection_counts_do_not_change_with_questionnaire_answers(self) -> None:
        upload = image_bytes(color=(55, 66, 77))
        first = self.client.post(
            "/predict",
            data=self.form_data("breakouts"),
            files={"image": ("same.png", upload, "image/png")},
        ).json()
        changed_answers = self.form_data("dryness,wrinkles")
        changed_answers.update({"gender": "man", "ageRange": "50+", "skinType": "dry", "goal": "simpler routine"})
        second = self.client.post(
            "/predict",
            data=changed_answers,
            files={"image": ("same.png", upload, "image/png")},
        ).json()

        self.assertEqual(first["input_sha256_prefix"], second["input_sha256_prefix"])
        self.assertEqual(first["total_detection_count"], second["total_detection_count"])
        self.assertEqual(first["approximate_face_region_counts"], second["approximate_face_region_counts"])
        self.assertEqual(first["detections"], second["detections"])

    def test_each_prediction_returns_fresh_response_provenance(self) -> None:
        upload = image_bytes()
        first = self.client.post(
            "/predict", data=self.form_data(), files={"image": ("face.png", upload, "image/png")}
        ).json()
        second = self.client.post(
            "/predict", data=self.form_data(), files={"image": ("face.png", upload, "image/png")}
        ).json()

        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(first["detections"], second["detections"])

    def test_empty_invalid_and_unsupported_files_are_rejected(self) -> None:
        empty = self.client.post(
            "/predict",
            data=self.form_data(),
            files={"image": ("empty.png", b"", "image/png")},
        )
        invalid = self.client.post(
            "/predict",
            data=self.form_data(),
            files={"image": ("fake.png", b"not an image", "image/png")},
        )
        unsupported = self.client.post(
            "/predict",
            data=self.form_data(),
            files={"image": ("face.gif", b"GIF89a", "image/gif")},
        )
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(unsupported.status_code, 415)

    def test_upload_size_limit_is_enforced(self) -> None:
        fake_model = FakeModelService()
        settings = Settings(
            model_path=Path("unused-by-fake.pt"),
            allowed_origins=("http://localhost:3000",),
            maximum_upload_bytes=8,
        )
        with TestClient(create_app(settings, fake_model)) as client:
            response = client.post(
                "/predict",
                data=self.form_data(),
                files={"image": ("large.png", b"123456789", "image/png")},
            )
        self.assertEqual(response.status_code, 413)
        self.assertFalse(fake_model.seen_paths)

    def test_invalid_concerns_json_is_rejected(self) -> None:
        response = self.client.post(
            "/predict",
            data=self.form_data('["breakouts"'),
            files={"image": ("face.png", image_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 422)

    def test_local_frontend_cors_preflight_is_allowed(self) -> None:
        response = self.client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")


class ConfigurationTests(unittest.TestCase):
    def test_default_model_path_resolution(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()
        self.assertEqual(settings.model_path, (repository_root() / "models/acne-yolo-best.pt").resolve())
        self.assertEqual(settings.yolo_device, "cpu")

    def test_model_path_environment_override(self) -> None:
        override = Path("models/custom.pt")
        with patch.dict(os.environ, {"MODEL_PATH": str(override), "YOLO_DEVICE": "cpu"}, clear=True):
            settings = Settings.from_environment()
        self.assertEqual(settings.model_path, (repository_root() / override).resolve())

    def test_cors_origin_parsing(self) -> None:
        self.assertEqual(parse_allowed_origins(None), DEFAULT_ALLOWED_ORIGINS)
        self.assertEqual(
            parse_allowed_origins(" https://wela-liff-prototype.vercel.app/, http://localhost:3001 "),
            ("https://wela-liff-prototype.vercel.app", "http://localhost:3001"),
        )


if __name__ == "__main__":
    unittest.main()
