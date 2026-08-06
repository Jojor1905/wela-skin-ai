"""Product API tests backed only by the approved local JSON catalog."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import DEFAULT_ALLOWED_ORIGINS, Settings
from src.api.schemas import PredictResponse, ProductQuery, ProductResponse
from src.api.services.product_service import DATA_PATH, search_products


class ReadyModelService:
    """Avoid loading or changing model weights in product-route tests."""

    is_loaded = True

    def load(self) -> None:
        return None

    def predict(self, image):  # pragma: no cover - product tests never infer
        raise AssertionError("Product API tests must not run model inference.")


def source_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def source_names() -> set[str]:
    return {
        product["name"]
        for condition in source_data()["conditions"]
        for products in condition.get("products", {}).values()
        for product in products
    }


def expected_names(condition_id: str, categories: set[str] | None = None) -> set[str]:
    names: set[str] = set()
    for condition in source_data()["conditions"]:
        if condition["id"] != condition_id:
            continue
        for category, products in condition.get("products", {}).items():
            if categories is None or category in categories:
                names.update(product["name"] for product in products)
    return names


class ProductApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = Settings(
            model_path=Path("unused-by-product-tests.pt"),
            allowed_origins=DEFAULT_ALLOWED_ORIGINS,
        )
        cls.client_context = TestClient(create_app(settings, ReadyModelService()))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def post_product(self, **overrides):
        payload = {"condition_ids": [], "categories": [], "limit": 12}
        payload.update(overrides)
        return self.client.post("/product", json=payload)

    def test_post_product_route_exists_once(self) -> None:
        matches = [
            route
            for route in self.client.app.routes
            if route.path == "/product" and "POST" in (route.methods or set())
        ]
        self.assertEqual(len(matches), 1)

    def test_oily_skin_returns_only_approved_catalog_products(self) -> None:
        response = self.post_product(condition_ids=["oily_skin"], limit=50)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual({item["name"] for item in response.json()["items"]}, expected_names("oily_skin"))

    def test_dry_skin_returns_only_approved_catalog_products(self) -> None:
        response = self.post_product(condition_ids=["dry_skin"], limit=50)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual({item["name"] for item in response.json()["items"]}, expected_names("dry_skin"))

    def test_category_filtering(self) -> None:
        response = self.post_product(condition_ids=["oily_skin"], categories=["serum"], limit=50)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual({item["name"] for item in response.json()["items"]}, expected_names("oily_skin", {"serum"}))
        self.assertTrue(all(item["category"] == "serum" for item in response.json()["items"]))

    def test_search_filtering(self) -> None:
        response = self.post_product(condition_ids=["oily_skin"], search="Niacinamide", limit=50)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["count"], 0)
        self.assertTrue(
            all(
                "niacinamide" in f"{item['name']} {item['reason']}".casefold()
                for item in response.json()["items"]
            )
        )

    def test_limit_validation_rejects_out_of_range_values(self) -> None:
        for limit in (0, -1, 51):
            with self.subTest(limit=limit):
                self.assertEqual(self.post_product(limit=limit).status_code, 422)

    def test_unknown_condition_returns_empty_response(self) -> None:
        response = self.post_product(condition_ids=["not_a_catalog_condition"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["count"], 0)
        self.assertEqual(response.json()["items"], [])

    def test_every_returned_name_exists_in_source_json(self) -> None:
        response = self.post_product(limit=50)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue({item["name"] for item in response.json()["items"]}.issubset(source_names()))

    def test_duplicate_names_merge_metadata(self) -> None:
        duplicate_data = {
            "conditions": [
                {
                    "id": "first",
                    "name_th": "หนึ่ง",
                    "source_pages": [1],
                    "products": {"serum": [{"name": "Shared  Product", "reason": "first"}]},
                },
                {
                    "id": "second",
                    "name_th": "สอง",
                    "source_pages": [2],
                    "products": {"serum": [{"name": "shared product", "reason": "second"}]},
                },
            ]
        }
        query = ProductQuery(condition_ids=["first", "second"], limit=50)
        with patch("src.api.services.product_service.load_product_rules", return_value=duplicate_data):
            products = search_products(query)
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].condition_ids, ["first", "second"])
        self.assertEqual(products[0].condition_names_th, ["หนึ่ง", "สอง"])
        self.assertEqual(products[0].source_pages, [1, 2])

    def test_product_response_schema_validation(self) -> None:
        response = self.post_product(condition_ids=["oily_skin"], limit=4)
        validated = ProductResponse.model_validate(response.json())
        self.assertEqual(validated.count, len(validated.items))
        self.assertIn("วิชาการ", validated.disclaimer)
        self.assertIn("ไม่ใช่การวินิจฉัย", validated.disclaimer)

    def test_source_json_is_valid_utf8_json(self) -> None:
        decoded = DATA_PATH.read_bytes().decode("utf-8")
        self.assertIsInstance(json.loads(decoded), dict)

    def test_existing_routes_remain_registered(self) -> None:
        routes = {(route.path, frozenset(route.methods or set())): route for route in self.client.app.routes}
        self.assertIn(("/health", frozenset({"GET"})), routes)
        self.assertIn(("/model-info", frozenset({"GET"})), routes)
        predict_route = routes[("/predict", frozenset({"POST"}))]
        self.assertIs(predict_route.response_model, PredictResponse)
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/model-info").status_code, 200)


if __name__ == "__main__":
    unittest.main()
