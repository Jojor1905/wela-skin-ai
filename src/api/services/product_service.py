from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..schemas import ProductItem, ProductQuery


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "wela_skin_rules_source.json"
)


@lru_cache(maxsize=1)
def load_product_rules() -> dict[str, Any]:
    """
    โหลดฐานข้อมูลหนึ่งครั้งต่อ Server Process
    เพื่อไม่ต้องเปิดไฟล์ JSON ใหม่ทุก Request
    """
    if not DATA_PATH.is_file():
        raise RuntimeError(
            "Product data file is missing: "
            "src/api/data/wela_skin_rules_source.json"
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    conditions = data.get("conditions")

    if not isinstance(conditions, list):
        raise RuntimeError(
            "Invalid product data: conditions must be a list"
        )

    return data


def create_product_id(name: str) -> str:
    digest = hashlib.sha256(
        name.strip().casefold().encode("utf-8")
    ).hexdigest()[:12]

    return f"product_{digest}"


def search_products(query: ProductQuery) -> list[ProductItem]:
    data = load_product_rules()

    requested_conditions = {
        value.strip()
        for value in query.condition_ids
        if value.strip()
    }

    requested_categories = {
        value.strip().casefold()
        for value in query.categories
        if value.strip()
    }

    search_text = (
        query.search.strip().casefold()
        if query.search
        else ""
    )

    # ใช้ชื่อสินค้าเป็น Key เพื่อป้องกันสินค้าซ้ำ
    records: dict[str, dict[str, Any]] = {}

    for condition in data["conditions"]:
        condition_id = str(condition.get("id", "")).strip()

        if not condition_id:
            continue

        if (
            requested_conditions
            and condition_id not in requested_conditions
        ):
            continue

        condition_name = str(
            condition.get("name_th", condition_id)
        ).strip()

        source_pages = [
            int(page)
            for page in condition.get("source_pages", [])
            if isinstance(page, int)
        ]

        product_groups = condition.get("products", {})

        if not isinstance(product_groups, dict):
            continue

        for category, product_list in product_groups.items():
            normalized_category = str(category).strip().casefold()

            if (
                requested_categories
                and normalized_category not in requested_categories
            ):
                continue

            if not isinstance(product_list, list):
                continue

            for raw_product in product_list:
                if not isinstance(raw_product, dict):
                    continue

                name = str(raw_product.get("name", "")).strip()
                reason = str(raw_product.get("reason", "")).strip()

                if not name:
                    continue

                searchable_text = " ".join(
                    [
                        name,
                        reason,
                        condition_id,
                        condition_name,
                        normalized_category,
                    ]
                ).casefold()

                if search_text and search_text not in searchable_text:
                    continue

                dedupe_key = name.casefold()

                if dedupe_key not in records:
                    records[dedupe_key] = {
                        "id": create_product_id(name),
                        "name": name,
                        "category": normalized_category,
                        "reason": reason,
                        "condition_ids": [],
                        "condition_names_th": [],
                        "source_pages": [],
                        "image_url": raw_product.get("image_url"),
                    }

                record = records[dedupe_key]

                if condition_id not in record["condition_ids"]:
                    record["condition_ids"].append(condition_id)

                if condition_name not in record["condition_names_th"]:
                    record["condition_names_th"].append(
                        condition_name
                    )

                record["source_pages"] = sorted(
                    set(record["source_pages"] + source_pages)
                )

    products = [
        ProductItem(**record)
        for record in records.values()
    ]

    products.sort(key=lambda item: item.name.casefold())

    return products[: query.limit]