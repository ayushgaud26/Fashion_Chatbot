"""Load a small, manually verified retail catalogue without changing the source dataset."""
from __future__ import annotations

import json
from pathlib import Path


def load_retail_products(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Retail catalogue must be a JSON list: {path}")
    products = []
    seen = set()
    required = ("id", "productDisplayName", "brand", "price", "currency", "productUrl",
                "retailer", "priceCheckedAt")
    for item in raw:
        missing = [field for field in required if item.get(field) in (None, "")]
        if missing:
            raise ValueError(f"Retail product is missing required fields {missing}: {item.get('id', '<no id>')}")
        try:
            item["id"] = int(item["id"])
            item["price"] = float(item["price"])
            if item.get("originalPrice") is not None:
                item["originalPrice"] = float(item["originalPrice"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Retail product has an invalid ID or price: {item.get('id')}") from exc
        if item["id"] in seen:
            raise ValueError(f"Duplicate retail product ID: {item['id']}")
        if item["price"] < 0 or not str(item["productUrl"]).startswith("https://"):
            raise ValueError(f"Retail product has an invalid price or URL: {item['id']}")
        seen.add(item["id"])
        item.setdefault("gender", None)
        item.setdefault("masterCategory", "Footwear")
        item.setdefault("subCategory", "Shoes")
        item.setdefault("articleType", "Running Shoes")
        item.setdefault("baseColour", None)
        item.setdefault("season", None)
        item.setdefault("year", None)
        item.setdefault("usage", "Sports")
        item["catalogSource"] = "retail"
        products.append(item)
    return products
