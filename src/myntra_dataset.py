"""Loader and normalizer for the supplied Myntra CSV catalogue."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .preprocessing import clean_text


@dataclass
class MyntraReport:
    total_rows: int
    valid_records: int
    invalid_id_rows: int
    duplicate_ids_dropped: int
    missing_by_field: dict = field(default_factory=dict)

    def to_text(self) -> str:
        missing = ", ".join(f"{k}={v}" for k, v in self.missing_by_field.items() if v) or "none"
        return (f"rows read={self.total_rows}, valid={self.valid_records}, invalid id={self.invalid_id_rows}, "
                f"duplicate ids dropped={self.duplicate_ids_dropped}, missing values: {missing}")


def _json(value):
    if not value or str(value).strip().lower() in {"null", "none", "nan"}:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _number(value):
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        result = float(s)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _details(raw):
    result = {}
    for item in raw or []:
        if isinstance(item, dict):
            key = clean_text(item.get("specification_name"))
            value = clean_text(item.get("specification_value"))
            if key and value and value.upper() != "NA":
                result[key] = value
    return result


_COLORS = ("navy blue", "sky blue", "light blue", "dark blue", "off white", "olive green", "sea green",
          "maroon", "beige", "black", "white", "blue", "red", "green", "pink", "purple", "orange",
          "yellow", "brown", "grey", "gray", "gold", "silver", "cream", "teal", "peach", "rust", "multi")


def _extract_color(text):
    """Extract explicit color words from the product description, since this CSV has no color column."""
    normalized = clean_text(text)
    if not normalized:
        return None
    matches = []
    for color in _COLORS:
        match = re.search(rf"(?<![a-z]){re.escape(color)}(?![a-z])", normalized, flags=re.I)
        if match:
            matches.append((match.start(), color.title().replace("Grey", "Grey")))
    matches.sort()
    found = list(dict.fromkeys(value for _, value in matches))
    return ", ".join(found) or None


def load_myntra_csv(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Myntra dataset not found: {path}. Set DATASET_PATH in .env to the CSV file.")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise RuntimeError(f"Could not parse Myntra CSV {path}: {exc}") from exc
    required = {"product_id", "title", "product_description", "url"}
    missing_cols = sorted(required - set(frame.columns))
    if missing_cols:
        raise ValueError(f"Myntra CSV is missing required columns: {missing_cols}")

    records, seen, invalid, duplicates = [], set(), 0, 0
    optional = ("brand", "masterCategory", "subCategory", "gender", "baseColour", "material", "size", "price", "originalPrice",
                "discountPercentage", "rating", "reviewCount", "availability", "imageUrl", "productUrl")
    missing_counts = Counter()
    for source in frame.to_dict(orient="records"):
        try:
            pid = int(str(source.get("product_id", "")).strip())
        except (TypeError, ValueError):
            invalid += 1
            continue
        if pid in seen:
            duplicates += 1
            continue
        seen.add(pid)
        specs = _details(_json(source.get("product_specifications")))
        details = _json(source.get("product_details")) or {}
        crumbs = _json(source.get("breadcrumbs")) or []
        images = _json(source.get("images")) or []
        size_rows = _json(source.get("sizes")) or []
        size_values = list(dict.fromkeys(clean_text(x.get("size")) for x in size_rows
                                         if isinstance(x, dict) and clean_text(x.get("size"))))
        gender = next((clean_text(x.get("name")) for x in crumbs if isinstance(x, dict)
                       and clean_text(x.get("name")) in {"Men", "Women", "Unisex"}), None)
        category = next((clean_text(x.get("name")) for x in crumbs[2:]
                         if isinstance(x, dict) and clean_text(x.get("name"))), None)
        description = clean_text(source.get("product_description"))
        brand = clean_text(source.get("title"))
        product_name = " ".join(x for x in (brand, description) if x)
        product_url = clean_text(source.get("url"))
        image_url = next((clean_text(x) for x in images if isinstance(x, str) and x.strip()), None)
        if image_url and image_url.startswith("http://"):
            image_url = "https://" + image_url[len("http://"):]
        product_details = details if isinstance(details, dict) else {}
        material_and_care = clean_text(product_details.get("material_and_care"))
        material = specs.get("Fabric")
        if not material and material_and_care:
            material = material_and_care.split(",", 1)[0].split(".", 1)[0].strip()
            material = re.sub(r"^(?:material|fabric)\s*:\s*", "", material, flags=re.I).strip() or None
        size_fit = clean_text(product_details.get("size_and_fit"))
        color = specs.get("Colour") or _extract_color(product_details.get("description") or description)
        # delivery_options describe shipping and returns, not whether an item is in stock.
        availability = None
        rec = {
            "id": pid, "productDisplayName": product_name or f"Myntra product {pid}",
            "brand": brand, "gender": gender,
            "masterCategory": clean_text(crumbs[0].get("name")) if crumbs and isinstance(crumbs[0], dict) else None,
            "subCategory": category,
            "articleType": specs.get("Type") or category, "baseColour": color,
            "color": color, "colorSource": "product description" if color and not specs.get("Colour") else None,
            "material": material, "size": ", ".join(size_values) or None,
            "sizeValues": size_values,
            "sizeFit": size_fit, "season": None, "year": None, "usage": specs.get("Occasion"),
            "price": _number(source.get("final_price")), "originalPrice": _number(source.get("initial_price")),
            "discountPercentage": _number(source.get("discount")), "currency": clean_text(source.get("currency")) or "INR",
            "rating": _number(source.get("rating")), "reviewCount": _number(source.get("ratings_count")),
            "availability": availability, "imageUrl": image_url,
            "productUrl": product_url if product_url and product_url.startswith(("https://", "http://")) else None,
            "platformUrl": None, "retailer": "Myntra", "sellerName": clean_text(source.get("seller_name")),
            "comparisonFacts": "; ".join(f"{k}: {v}" for k, v in specs.items()),
            "specifications": specs, "customerSignals": _json(source.get("what_customers_said")),
            "starBreakdown": _json(source.get("amount_of_stars")), "sourceFields": source,
        }
        for key in optional:
            if rec.get(key) in (None, ""):
                missing_counts[key] += 1
        records.append(rec)
    report = MyntraReport(len(frame), len(records), invalid, duplicates,
                          {key: missing_counts.get(key, 0) for key in optional})
    return records, report, frame
