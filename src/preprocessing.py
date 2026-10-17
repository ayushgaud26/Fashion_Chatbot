"""Validation, cleaning, normalisation and searchable-text construction.

Rules: preserve original display values; never invent missing values; one record = one retrieval unit."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

EXPECTED_COLUMNS = [
    "id", "gender", "masterCategory", "subCategory", "articleType", "baseColour",
    "season", "year", "usage", "productDisplayName",
]
TEXT_FIELDS = [
    "gender", "masterCategory", "subCategory", "articleType", "baseColour",
    "season", "usage", "productDisplayName",
]


class DatasetValidationError(ValueError):
    pass


def validate_columns(columns: Iterable[str]) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in set(columns)]
    if missing:
        raise DatasetValidationError(f"Dataset is missing expected columns: {missing}")


_APOS = re.compile(r"[’'`]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(value) -> str:
    """lower-case, drop apostrophes (men's -> mens), punctuation -> space, collapse whitespace."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).lower()
    s = _APOS.sub("", s)
    return _NON_ALNUM.sub(" ", s).strip()


def stem(tok: str) -> str:
    """Tiny, deterministic plural stemmer applied to BOTH queries and vocabulary (jeans->jean, watches->watch)."""
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("sses", "shes", "ches", "xes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def stem_phrase(text) -> tuple[str, ...]:
    return tuple(stem(t) for t in normalize_text(text).split())


def clean_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s or None


def clean_year(value) -> int | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return int(f)


def clean_id(value) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PreprocessReport:
    total_rows: int = 0
    valid_records: int = 0
    invalid_id_rows: int = 0
    duplicate_ids_dropped: int = 0
    missing_by_field: dict = field(default_factory=dict)

    def to_text(self) -> str:
        miss = ", ".join(f"{k}={v}" for k, v in self.missing_by_field.items() if v) or "none"
        return (
            f"rows read={self.total_rows}, valid={self.valid_records}, invalid id={self.invalid_id_rows}, "
            f"duplicate ids dropped={self.duplicate_ids_dropped}, missing values: {miss}"
        )


def preprocess_records(raw: Iterable[dict]) -> tuple[list[dict], PreprocessReport]:
    """Return normalised JSON-compatible records (original IDs/values preserved; missing -> None)."""
    rep = PreprocessReport(missing_by_field={f: 0 for f in TEXT_FIELDS + ["year"]})
    seen: set[int] = set()
    out: list[dict] = []
    for row in raw:
        rep.total_rows += 1
        pid = clean_id(row.get("id"))
        if pid is None:
            rep.invalid_id_rows += 1
            continue
        if pid in seen:
            rep.duplicate_ids_dropped += 1  # keep the first occurrence
            continue
        seen.add(pid)
        rec: dict = {"id": pid}
        for f in TEXT_FIELDS:
            v = clean_text(row.get(f))
            rec[f] = v
            if v is None:
                rep.missing_by_field[f] += 1
        rec["year"] = clean_year(row.get("year"))
        if rec["year"] is None:
            rep.missing_by_field["year"] += 1
        out.append(rec)
    rep.valid_records = len(out)
    return out, rep


def build_search_text(rec: dict) -> str:
    """Record-level searchable text. Only fields that actually have values are included."""
    parts = [f"Product ID: {rec['id']}."]
    labels = [
        ("productDisplayName", "Product name"), ("gender", "Gender"), ("masterCategory", "Category"),
        ("subCategory", "Subcategory"), ("articleType", "Article type"), ("baseColour", "Colour"),
        ("season", "Season"), ("year", "Year"), ("usage", "Usage"),
    ]
    for key, label in labels:
        v = rec.get(key)
        if v is not None:
            parts.append(f"{label}: {v}.")
    for key, label in (("brand", "Brand"), ("material", "Material"),
                       ("comparisonFacts", "Product details"), ("retailer", "Retailer")):
        value = rec.get(key)
        if value:
            parts.append(f"{label}: {value}.")
    if rec.get("price") is not None:
        parts.append(f"Price: {rec.get('currency', '')} {rec['price']}.")
    for key, label in (("size", "Available sizes"), ("rating", "Customer rating"),
                       ("reviewCount", "Review count"), ("discountPercentage", "Discount percent"),
                       ("availability", "Availability"), ("sizeFit", "Size and fit")):
        if rec.get(key) not in (None, ""):
            parts.append(f"{label}: {rec[key]}.")
    return " ".join(parts)


@dataclass
class IndexDoc:
    id: int
    text: str
    metadata: dict
    content_hash: str


def make_index_doc(rec: dict) -> IndexDoc:
    text = build_search_text(rec)
    # Chroma metadata accepts primitive values and flat primitive lists, not nested structures.
    meta = {k: v for k, v in rec.items() if v is not None and isinstance(v, (str, int, float, bool))}
    for k, value in rec.items():
        if isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
            meta[k] = value
    h = hashlib.sha256((text + json.dumps(meta, sort_keys=True)).encode("utf-8")).hexdigest()[:16]
    meta["content_hash"] = h
    return IndexDoc(id=rec["id"], text=text, metadata=meta, content_hash=h)
