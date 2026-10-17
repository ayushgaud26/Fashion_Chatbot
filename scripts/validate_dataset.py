"""Inspect the configured Arrow or Myntra CSV catalogue."""
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config
from src.ingestion import load_records

cfg = get_config()
records, report, source = load_records(cfg)
columns = list(source.column_names) if hasattr(source, "column_names") else list(source.columns)
print(f"File: {cfg.dataset_path}\nRows in file: {report.total_rows}\nColumns ({len(columns)}): {', '.join(columns)}")
print("\nPreprocessing:", report.to_text())
print(f"\nDuplicate IDs: {report.duplicate_ids_dropped}; invalid product IDs: {report.invalid_id_rows}")
bad_product = sum(1 for p in records if p.get("productUrl") and urlparse(p["productUrl"]).scheme not in {"http", "https"})
bad_images = sum(1 for p in records if p.get("imageUrl") and urlparse(p["imageUrl"]).scheme not in {"http", "https"})
print(f"Invalid product URL schemes: {bad_product}; invalid image URL schemes: {bad_images}")
print(f"Products with product links: {sum(bool(p.get('productUrl')) for p in records)}/{len(records)}")
print(f"Products with image links: {sum(bool(p.get('imageUrl')) for p in records)}/{len(records)}")
print(f"Products with price: {sum(p.get('price') is not None for p in records)}/{len(records)}")
print(f"Products with rating: {sum(p.get('rating') is not None for p in records)}/{len(records)}")
for field in ("gender", "masterCategory", "subCategory", "brand", "material"):
    values = {p.get(field) for p in records if p.get(field)}
    print(f"{field}: {len(values)} distinct values")
