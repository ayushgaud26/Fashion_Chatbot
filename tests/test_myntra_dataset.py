import dataclasses
from src.config import PROJECT_ROOT, get_config
from src.embeddings import HashingEmbedder
from src.engine import load_engine
from src.ingestion import load_records


DATASET = PROJECT_ROOT / "data/myntra/myntra-dataset-sample-main/Myntra products .csv"


def test_myntra_csv_maps_fields_and_quality_report():
    records, report, frame = load_records(dataclasses.replace(get_config(), dataset_path=DATASET))
    assert len(records) == report.total_rows == 1000
    assert len(frame.columns) == 24
    assert report.invalid_id_rows == report.duplicate_ids_dropped == 0
    p = records[0]
    assert p["id"] == int(frame.iloc[0]["product_id"])
    assert p["productUrl"].startswith("https://www.myntra.com/")
    assert p["imageUrl"].startswith("https://assets.myntassets.com/")
    assert p["price"] is not None and p["originalPrice"] is not None
    assert p["rating"] is not None and p["reviewCount"] is not None
    assert isinstance(p["specifications"], dict) and isinstance(p["sourceFields"], dict)


def test_myntra_index_cards_and_filters_use_only_myntra_records(tmp_path):
    cfg = dataclasses.replace(get_config(), dataset_path=DATASET,
                              collection_name="test_myntra_products",
                              chroma_dir=tmp_path / "chroma", products_json_path=tmp_path / "products.json",
                              embedding_backend="hashing", dev_subset_size=0)
    engine = load_engine(cfg, embedder=HashingEmbedder())
    assert engine.total_rows == engine.store.count() == len(engine.catalog.records) == 1000
    assert all(p["retailer"] == "Myntra" for p in engine.catalog.records)
    p = engine.catalog.records[0]
    result = engine.retriever.search(str(p["id"]))
    assert result.products[0]["productUrl"] == p["productUrl"]
    assert engine.images.get_url(p["id"]) == p["imageUrl"]

    sizes = [r for r in engine.catalog.records if "M" in (r.get("sizeValues") or [])]
    assert sizes
    sized = engine.retriever.search("size M")
    assert sized.products and all("M" in (r.get("sizeValues") or []) for r in sized.products)

    rated = engine.retriever.search("shirts rating above 4.0")
    assert rated.products and all((r.get("rating") or 0) >= 4.0 for r in rated.products)
