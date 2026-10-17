import pytest

from src.preprocessing import (EXPECTED_COLUMNS, DatasetValidationError, build_search_text, preprocess_records,
                               validate_columns)


def test_loads_and_has_expected_schema(loaded):
    records, rep, table = loaded
    validate_columns(table.column_names)
    assert set(EXPECTED_COLUMNS) <= set(table.column_names)
    assert rep.valid_records == len(records) == table.num_rows
    assert rep.duplicate_ids_dropped == 0


def test_missing_column_rejected():
    with pytest.raises(DatasetValidationError):
        validate_columns(["id", "gender"])


def test_duplicates_missing_and_invalid_ids_handled():
    raw = [
        {"id": 1, "gender": "Men", "productDisplayName": "A B", "year": 2012.0},
        {"id": 1, "gender": "Women", "productDisplayName": "Duplicate"},
        {"id": None, "gender": "Men"},
        {"id": 2, "gender": "  ", "baseColour": float("nan"), "year": float("nan")},
    ]
    recs, rep = preprocess_records(raw)
    assert [r["id"] for r in recs] == [1, 2]
    assert recs[0]["gender"] == "Men"  # first occurrence kept
    assert (rep.duplicate_ids_dropped, rep.invalid_id_rows) == (1, 1)
    assert recs[1]["gender"] is None and recs[1]["baseColour"] is None and recs[1]["year"] is None
    assert rep.missing_by_field["gender"] >= 1


def test_missing_attributes_not_fabricated():
    text = build_search_text({"id": 7, "productDisplayName": "Thing", "gender": None, "baseColour": None, "year": None})
    assert "Gender" not in text and "Colour" not in text and "Year" not in text
    assert "Product ID: 7" in text and "Thing" in text
    assert "price" not in text.lower()


def test_original_ids_and_values_preserved(loaded):
    records, _, table = loaded
    raw_ids = table.column("id").to_pylist()
    assert [r["id"] for r in records] == raw_ids
    assert records[0]["productDisplayName"] == table.column("productDisplayName")[0].as_py()


def test_images_are_real_bytes(engine, records):
    assert engine.images.get_bytes(records[0]["id"]) is not None
    assert engine.images.get_bytes(-1) is None
