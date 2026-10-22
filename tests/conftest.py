import dataclasses

import pytest

from src.config import get_config
from src.embeddings import HashingEmbedder
from src.engine import load_engine
from src.data_loader import load_dataset_table, table_to_records
from src.preprocessing import preprocess_records
from src.config import PROJECT_ROOT

ARROW_DATASET = PROJECT_ROOT / "data/source/data-00000-of-00001.arrow"
SHOP_DATASET = PROJECT_ROOT / "data/myntra/myntra-dataset-sample-main/Myntra products .csv"


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    base = get_config()
    return dataclasses.replace(base, dataset_path=SHOP_DATASET,
                               collection_name="test_fashion_products",
                               chroma_dir=tmp_path_factory.mktemp("chroma"),
                               products_json_path=tmp_path_factory.mktemp("json") / "products.json",
                               dev_subset_size=0)


@pytest.fixture(scope="session")
def loaded():
    if not ARROW_DATASET.exists():
        pytest.skip("dataset not available")
    table = load_dataset_table(ARROW_DATASET)
    records, report = preprocess_records(table_to_records(table))
    return records, report, table


@pytest.fixture(scope="session")
def records(engine):
    return engine.catalog.records


@pytest.fixture(scope="session")
def engine(cfg):
    if not cfg.dataset_path.exists():
        pytest.skip("dataset not available")
    return load_engine(cfg, embedder=HashingEmbedder())  # offline embedder: tests need no downloads


@pytest.fixture(scope="session")
def retriever(engine):
    return engine.retriever
