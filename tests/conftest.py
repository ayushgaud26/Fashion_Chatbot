import dataclasses

import pytest

from src.config import get_config
from src.embeddings import HashingEmbedder
from src.engine import load_engine
from src.ingestion import load_records
from src.config import PROJECT_ROOT


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    base = get_config()
    # Keep the existing Arrow regression tests isolated from the app's selected Myntra source.
    return dataclasses.replace(base, dataset_path=PROJECT_ROOT / "data/source/data-00000-of-00001.arrow",
                               collection_name="test_fashion_products",
                               chroma_dir=tmp_path_factory.mktemp("chroma"),
                               products_json_path=tmp_path_factory.mktemp("json") / "products.json",
                               dev_subset_size=0)


@pytest.fixture(scope="session")
def loaded(cfg):
    if not cfg.dataset_path.exists():
        pytest.skip("dataset not available")
    return load_records(cfg)  # (records, report, table)


@pytest.fixture(scope="session")
def records(loaded):
    return loaded[0]


@pytest.fixture(scope="session")
def engine(cfg):
    if not cfg.dataset_path.exists():
        pytest.skip("dataset not available")
    return load_engine(cfg, embedder=HashingEmbedder())  # offline embedder: tests need no downloads


@pytest.fixture(scope="session")
def retriever(engine):
    return engine.retriever
