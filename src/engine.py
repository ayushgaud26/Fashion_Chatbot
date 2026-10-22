"""Wires every component together (used by the Streamlit app and the demo script)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config, get_config
from .data_loader import ImageStore
from .embeddings import get_embedder
from .filters import Catalog
from .ingestion import index_records, load_records
from .llm_service import LLMService
from .retrieval import HybridRetriever
from .retail_catalog import load_retail_products
from .vector_store import VectorStore


@dataclass
class Engine:
    config: Config
    catalog: Catalog
    images: ImageStore
    embedder: object
    store: VectorStore
    retriever: HybridRetriever
    llm: LLMService
    total_rows: int
    preprocess_summary: str
    indexed_now: int


def load_engine(cfg: Config | None = None, embedder=None, auto_index: bool = True,
                include_retail: bool = False) -> Engine:
    cfg = cfg or get_config()
    records, prep, table = load_records(cfg)
    if include_retail:
        retail = load_retail_products(cfg.retail_catalog_path)
        existing_ids = {r["id"] for r in records}
        duplicates = existing_ids.intersection(r["id"] for r in retail)
        if duplicates:
            raise ValueError(f"Retail product IDs overlap the source catalogue: {sorted(duplicates)}")
        records.extend(retail)
    embedder = embedder or get_embedder(cfg.embedding_backend, cfg.embedding_model)
    store = VectorStore(cfg.chroma_dir, cfg.collection_name, embedder)
    indexed_now = 0
    if auto_index:
        rep = index_records(records, store, cfg.batch_size)
        indexed_now = rep.added + rep.updated
    catalog = Catalog(records)
    retriever = HybridRetriever(catalog, store, cfg.max_results, cfg.semantic_top_k, cfg.semantic_max_distance)
    llm = LLMService(cfg.llm_api_key, cfg.llm_model, cfg.llm_temperature,
                     provider=cfg.llm_provider)
    return Engine(cfg, catalog, ImageStore(table if hasattr(table, "column_names") else None, records), embedder, store, retriever, llm,
                  prep.total_rows, prep.to_text(), indexed_now)
