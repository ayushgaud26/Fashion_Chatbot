"""Ingestion pipeline: load -> validate -> clean -> JSON export -> embed -> persistent Chroma index."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config, get_config
from .data_loader import load_dataset_table, table_to_records
from .embeddings import get_embedder
from .preprocessing import PreprocessReport, make_index_doc, preprocess_records, validate_columns
from .vector_store import SyncReport, VectorStore
from .myntra_dataset import load_myntra_csv

log = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    dataset_path: str
    embedder: str
    preprocess: PreprocessReport
    sync: SyncReport
    collection_count: int
    json_path: str | None
    seconds: float
    rebuilt: bool

    def to_text(self) -> str:
        s = self.sync
        return "\n".join([
            f"Dataset: {self.dataset_path}", f"Embedder: {self.embedder}",
            f"Preprocessing: {self.preprocess.to_text()}",
            f"Index sync: added={s.added}, updated={s.updated}, unchanged={s.unchanged}, deleted={s.deleted}, "
            f"failed batches={s.failed_batches}",
            f"Collection now holds {self.collection_count} documents"
            + (" (rebuilt from scratch)" if self.rebuilt else ""),
            f"Normalised JSON: {self.json_path or 'not written'}", f"Time: {self.seconds:.1f}s",
        ])


def load_records(cfg: Config, limit: int | None = None):
    """Load -> validate -> preprocess. Returns (records, report, arrow_table)."""
    if cfg.dataset_path.suffix.lower() == ".csv":
        records, rep, frame = load_myntra_csv(cfg.dataset_path)
        cap = limit or cfg.dev_subset_size
        if cap:
            records = records[:cap]
            rep.valid_records = len(records)
        return records, rep, frame
    table = load_dataset_table(cfg.dataset_path)
    validate_columns(table.column_names)
    records, rep = preprocess_records(table_to_records(table, limit or cfg.dev_subset_size or None))
    return records, rep, table


def export_json(records: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(path)


def index_records(records: list[dict], store: VectorStore, batch_size: int) -> SyncReport:
    return store.sync([make_index_doc(r) for r in records], batch_size=batch_size)


def build_index(cfg: Config | None = None, *, limit: int | None = None, rebuild: bool = False,
                embedder=None, write_json: bool = True) -> IngestionReport:
    cfg = cfg or get_config()
    t0 = time.time()
    records, prep, _ = load_records(cfg, limit)
    embedder = embedder or get_embedder(cfg.embedding_backend, cfg.embedding_model)
    store = VectorStore(cfg.chroma_dir, cfg.collection_name, embedder)
    if rebuild:
        store.reset()
    sync = index_records(records, store, cfg.batch_size)
    path = export_json(records, cfg.products_json_path) if write_json else None
    return IngestionReport(str(cfg.dataset_path), embedder.name, prep, sync, store.count(), path,
                           time.time() - t0, rebuilt=rebuild or store.was_reset)
