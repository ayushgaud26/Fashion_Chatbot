"""Persistent ChromaDB wrapper: one document per product, id = original product id (as string).

Idempotent sync: each document stores a content hash, so re-running ingestion adds nothing twice,
updates only changed records and removes records that disappeared from the source."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .preprocessing import IndexDoc

log = logging.getLogger(__name__)


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed_batches: int = 0


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str, embedder):
        self.embedder = embedder
        self.collection_name = collection_name
        self.was_reset = False
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
        self.collection = self._open()

    def _open(self):
        try:
            col = self.client.get_collection(self.collection_name)
        except Exception:
            col = None
        if col is not None and (col.metadata or {}).get("embedder") != self.embedder.name:
            log.warning("Collection was built with a different embedder (%s); rebuilding for %s.",
                        (col.metadata or {}).get("embedder"), self.embedder.name)
            self.client.delete_collection(self.collection_name)
            col, self.was_reset = None, True
        if col is None:
            col = self.client.create_collection(
                self.collection_name, metadata={"embedder": self.embedder.name, "hnsw:space": "cosine"}
            )
        return col

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            self.collection_name, metadata={"embedder": self.embedder.name, "hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def _existing_hashes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        offset, page = 0, 5000
        while True:
            res = self.collection.get(include=["metadatas"], limit=page, offset=offset)
            ids = res["ids"]
            if not ids:
                break
            for i, m in zip(ids, res["metadatas"]):
                out[i] = (m or {}).get("content_hash", "")
            offset += len(ids)
        return out

    def sync(self, docs: list[IndexDoc], batch_size: int = 256, delete_missing: bool = True) -> SyncReport:
        rep = SyncReport()
        existing = self._existing_hashes()
        wanted = {str(d.id) for d in docs}
        todo = []
        for d in docs:
            old = existing.get(str(d.id))
            if old is None:
                todo.append((d, True))
            elif old != d.content_hash:
                todo.append((d, False))
            else:
                rep.unchanged += 1
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            try:
                embs = self.embedder.embed([d.text for d, _ in chunk])
                self.collection.upsert(
                    ids=[str(d.id) for d, _ in chunk], embeddings=embs,
                    documents=[d.text for d, _ in chunk], metadatas=[d.metadata for d, _ in chunk],
                )
                rep.added += sum(1 for _, new in chunk if new)
                rep.updated += sum(1 for _, new in chunk if not new)
            except Exception:
                rep.failed_batches += 1
                log.exception("Failed to index batch starting at %d (ids %s...)", start, [d.id for d, _ in chunk[:3]])
        if delete_missing:
            stale = [i for i in existing if i not in wanted]
            for s in range(0, len(stale), 5000):
                self.collection.delete(ids=stale[s : s + 5000])
            rep.deleted = len(stale)
        return rep

    def query(self, text: str, n_results: int, where: dict | None = None) -> list[tuple[int, float]]:
        """Return [(product_id, cosine_distance)] ordered best-first. Empty text -> []."""
        if not text or not text.strip():
            return []
        n = min(n_results, self.count())
        if n <= 0:
            return []
        emb = self.embedder.embed([text])
        res = self.collection.query(query_embeddings=emb, n_results=n, where=where, include=["distances"])
        return [(int(i), float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]
