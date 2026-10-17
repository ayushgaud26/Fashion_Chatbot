"""Embedding backends.

* SentenceTransformerEmbedder - the real model (default sentence-transformers/all-MiniLM-L6-v2).
* HashingEmbedder - deterministic, download-free *fallback/test* embedder (word + character n-gram
  hashing). It is NOT a semantic model; it exists so the pipeline and tests run offline.
The active embedder's name is stored on the Chroma collection so vectors from different models are never mixed."""
from __future__ import annotations

import hashlib
import logging
import math

from .preprocessing import normalize_text, stem

log = logging.getLogger(__name__)


class EmbeddingBackendError(RuntimeError):
    pass


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingBackendError("sentence-transformers is not installed (pip install sentence-transformers).") from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingBackendError(f"Could not load embedding model '{model_name}': {exc}") from exc
        self.name = model_name
        self.is_fallback = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim} (offline fallback, not semantic)"
        self.is_fallback = True

    def _features(self, text: str):
        toks = [stem(t) for t in normalize_text(text).split()]
        for t in toks:
            yield t, 1.0
            padded = f"#{t}#"
            for i in range(len(padded) - 2):
                yield padded[i : i + 3], 0.3
        for a, b in zip(toks, toks[1:]):
            yield f"{a} {b}", 0.5

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat, w in self._features(text):
                h = hashlib.md5(feat.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                vec[idx] += sign * w
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder(backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    backend = (backend or "auto").lower()
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "auto":
        try:
            return SentenceTransformerEmbedder(model_name)
        except EmbeddingBackendError as exc:
            log.warning("Falling back to the offline hashing embedder: %s", exc)
            return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use auto, sentence-transformers or hashing).")
