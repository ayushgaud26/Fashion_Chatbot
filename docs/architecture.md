# Architecture

```
Arrow file ──► data_loader ──► preprocessing ──► Catalog (in-memory records, vocabulary)   ─┐
                    │                │                                                      │
                    │                └─► IndexDoc (text + metadata + hash) ─► embeddings ─► ChromaDB (persistent)
                    └─► ImageStore (lazy image bytes, display only)                          │
User question ─► retrieval.HybridRetriever ──(1 ID, 2 exact name, 3 filters, 4 semantic)──────┘
                    └─► RetrievalResult ─► llm_service (OpenAI, grounded) ─► response_formatter ─► Streamlit
                                              └─ on error / no key / bad IDs ─► catalogue fallback
```

**Data ingestion** (`data_loader.py`, `ingestion.py`): reads the Arrow stream with PyArrow (memory-mapped; falls
back to `datasets`). Validates columns, drops invalid/duplicate IDs (first kept), exports normalised JSON
(`data/products.json`, no image bytes) and syncs the vector index. Reports rows, missing values and index changes.

**Preprocessing** (`preprocessing.py`): trims/collapses whitespace, converts NaN/blank to `None`, year float → int.
Original display values are preserved; `normalize_text` (lower-case, apostrophes removed, punctuation → spaces) is used
only for matching. Searchable text is built from fields that exist, e.g. `Product ID: 15970. Product name: … Gender: Men. …`.
Missing fields are omitted, never filled.

**Embeddings** (`embeddings.py`): `all-MiniLM-L6-v2` via sentence-transformers (normalised vectors). An offline
`HashingEmbedder` exists purely so tests/demos run without downloads; it is labelled as non-semantic everywhere.

**ChromaDB storage** (`vector_store.py`): persistent client, cosine space, one document per product, id = product ID,
metadata = all structured fields + `content_hash`. `sync()` compares hashes: new → add, changed → update, unchanged → skip,
removed from source → delete. Hence no duplicates on repeated ingestion and safe re-indexing. Batches are configurable;
a failed batch is logged and counted. The collection stores the embedder name; a mismatch triggers a rebuild.

**Exact matching** (`retrieval.py`, `filters.Catalog`): product IDs (bare number, `id 123`, `#123`, or a known ID in the text);
exact names (whole query minus a command word, or a full name of ≥3 words contained in the query). Partial names never count.
If several products share the exact name, all are returned with a note.

**Structured filters** (`filters.py`): vocabulary is built from the data; a plural-insensitive, longest-phrase-first matcher
maps words to fields (`navy blue`, `formal shoes`). Same-field matches are OR-ed, different fields AND-ed. Applied directly to the records, returning
*all* matches up to `MAX_RESULTS` (with total count and truncation note). Zero matches → explicit no-match.

**Dense retrieval**: used for descriptive queries. With filters present, Chroma is queried with the *same* filters as a `where`
clause, so semantic ranking can only reorder products that already satisfy the constraints. Name keywords that appear in a
product name are boosted. Without filters, pure semantic results are capped by `SEMANTIC_TOP_K` and `SEMANTIC_MAX_DISTANCE`
and are labelled as suggestions, not verified matches.

**LLM generation** (`llm_service.py`): system prompt forbids invention; temperature 0.1; ≤15 records as JSON context;
errors are caught and API keys redacted; if the answer mentions an ID outside the retrieved set it is rejected.

**Response formatting & fallback** (`response_formatter.py`): every answer is labelled either *AI-generated* or
*Catalogue-search fallback*; no-match/clarification messages are deterministic; source product IDs are always listed;
a note appears when the user asks for price/size/stock/brand.
