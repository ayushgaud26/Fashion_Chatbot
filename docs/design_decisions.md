# Design decisions

**Why RAG.** An LLM alone does not know this catalogue and will happily invent products. RAG retrieves real records first and
lets the LLM only phrase an answer from them; retrieval also works without an LLM (the fallback).

**Why ChromaDB.** Embedded, no server, persistent on disk, stores vectors + metadata + metadata filtering together, trivial to
install on Windows, and comfortably handles tens of thousands of short records. For millions of products or multi-user serving
a managed/server vector DB would be considered.

**Why all-MiniLM-L6-v2.** Small (~80 MB), fast on CPU, 384 dimensions, good general sentence similarity for short texts, widely used and easy to explain. A fashion-tuned or multilingual model could be swapped in via `EMBEDDING_MODEL` (the index rebuilds automatically).
Embeddings turn text into vectors so "comfortable everyday clothing" can land near casual apparel even without shared words.

**Why record-level chunking.** Each product is a ~20-word structured record. Splitting it would separate name, colour and article type
and hurt both retrieval and filtering; one record = one retrieval unit, keyed by the original product ID. If long descriptions appear
later, add chunking with size/overlap and keep the product ID in each chunk's metadata.

**Keyword vs dense search.** Keyword/exact search is precise for IDs, names and attributes but misses paraphrases; dense search
handles vague intent but can't guarantee "blue" or "men". So both are used, with exact → structured → semantic priority.

**How exact filters improve reliability.** "Show blue men's jeans" is answered by filtering the actual fields, not by similarity,
so every result truly satisfies every constraint and *all* matches can be listed. Semantic ranking only reorders results inside the filtered set.

**How hallucinations are reduced.** (1) retrieval before generation; (2) the LLM sees only retrieved records; (3) strict system prompt;
(4) temperature 0.1; (5) answers citing IDs outside the retrieved set are discarded; (6) no-match and missing-field cases bypass the LLM
with deterministic messages; (7) every response lists source IDs and is labelled AI vs fallback.

**Why missing attributes must not be invented.** Fabricated prices, sizes or stock would look authoritative and mislead users.
The dataset has no such fields, so the app says so; absent fields are omitted from embeddings text, cards and answers.

**Other trade-offs.** Offline hashing embedder exists for reproducible tests, at the cost that real semantic behaviour must be validated separately with MiniLM.
Structured search uses an in-memory scan (fast at 44k records); at much larger scale move filtering into the database.
