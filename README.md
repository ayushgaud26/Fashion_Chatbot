# Fashion Product RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for **Myntra product discovery**, using the supplied Myntra CSV
as its primary catalogue. It supports product lookup, structured filters, semantic search, comparison, and cards
with dataset images, prices, discounts, ratings, sizes, materials, and Myntra links.

## Dataset
The configured file is `data/myntra/myntra-dataset-sample-main/Myntra products .csv` (the supplied ZIP sample).
It contains 1,000 rows and 24 columns. The loader maps product IDs, names, brands, categories, sizes, materials and
specifications, prices, discounts, ratings/review counts, image URLs, and product URLs into the existing catalog
pipeline while retaining original source fields in the normalized export. Missing values remain missing; stock is
not inferred from delivery-option text. Prices/ratings are the dataset snapshot, not live Myntra values.

Set `DATASET_PATH` in `.env` to select another Myntra CSV. Arrow input remains supported for the previous catalog.
Myntra uses the separate `fashion_products_myntra` Chroma collection, so its IDs/vectors do not mix with old data.
The old retail demo catalog is excluded when a CSV is selected.

## Install on Windows (PowerShell)
```powershell
cd fashion-rag-chatbot
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
copy .env.example .env              # then edit .env and add GEMINI_API_KEY or OPENAI_API_KEY
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

## Workflow
```powershell
python scripts/validate_dataset.py     # inspect schema, counts, missing values, and URL schemes
python scripts/build_index.py          # load → clean → data/products.json → embed → ChromaDB (idempotent)
python scripts/build_index.py --rebuild   # drop and rebuild from scratch
python scripts/build_index.py --limit 500 # development subset
streamlit run app.py                   # open http://localhost:8501
python -m pytest -q                    # tests (no paid API calls, no model download needed)
python scripts/run_demo_queries.py     # runs the 10 demo queries, verifies against raw data → docs/demo_results.md
```
The app also builds the index automatically on first start if it is missing.

## Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `gemini`, `openai`, or `auto` (uses OpenAI when its key is set, otherwise Gemini) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | – / `gemini-3.5-flash-lite` | Gemini API key and model; Gemini offers a free tier with model-specific limits |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | – / `gpt-4o-mini` | Optional OpenAI API key and model |
| `LLM_TEMPERATURE` | `0.1` | LLM response temperature |
| `EMBEDDING_BACKEND` | `auto` | `auto` tries sentence-transformers then falls back to an offline hashing embedder; `sentence-transformers` is strict; `hashing` is offline/test only |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DATASET_PATH`, `CHROMA_DIR`, `COLLECTION_NAME`, `RETAIL_CATALOG_PATH` | see `.env.example` | Locations |
| `BATCH_SIZE`, `DEV_SUBSET_SIZE`, `MAX_RESULTS`, `SEMANTIC_TOP_K`, `SEMANTIC_MAX_DISTANCE` | `256`, `0`, `50`, `10`, `0.8` | Indexing and retrieval limits |

If the embedder changes, the collection is rebuilt automatically (vectors from different models must not be mixed).

## Optional retail demo catalogue and chat memory
`data/retail_products.json` is a small India-market demo catalogue, separate from the historical fashion dataset.
Each retail record includes a retailer URL, INR price, original listed price, and `priceCheckedAt`; refresh values
from the retailer before presenting them as current. Product details are taken from the linked retailer pages.
Add or edit records in this JSON file using the same fields. The app validates IDs, prices and HTTPS product URLs.
It is enabled only with the Arrow source; selecting the Myntra CSV keeps the catalogue Myntra-only.

Chat messages and retrieved product context are stored locally in `data/chat_history.sqlite3`, keyed by a random
conversation ID in the app URL. The app sends up to the latest 20 messages to the LLM as conversational context.
Use **Reset chat** to delete the current conversation from the local database. The database is ignored by Git.

## Example queries
`Find men jeans under 1500` · `Find white women tshirts` · `Find cotton shirts` ·
`shirts rating above 4.0` · `shirts discount above 40%` · `show size M` ·
`Which one is cheaper?` after retrieving products · a Myntra product ID or exact name.

## Myntra CSV
The supplied ZIP is extracted under `data/myntra/`. CSV images are loaded from their source URLs when product
cards render; the app does not download them during indexing. Set the dataset path in `.env` if you move the CSV.

## Troubleshooting
- **Sidebar shows "hashing" embedder / warning** – sentence-transformers or the model download is unavailable. Install it and check internet access to huggingface.co, then re-run with `EMBEDDING_BACKEND=sentence-transformers` to see the exact error.
- **No AI answers** – the selected provider key is missing or the API call failed; the response says which.
- **Dataset not found** – check `DATASET_PATH`.
- **Stale or odd results after changing models** – `python scripts/build_index.py --rebuild`.
- **PowerShell won't activate the venv** – `Set-ExecutionPolicy -Scope Process RemoteSigned`.

## Known limitations / honest status
- This ZIP is a sample of 1,000 Myntra product records, not the full catalog.
- The CSV has no stock-status column or standalone color column. Color is extracted only when a color word appears explicitly in the product description; stock is never inferred. Some records lack gender, color, material, or discount.
- Prices, discounts, ratings, and product links are a static snapshot; live Myntra availability and URL responses were not verified in this environment.
- The current local `.env` selects the offline hashing embedder because model downloads from Hugging Face are unavailable in this environment. It is lexical, not semantic. Set `EMBEDDING_BACKEND=auto` with network access to try the configured SentenceTransformer model.
- The OpenAI call was tested only with a mocked client, never against the live API.
- Synonyms (`sportswear`, `clothing`, `sneakers`, `pants`, …) are a small documented table in `src/filters.py`; unknown slang is not understood by the structured layer.
- Navy Blue is a separate colour from Blue (exact dataset values); "blue jeans" does not include Navy Blue.
- No image upload / visual recognition is implemented; images are only displayed.
