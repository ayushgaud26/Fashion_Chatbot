# Fashion Product RAG Chatbot

A Streamlit shopping assistant for searching a Myntra product catalogue. It combines structured filters and semantic retrieval, then presents matching catalogue records as product cards with images, prices, discounts, ratings, sizes, and links.

## Catalogue

The default dataset is the 1,000-row, 24-column Myntra sample at `data/myntra/myntra-dataset-sample-main/Myntra products .csv`. The loader maps product IDs, names, brands, categories, sizes, materials, prices, discounts, ratings, image URLs, and product URLs into the catalogue while preserving source fields. Missing values stay missing, and stock is never inferred from delivery text.

Prices, ratings, discounts, and product links come from a dataset snapshot; they are not live Myntra values. The sample does not include stock status or a standalone colour column. Some fields are missing for some products.

Set `DATASET_PATH` in `.env` to use a different Myntra CSV. Arrow input is also supported for the previous catalogue. Myntra uses a separate Chroma collection, `fashion_products_myntra`.

## Install (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add a Gemini or OpenAI API key to .env
```

On macOS or Linux, create and activate a virtual environment with `python3 -m venv .venv` and `source .venv/bin/activate`.

## Run

```powershell
python scripts/validate_dataset.py
python scripts/build_index.py
streamlit run app.py
```

The app builds the index on first start if it is missing. To rebuild it from scratch, run `python scripts/build_index.py --rebuild`. For a smaller development index, use `python scripts/build_index.py --limit 500`.

Run the test suite with `python -m pytest -q`. The demo query report can be regenerated with `python scripts/run_demo_queries.py`.

## Configuration

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini`, `openai`, or `auto` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini credentials and model |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Optional OpenAI credentials and model |
| `LLM_TEMPERATURE` | Response temperature |
| `EMBEDDING_BACKEND` | `auto`, `sentence-transformers`, or offline `hashing` fallback |
| `EMBEDDING_MODEL` | SentenceTransformer model name |
| `DATASET_PATH`, `CHROMA_DIR`, `COLLECTION_NAME`, `RETAIL_CATALOG_PATH` | Dataset and index locations |
| `BATCH_SIZE`, `DEV_SUBSET_SIZE`, `MAX_RESULTS`, `SEMANTIC_TOP_K`, `SEMANTIC_MAX_DISTANCE` | Indexing and retrieval limits |

Copy `.env.example` to `.env` and keep API keys private. If the embedding model changes, rebuild the collection so vectors from different models are not mixed.

An optional small India-market retail catalogue lives at `data/retail_products.json`. Its prices should be refreshed from the retailer before being presented as current. It is enabled with the Arrow source; selecting the Myntra CSV keeps the catalogue Myntra-only.

## Example searches

- `Find men jeans under 1500`
- `Find white women tshirts`
- `Find cotton shirts`
- `shirts rating above 4.0`
- `shirts discount above 40%`
- `show size M`
- Ask “Which one is cheaper?” after a product search
- Search by Myntra product ID or exact product name

## Chat history

Chat messages and retrieved product context are stored in `data/chat_history.sqlite3`, keyed by a random conversation ID in the app URL. Up to the latest 20 messages are sent to the language model as conversation context. Use **Reset chat** to clear the current conversation. The database is ignored by Git.

## Limitations

- The included dataset is a 1,000-record sample, not the full Myntra catalogue.
- Prices, ratings, discounts, and links may be outdated; availability is not verified.
- Colour is extracted only when a colour word appears in the product description. Navy Blue and Blue are separate dataset values.
- The app has a small deterministic synonym table; unfamiliar slang may not be understood by the structured search layer.
- Images are displayed from catalogue URLs; image upload and visual recognition are not implemented.
- If the sentence-transformer model cannot be loaded, `auto` may fall back to hashing embeddings, which provide lexical rather than semantic search. The app indicates the active embedder in its status area.

## Troubleshooting

- **No AI answers:** Check the selected provider and API key, then restart Streamlit.
- **Dataset not found:** Check `DATASET_PATH`.
- **Unexpected results after changing embedding models:** Run `python scripts/build_index.py --rebuild`.
- **PowerShell blocks activation:** Run `Set-ExecutionPolicy -Scope Process RemoteSigned` in that terminal.
