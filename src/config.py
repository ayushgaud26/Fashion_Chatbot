"""Central configuration. Every value can be overridden with an environment variable
(or a .env file). API keys are read from the environment only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Config:
    dataset_path: Path
    products_json_path: Path
    retail_catalog_path: Path
    chroma_dir: Path
    collection_name: str
    embedding_backend: str      # auto | sentence-transformers | hashing
    embedding_model: str
    batch_size: int
    dev_subset_size: int        # 0 = all records
    max_results: int            # cap for "show all matching" queries
    semantic_top_k: int
    semantic_max_distance: float
    llm_provider: str
    llm_model: str
    llm_fallback_model: str | None
    llm_temperature: float
    llm_api_key: str | None  # read from the environment only


def get_config() -> Config:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider == "auto":
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "gemini"
    if provider not in {"openai", "gemini"}:
        raise ValueError("LLM_PROVIDER must be 'auto', 'openai' or 'gemini'.")
    if provider == "openai":
        llm_api_key = os.getenv("OPENAI_API_KEY") or None
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        llm_api_key = os.getenv("GEMINI_API_KEY") or None
        llm_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    dataset_path = _path("DATASET_PATH", "data/myntra/myntra-dataset-sample-main/Myntra products .csv")
    preferred_csv = PROJECT_ROOT / "data" / "myntra" / "myntra-dataset-sample-main" / "Myntra products .csv"
    # The Arrow sample has blurry thumbnails and lacks prices/review counts. If a
    # deployment still points DATASET_PATH at Arrow, prefer the enriched CSV shared
    # with this project so product cards show its URLs, prices and ratings.
    if dataset_path.suffix.lower() != ".csv" and preferred_csv.exists():
        dataset_path = preferred_csv
    return Config(
        dataset_path=dataset_path,
        products_json_path=_path("PRODUCTS_JSON_PATH", "data/products.json"),
        retail_catalog_path=_path("RETAIL_CATALOG_PATH", "data/retail_products.json"),
        chroma_dir=_path("CHROMA_DIR", "chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "fashion_products_myntra"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto").lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=_int("BATCH_SIZE", 256),
        dev_subset_size=_int("DEV_SUBSET_SIZE", 0),
        max_results=_int("MAX_RESULTS", 50),
        semantic_top_k=_int("SEMANTIC_TOP_K", 10),
        semantic_max_distance=_float("SEMANTIC_MAX_DISTANCE", 0.8),
        llm_provider=provider,
        llm_model=llm_model,
        llm_fallback_model=(os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash").strip()
                            or "gemini-3.8-flash")
        if provider == "gemini" else None,
        llm_temperature=_float("LLM_TEMPERATURE", 0.1),
        llm_api_key=llm_api_key,
    )
