import dataclasses

from src.embeddings import HashingEmbedder
from src.ingestion import build_index
from src.llm_service import LLMService
from src.response_formatter import compose
from src.vector_store import VectorStore


def test_dense_semantic_retrieval_returns_ranked_ids(engine, records):
    hits = engine.store.query("blue denim jeans", 5)
    assert len(hits) == 5 and all(h[0] in engine.catalog.by_id for h in hits)
    assert [d for _, d in hits] == sorted(d for _, d in hits)


def test_price_no_match_keeps_category_filters(retriever):
    result = retriever.search("Show women's kurtas under 0")
    assert result.method == "no_match"
    assert result.filters.get("gender") == ["Women"]
    assert result.filters.get("subCategory") == ["Kurtas"]


def test_empty_query_to_vector_store(engine):
    assert engine.store.query("   ", 5) == []


def test_index_is_persistent_and_complete(engine, cfg, records):
    assert engine.store.count() == len(records)
    reopened = VectorStore(cfg.chroma_dir, cfg.collection_name, HashingEmbedder())
    assert reopened.count() == len(records)
    got = reopened.collection.get(ids=[str(records[0]["id"])], include=["metadatas"])
    meta = got["metadatas"][0]
    assert meta["id"] == records[0]["id"] and meta["articleType"] == records[0]["articleType"]


def test_repeated_ingestion_adds_no_duplicates(cfg, records):
    r1 = build_index(cfg, embedder=HashingEmbedder())
    r2 = build_index(cfg, embedder=HashingEmbedder())
    assert r2.sync.added == 0 and r2.sync.updated == 0 and r2.sync.unchanged == len(records)
    assert r2.collection_count == len(records)


def test_subset_reindex_removes_stale_documents(cfg):
    sub = dataclasses.replace(cfg, chroma_dir=cfg.chroma_dir / "subset")
    full = build_index(sub, embedder=HashingEmbedder(), write_json=False)
    small = build_index(sub, limit=100, embedder=HashingEmbedder(), write_json=False)
    assert small.collection_count == 100 and small.sync.deleted == full.collection_count - 100


def test_json_export_roundtrip(cfg, records):
    import json
    build_index(cfg, embedder=HashingEmbedder())
    data = json.loads(cfg.products_json_path.read_text(encoding="utf-8"))
    assert len(data) == len(records) and data[0]["id"] == records[0]["id"]


# ---- LLM: mocked, no paid calls -----------------------------------------------------------
class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]
class _Completions:
    def __init__(self, behaviour): self.behaviour, self.calls = behaviour, []
    def create(self, **kw):
        self.calls.append(kw)
        behaviour = self.behaviour.pop(0) if isinstance(self.behaviour, list) else self.behaviour
        if isinstance(behaviour, Exception):
            raise behaviour
        return _Resp(behaviour)
class _Client:
    def __init__(self, behaviour):
        self.chat = type("C", (), {"completions": _Completions(behaviour)})()


class _GeminiResponse:
    def __init__(self, text): self.text = text


class _GeminiModels:
    def __init__(self, behaviour): self.behaviour, self.calls = behaviour, []
    def generate_content(self, **kw):
        self.calls.append(kw)
        behaviour = self.behaviour.pop(0) if isinstance(self.behaviour, list) else self.behaviour
        if isinstance(behaviour, Exception):
            raise behaviour
        return _GeminiResponse(behaviour)


class _GeminiClient:
    def __init__(self, behaviour): self.models = _GeminiModels(behaviour)


class _ServiceUnavailable(RuntimeError):
    status_code = 503


def test_llm_answer_used_and_labelled(retriever):
    res = retriever.search("Show black shirts")
    pid = res.ids[0]
    client = _Client(f"Here is one option: product {pid}.")
    ans = LLMService("sk-test", client=client).generate("Show black shirts", res)
    assert ans.source == "llm"
    kw = client.chat.completions.calls[0]
    assert kw["temperature"] <= 0.2 and "only the supplied" in kw["messages"][0]["content"]
    out = compose(res, ans)
    assert "AI-generated" in out.markdown and str(pid) in out.markdown and out.source == "llm"


def test_llm_api_failure_falls_back(retriever):
    res = retriever.search("Show black shirts")
    ans = LLMService("sk-secret-123", client=_Client(RuntimeError("boom sk-secret-123"))).generate("q", res)
    assert ans.source == "fallback" and "sk-secret-123" not in ans.error
    out = compose(res, ans)
    assert "fallback" in out.markdown.lower() and "AI-generated" not in out.markdown
    assert all(str(i) in out.markdown for i in res.ids[:10])


def test_gemini_answer_uses_configured_model_and_catalogue_context(retriever):
    res = retriever.search("Show black shirts")
    client = _GeminiClient("Recovered grounded answer.")
    answer = LLMService(
        "k", model="gemini-3.5-flash-lite", provider="gemini",
        client=client,
    ).generate("Show black shirts", res)
    calls = client.models.calls
    assert [call["model"] for call in calls] == ["gemini-3.5-flash-lite"]
    assert "Catalogue records" in calls[0]["contents"]
    assert "productUrl" not in calls[0]["contents"]
    assert answer.source == "llm" and answer.model == "gemini-3.5-flash-lite"


def test_gemini_client_configures_bounded_sdk_backoff(monkeypatch):
    import sys
    import types

    captured = {}
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_genai_types = types.ModuleType("google.genai.types")
    fake_genai_types.HttpRetryOptions = lambda **kw: kw
    fake_genai_types.HttpOptions = lambda **kw: kw
    fake_genai.Client = lambda **kw: captured.update(kw) or object()
    fake_genai.types = fake_genai_types
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai_types)

    client = LLMService("k", model="gemini-3.5-flash-lite", provider="gemini")._get_client()
    retry = captured["http_options"]["retry_options"]
    assert client is not None
    assert retry["attempts"] == 3
    assert retry["http_status_codes"] == [408, 429, 500, 502, 503, 504]


def test_gemini_failure_is_reported_without_switching_models(retriever):
    res = retriever.search("Show black shirts")
    client = _GeminiClient(_ServiceUnavailable("high demand"))
    answer = LLMService(
        "k", model="gemini-3.5-flash-lite", provider="gemini", client=client,
    ).generate("Show black shirts", res)
    calls = client.models.calls
    assert [call["model"] for call in calls] == ["gemini-3.5-flash-lite"]
    assert answer.source == "fallback" and "high demand" in answer.error


def test_no_api_key_uses_fallback(retriever):
    res = retriever.search("Show men's casual shirts")
    ans = LLMService(None).generate("q", res)
    assert ans.source == "fallback" and "not configured" in ans.error


def test_llm_answer_with_invented_id_is_rejected(retriever):
    res = retriever.search("Show black shirts")
    ans = LLMService("k", client=_Client("Try product ID 987654, it is great.")).generate("Show product ID for black shirts", res)
    assert ans.source == "fallback" and "IDs not in" in ans.error


def test_final_response_lists_correct_source_ids(retriever):
    res = retriever.search("Find blue jeans")
    out = compose(res, LLMService(None).generate("Find blue jeans", res))
    assert out.product_ids == res.ids
    assert f"Source product IDs:** {res.ids[0]}" in out.markdown
