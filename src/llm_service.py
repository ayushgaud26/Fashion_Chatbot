"""Grounded answer generation with an OpenAI chat model, plus a guarded fallback."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .retrieval import RetrievalResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a fashion product discovery assistant. Answer using only the supplied fashion catalogue records. "
    "Never invent product names, IDs, colours, categories, sizes, brands, prices, discounts, stock levels, or "
    "availability. Distinguish retrieved facts from recommendations. If the dataset does not contain the "
    "requested information, say so. When presenting products, preserve their original names. Do not include "
    "numeric product IDs in prose unless the user asks for IDs; if asked, copy the exact id field from the "
    "supplied records and never mistake a price, review count, year, or other number for an ID. "
    "For a follow-up question, treat records marked as coming from a previous query as the products being discussed. "
    "When comparing products, compare only attributes present in the supplied records. Do not declare an "
    "objective winner when the catalogue has no quality, comfort, durability, or review data; explain what "
    "can be compared and ask what matters most to the user. "
    "Do not claim that you have identified a user's personal style or visually recognized an image unless the "
    "relevant functionality has actually been implemented."
)


@dataclass
class LLMAnswer:
    text: str | None
    source: str            # "llm" | "fallback"
    model: str | None = None
    error: str | None = None   # why the fallback was used


def build_context(result: RetrievalResult, max_items: int = 15) -> str:
    lines = [f"Retrieval method: {result.method}", f"Total matching records: {result.total_matches}",
             f"Records supplied below: {min(len(result.products), max_items)}"]
    if result.filters:
        lines.append(f"Filters applied: {json.dumps(result.filters)}")
    if result.unsupported:
        lines.append("The user asked about fields NOT in the dataset: " + ", ".join(result.unsupported))
    if result.notes:
        lines.append("Notes: " + " ".join(result.notes))
    lines.append("Catalogue records (JSON, one per line):")
    for p in result.products[:max_items]:
        fields = ("id", "productDisplayName", "brand", "gender", "masterCategory", "subCategory", "articleType",
                  "baseColour", "material", "size", "sizeFit", "price", "originalPrice", "discountPercentage",
                  "currency", "rating", "reviewCount", "availability", "comparisonFacts", "customerSignals",
                  "productUrl", "retailer")
        lines.append(json.dumps({k: p[k] for k in fields if p.get(k) is not None}, ensure_ascii=False))
    return "\n".join(lines)


def unknown_ids_in(text: str, result: RetrievalResult, question: str) -> list[str]:
    allowed = {str(i) for i in result.ids} | {str(p["year"]) for p in result.products if p.get("year")}
    # Catalogue facts such as prices and review counts can look like product IDs.
    # Validate only numbers explicitly labelled as IDs/SKUs, not every 4-6 digit number.
    for product in result.products:
        for field in ("price", "originalPrice", "reviewCount"):
            try:
                if product.get(field) is not None:
                    allowed.add(str(int(round(float(product[field])))))
            except (TypeError, ValueError, OverflowError):
                pass
    allowed |= set(re.findall(r"\b\d{3,7}\b", question))
    candidates = re.findall(
        r"(?:\b(?:product\s+|item\s+)?(?:id|sku)\s*(?:is\s*)?[:#]?\s*|#)(\d{4,10})\b",
        text, re.I,
    )
    return [n for n in candidates if n not in allowed]


class LLMService:
    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini", temperature: float = 0.1,
                 client=None, max_context_items: int = 15, timeout: float = 30.0,
                 provider: str = "openai"):
        self.api_key, self.model, self.temperature = api_key, model, temperature
        self.provider = provider
        self.max_context_items, self.timeout = max_context_items, timeout
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            options = {"api_key": self.api_key, "timeout": self.timeout}
            if self.provider == "gemini":
                options["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
            self._client = OpenAI(**options)
        return self._client


    def _redact(self, msg: str) -> str:
        return msg.replace(self.api_key, "***") if self.api_key else msg

    def generate(self, question: str, result: RetrievalResult, history: list[dict] | None = None) -> LLMAnswer:
        if not self.available:
            return LLMAnswer(None, "fallback", error=f"{self.provider.upper()} API key is not configured")
        if not result.products:
            return LLMAnswer(None, "fallback", error="no products to describe")
        history_lines = []
        for message in (history or [])[-20:]:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                history_lines.append(f"{role.title()}: {content[:800]}")
        history_context = "\n".join(history_lines) or "No earlier conversation."
        language_instruction = (
            "Always reply in clear, natural English, regardless of the language used in the question. "
            "Keep exact product names, brand names, prices, sizes, and URLs as they appear in the catalogue."
        )
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=600,
                messages=[{"role": "system", "content": f"{SYSTEM_PROMPT} {language_instruction}"},
                          {"role": "user", "content": (f"Recent conversation:\n{history_context}\n\n"
                                                           f"Current question: {question}\n\n"
                                                           f"{build_context(result, self.max_context_items)}")}],
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # network, auth, rate limit, malformed response...
            log.warning("LLM call failed: %s", type(exc).__name__)
            return LLMAnswer(None, "fallback", error=f"LLM unavailable ({type(exc).__name__}: {self._redact(str(exc))[:160]})")
        if not text:
            return LLMAnswer(None, "fallback", error="LLM returned an empty answer")
        bad = unknown_ids_in(text, result, question)
        if bad:
            return LLMAnswer(None, "fallback", error=f"LLM answer mentioned IDs not in the retrieved records ({', '.join(bad[:3])})")
        return LLMAnswer(text, "llm", model=self.model)
