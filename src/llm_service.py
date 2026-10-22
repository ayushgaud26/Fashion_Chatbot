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
    # Product-search prose does not need IDs; validate them strictly only when requested.
    if not re.search(r"\b(?:(?:product|item)\s+)?(?:id|sku)\b|#\s*\d", question, re.I):
        return []
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
                 client=None, max_context_items: int = 12, timeout: float = 12.0,
                 provider: str = "openai", fallback_model: str | None = None):
        self.api_key, self.model, self.temperature = api_key, model, temperature
        self.provider = provider
        self.fallback_model = fallback_model
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
                # Let the service-level model fallback handle transient errors
                # immediately instead of spending the timeout retrying one model.
                options["max_retries"] = 0
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
        # Keep enough turns to resolve follow-ups without sending a long transcript
        # on every request (which adds latency and can distract the model).
        for message in (history or [])[-8:]:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                history_lines.append(f"{role.title()}: {content[:500]}")
        history_context = "\n".join(history_lines) or "No earlier conversation."
        language_instruction = (
            "Always reply in clear, natural English, regardless of the language used in the question. "
            "Keep exact product names, brand names, prices, sizes, and URLs as they appear in the catalogue."
        )
        request_options = {}
        if self.provider == "gemini":
            request_options["extra_body"] = {
                "extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}
            }
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT} {language_instruction}"},
            {"role": "user", "content": (f"Recent conversation:\n{history_context}\n\n"
                                             f"Current question: {question}\n\n"
                                             f"{build_context(result, self.max_context_items)}")},
        ]
        client = self._get_client()
        response_model = self.model
        try:
            resp = client.chat.completions.create(
                model=response_model, temperature=self.temperature, max_tokens=450,
                messages=messages, **request_options,
            )
        except Exception as exc:  # network, auth, rate limit, malformed response...
            status = getattr(exc, "status_code", None)
            transient = status == 429 or (isinstance(status, int) and 500 <= status <= 599)
            fallback_models = []
            if self.fallback_model and self.fallback_model != self.model:
                fallback_models.append(self.fallback_model)
            # Keep a second model in the chain in case the configured alternate
            # shares the primary model's temporary capacity issue.
            if self.provider == "gemini" and self.model != "gemini-3.6-flash":
                if "gemini-3.6-flash" not in fallback_models:
                    fallback_models.append("gemini-3.6-flash")
            if not (self.provider == "gemini" and fallback_models and transient):
                log.warning("LLM call failed: %s", type(exc).__name__)
                return LLMAnswer(None, "fallback", error=f"LLM unavailable ({type(exc).__name__}: {self._redact(str(exc))[:160]})")
            log.warning("Primary Gemini model returned %s; trying fallback models", status)
            last_error = exc
            resp = None
            for fallback_model in fallback_models:
                response_model = fallback_model
                try:
                    resp = client.chat.completions.create(
                        model=response_model, temperature=self.temperature, max_tokens=450,
                        messages=messages, **request_options,
                    )
                    break
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    fallback_status = getattr(fallback_exc, "status_code", None)
                    log.warning("Gemini fallback model %s failed with %s", fallback_model, fallback_status or type(fallback_exc).__name__)
            if resp is None:
                return LLMAnswer(None, "fallback", error=(
                    f"Primary and fallback Gemini models unavailable ({type(last_error).__name__}: "
                    f"{self._redact(str(last_error))[:160]})"
                ))
        try:
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # network, auth, rate limit, malformed response...
            log.warning("LLM call failed: %s", type(exc).__name__)
            return LLMAnswer(None, "fallback", error=f"LLM unavailable ({type(exc).__name__}: {self._redact(str(exc))[:160]})")
        if not text:
            return LLMAnswer(None, "fallback", error="LLM returned an empty answer")
        bad = unknown_ids_in(text, result, question)
        if bad:
            return LLMAnswer(None, "fallback", error=f"LLM answer mentioned IDs not in the retrieved records ({', '.join(bad[:3])})")
        return LLMAnswer(text, "llm", model=response_model)
