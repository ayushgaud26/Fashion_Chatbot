"""Gemini answer generation grounded in retrieved catalogue records."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from .retrieval import RetrievalResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a fashion product discovery assistant. Answer using only the supplied fashion catalogue records. "
    "The retrieval notes explain whether records are exact matches or only suggestions. Never claim a suggestion "
    "satisfies a requested filter unless the record and retrieval notes verify it. If no records were retrieved, "
    "say that the catalogue has no matching records and ask the user to change the search; do not answer from "
    "general knowledge. "
    "Never invent product names, IDs, colours, categories, sizes, brands, prices, discounts, stock levels, or "
    "availability. Distinguish retrieved facts from recommendations. If the dataset does not contain the "
    "requested information, say so. When presenting products, preserve their original names. Do not include "
    "numeric product IDs in prose unless the user asks for IDs; if asked, copy the exact id field from the "
    "supplied records and never mistake a price, review count, year, or other number for an ID. "
    "For a follow-up question, treat records marked as coming from a previous query as the products being discussed. "
    "When the user asks which product is best, make a useful, direct recommendation from the supplied candidates: "
    "prefer stronger ratings supported by more reviews, and use lower price as a tie-breaker. Name the selected "
    "product and explain the catalogue facts behind the choice. Make clear that ratings and price are only the "
    "available evidence; never claim unlisted comfort, quality, durability, or personal fit. If those fields are "
    "missing, say what evidence is missing and ask one concise preference question. For comparisons, compare "
    "only attributes present in the supplied records. "
    "For product searches, recommend no more than three retrieved products and include only useful supported "
    "facts such as price, material, size, fit, and rating. Do not output product URLs; product cards already "
    "provide links. Only include an ID if the user explicitly asks for one, copied exactly from its field. "
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
    if result.message:
        lines.append(f"Retrieval message: {result.message}")
    if result.notes:
        lines.append("Notes: " + " ".join(result.notes))
    lines.append("Catalogue records (JSON, one per line):")
    if not result.products:
        lines.append("No catalogue records were retrieved.")
    for p in result.products[:max_items]:
        fields = ("id", "productDisplayName", "brand", "gender", "masterCategory", "subCategory", "articleType",
                  "baseColour", "material", "size", "sizeFit", "price", "originalPrice", "discountPercentage",
                  "currency", "rating", "reviewCount", "availability", "comparisonFacts", "customerSignals",
                  "retailer")
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
                 client=None, max_context_items: int = 8, timeout: float = 20.0,
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
            if self.provider == "gemini":
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=int(self.timeout * 1000),
                        retry_options=types.HttpRetryOptions(
                            # Retries are managed below so a transient Gemini 503 does not
                            # silently fall back after the SDK's short built-in retry window.
                            attempts=1,
                            initial_delay=0.5,
                            max_delay=1.0,
                            exp_base=2.0,
                            jitter=0.2,
                            http_status_codes=[408, 429, 500, 502, 503, 504],
                        ),
                    ),
                )
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        return self._client


    def _redact(self, msg: str) -> str:
        return msg.replace(self.api_key, "***") if self.api_key else msg

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for value in (getattr(exc, "code", None), getattr(exc, "status_code", None)):
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            try:
                return int(value)
            except (TypeError, ValueError):
                match = re.search(r"\b([45]\d\d)\b", str(value or ""))
                if match:
                    return int(match.group(1))
        return None

    @classmethod
    def _is_transient_gemini_error(cls, exc: Exception) -> bool:
        if cls._status_code(exc) in {408, 429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__.casefold()
        return any(token in name for token in ("timeout", "connection", "temporarilyunavailable"))

    def generate(self, question: str, result: RetrievalResult, history: list[dict] | None = None) -> LLMAnswer:
        if not self.available:
            return LLMAnswer(None, "fallback", error=f"{self.provider.upper()} API key is not configured")
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
            "Keep exact product names, brand names, prices, sizes, and other catalogue facts as they appear. "
            "Never include URLs; include IDs only when explicitly requested."
        )
        system_instruction = f"{SYSTEM_PROMPT} {language_instruction}"
        user_content = (f"Recent conversation:\n{history_context}\n\n"
                        f"Current question: {question}\n\n"
                        f"{build_context(result, self.max_context_items)}")
        response_model = self.model
        try:
            client = self._get_client()
        except Exception as exc:
            log.exception("Could not initialize LLM client: provider=%s model=%s error=%s",
                          self.provider, self.model, type(exc).__name__)
            return LLMAnswer(None, "fallback", error=(
                f"LLM client initialization failed ({type(exc).__name__}: {self._redact(str(exc))[:160]})"
            ))
        if self.provider == "gemini":
            attempt, delay = 0, 1.0
            while True:
                try:
                    resp = client.models.generate_content(
                        model=response_model,
                        contents=user_content,
                        config={
                            "system_instruction": system_instruction,
                            "temperature": self.temperature,
                            "max_output_tokens": 360,
                            "thinking_config": {"thinking_level": "minimal"},
                        },
                    )
                    text = (resp.text or "").strip()
                    break
                except Exception as exc:
                    status = self._status_code(exc)
                    if not self._is_transient_gemini_error(exc):
                        log.warning("LLM call failed: provider=gemini model=%s status=%s error=%s detail=%s",
                                    self.model, status, type(exc).__name__, self._redact(str(exc))[:240])
                        return LLMAnswer(None, "fallback", error=(
                            f"LLM unavailable ({type(exc).__name__}: {self._redact(str(exc))[:160]})"
                        ))
                    attempt += 1
                    log.warning("Transient Gemini error; waiting and retrying: model=%s status=%s error=%s attempt=%s delay=%.1fs",
                                self.model, status, type(exc).__name__, attempt, delay)
                    time.sleep(delay)
                    delay = min(delay * 2, 15.0)
        else:
            try:
                resp = client.chat.completions.create(
                    model=response_model,
                    temperature=self.temperature,
                    max_tokens=360,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content},
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # network, auth, rate limit, malformed response...
                status = self._status_code(exc)
                log.warning("LLM call failed: provider=%s model=%s status=%s error=%s detail=%s",
                            self.provider, self.model, status, type(exc).__name__, self._redact(str(exc))[:240])
                return LLMAnswer(None, "fallback", error=(
                    f"LLM unavailable ({type(exc).__name__}: {self._redact(str(exc))[:160]})"
                ))
        if not text:
            return LLMAnswer(None, "fallback", error="LLM returned an empty answer")
        bad = unknown_ids_in(text, result, question)
        if bad:
            return LLMAnswer(None, "fallback", error=f"LLM answer mentioned IDs not in the retrieved records ({', '.join(bad[:3])})")
        return LLMAnswer(text, "llm", model=response_model)
