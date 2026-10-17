"""Turn retrieval results (+ optional LLM answer) into clear Markdown. Only fields that exist are shown."""
from __future__ import annotations

from dataclasses import dataclass

from .llm_service import LLMAnswer
from .retrieval import RetrievalResult

NO_FIELDS_NOTE = ("ℹ️ This catalogue has no verified {fields} for the matching product(s), so I can't answer "
                  "that part from the available records.")
METHOD_LABEL = {
    "exact_id": "exact product-ID match", "exact_name": "exact product-name match",
    "structured": "structured attribute filters", "structured+semantic": "structured filters + semantic ranking",
    "keyword+semantic": "name keywords + semantic search", "semantic": "semantic (embedding) search",
    "brand_comparison": "brand comparison", "follow_up": "follow-up using the previous products",
}


def product_markdown(p: dict) -> str:
    bits = []
    cat = " › ".join(x for x in (p.get("masterCategory"), p.get("subCategory"), p.get("articleType")) if x)
    for label, val in (("Gender", p.get("gender")), ("Category", cat), ("Colour", p.get("baseColour")),
                       ("Brand", p.get("brand")), ("Material", p.get("material")), ("Size", p.get("size")),
                       ("Season", p.get("season")), ("Year", p.get("year")), ("Usage", p.get("usage"))):
        if val:
            bits.append(f"{label}: {val}")
    name = p.get("productDisplayName") or "(no display name in dataset)"
    line = f"**{name}** (ID {p['id']}) — " + " · ".join(bits)
    currency = "₹" if p.get("currency", "").upper() == "INR" else p.get("currency", "")
    if p.get("price") is not None:
        line += f" · Price: {currency} {p['price']:,.0f}"
    if p.get("originalPrice") is not None:
        line += f" · Original: {currency} {p['originalPrice']:,.0f}"
    if p.get("discountPercentage") is not None:
        line += f" · Discount: {p['discountPercentage']:g}%"
    if p.get("rating") is not None:
        line += f" · Rating: {p['rating']:g}/5 ({int(p.get('reviewCount') or 0)} reviews)"
        if p.get("priceCheckedAt"):
            line += f" (checked {p['priceCheckedAt']})"
    if p.get("productUrl"):
        line += f" · [View at {p.get('retailer', 'retailer')}]({p['productUrl']})"
    return line


def describe_filters(filters: dict) -> str:
    return ", ".join(f"{f}={'/'.join(map(str, v))}" for f, v in filters.items())


def format_fallback(result: RetrievalResult, reason: str | None = None, list_limit: int = 10) -> str:
    head = "**📋 Catalogue-search result (fallback — not an AI-written answer)**"
    if reason:
        head += f"  \n_Reason: {reason}._"
    lines = [head, ""]
    label = METHOD_LABEL.get(result.method, result.method)
    summary = f"Matched by **{label}**"
    if result.filters:
        summary += f" ({describe_filters(result.filters)})"
    if result.method.startswith(("structured", "exact")) or result.total_matches:
        summary += f" — {result.total_matches} matching product(s)"
    lines.append(summary + ".")
    lines.append("")
    for p in result.products[:list_limit]:
        lines.append(f"- {product_markdown(p)}")
    if len(result.products) > list_limit:
        lines.append(f"- …and {len(result.products) - list_limit} more (see the product cards / source panel).")
    return "\n".join(lines)


@dataclass
class FinalResponse:
    markdown: str
    source: str            # "llm" | "fallback" | "system"
    product_ids: list


def compose(result: RetrievalResult, answer: LLMAnswer | None = None) -> FinalResponse:
    if not result.products:  # no-match, clarification, empty query: deterministic system message
        msg = result.message or "No matching products were found."
        icon = "🔎" if result.method == "no_match" else "💬"
        parts = [f"**{icon} {msg}**"]
        if result.unsupported:
            parts.append(NO_FIELDS_NOTE.format(fields=", ".join(result.unsupported)))
        return FinalResponse("\n\n".join(parts), "system", [])
    if answer is not None and answer.source == "llm":
        body = "**🤖 AI-generated answer** (grounded in the retrieved catalogue records)\n\n" + answer.text
    else:
        body = format_fallback(result, answer.error if answer else None)
    extras = []
    if result.unsupported:
        extras.append(NO_FIELDS_NOTE.format(fields=", ".join(result.unsupported)))
    extras.extend(f"_{n}_" for n in result.notes)
    ids = result.ids
    shown = ", ".join(str(i) for i in ids[:15]) + (" …" if len(ids) > 15 else "")
    extras.append(f"**Source product IDs:** {shown}")
    return FinalResponse(body + "\n\n" + "\n\n".join(extras), answer.source if answer else "fallback", ids)
