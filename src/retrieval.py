"""Hybrid retrieval: exact ID -> exact name -> structured filters -> dense semantic -> clear no-match.

Structured filters are applied to the real records (never guessed by an LLM or inferred from vector
similarity). Semantic search (ChromaDB) is used for descriptive queries and to *rank within* the filtered set."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .filters import Catalog, build_where, parse_query, translate_query

_EXPLICIT_ID = re.compile(r"(?:\b(?:id|sku|product|item|article)\b\s*(?:id|number|no|#)?\s*[:#]?\s*|#)(\d{3,12})\b", re.I)
_BARE_ID = re.compile(r"\s*#?(\d{3,12})\s*")
_PRICE_LIMIT = re.compile(
    r"\b(under|below|less than|up to|upto|at most|within|budget(?: of)?|over|above|more than|at least)"
    r"\s*(?:₹|rs\.?|inr)?\s*([\d,]+)\b", re.I,
)
_RATING_DISCOUNT_LIMIT = re.compile(
    r"\b(rating|discount)\s*(under|below|less than|up to|upto|at most|over|above|more than|at least)\s*([\d.]+)\s*%?",
    re.I,
)


@dataclass
class RetrievalResult:
    query: str
    method: str  # empty_query|exact_id|exact_name|structured|structured+semantic|keyword+semantic|semantic|no_match|clarification
    products: list = field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    filters: dict = field(default_factory=dict)
    unsupported: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    distances: dict = field(default_factory=dict)
    message: str | None = None

    @property
    def ids(self) -> list[int]:
        return [p["id"] for p in self.products]


class HybridRetriever:
    def __init__(self, catalog: Catalog, store, max_results: int = 50, semantic_top_k: int = 10,
                 semantic_max_distance: float = 0.9):
        self.catalog, self.store = catalog, store
        self.max_results, self.semantic_top_k, self.max_distance = max_results, semantic_top_k, semantic_max_distance

    # ---- helpers -------------------------------------------------------------------------
    def _detect_id(self, q: str):
        m = _EXPLICIT_ID.search(q) or _BARE_ID.fullmatch(q)
        if m:
            return int(m.group(1)), True
        for num in re.findall(r"\b\d{3,12}\b", q):
            n = int(num)
            if n in self.catalog.by_id and n not in self.catalog.vocab.years:
                return n, False
        return None

    def _products(self, ids):
        return [self.catalog.by_id[i] for i in ids if i in self.catalog.by_id]

    # ---- main entry ----------------------------------------------------------------------
    def search(self, query: str) -> RetrievalResult:
        q = translate_query(query).strip()
        special = _RATING_DISCOUNT_LIMIT.search(q)
        if special:
            field, operator, amount = special.groups()
            amount = float(amount)
            ceiling = operator.lower() not in {"over", "above", "more than", "at least"}
            product_query = _RATING_DISCOUNT_LIMIT.sub(" ", q)
            result = self._search_catalog(re.sub(r"\s+", " ", product_query).strip() or q)
            source = self.catalog.filter(result.filters) if result.filters else result.products
            key = "rating" if field.lower() == "rating" else "discountPercentage"
            filtered = []
            for product in source:
                value = product.get(key)
                if value is not None and (value <= amount if ceiling else value >= amount):
                    filtered.append(product)
            if not filtered:
                return RetrievalResult(query=q, method="no_match",
                                       message=f"I couldn't find matching products with {field.lower()} "
                                               f"{'at or below' if ceiling else 'at or above'} {amount:g}.",
                                       filters=result.filters)
            result.query = q
            result.products = filtered[: self.max_results]
            result.total_matches = len(filtered)
            result.truncated = len(filtered) > self.max_results
            result.notes.append(f"Filtered by dataset {field.lower()} {'at or below' if ceiling else 'at or above'} {amount:g}.")
            return result
        limit = _PRICE_LIMIT.search(q)
        if not limit:
            return self._search_catalog(q)
        operator, amount = limit.groups()
        ceiling = operator.lower() not in {"over", "above", "more than", "at least"}
        amount = float(amount.replace(",", ""))
        product_query = _PRICE_LIMIT.sub(" ", q)
        product_query = re.sub(r"\b(?:price|cost|budget)\b", " ", product_query, flags=re.I)
        product_query = re.sub(r"\s+", " ", product_query).strip()
        result = self._search_catalog(product_query or q)
        candidates = self.catalog.filter(result.filters) if result.filters else result.products
        priced = []
        for product in candidates:
            try:
                price = float(product["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if (price <= amount if ceiling else price >= amount):
                priced.append(product)
        if not priced:
            return RetrievalResult(
                query=q, method="no_match",
                message=f"I couldn't find products with a verified price {'at or below' if ceiling else 'at or above'} {amount:g} {result.products[0].get('currency', 'INR') if result.products else 'INR'} in this result.",
                total_matches=0,
                filters=result.filters,
                unsupported=result.unsupported,
            )
        result.query = q
        result.products = priced
        result.total_matches = len(priced)
        result.truncated = False
        result.products.sort(key=lambda p: float(p.get("price") or 0))
        result.notes.append(f"Filtered to verified prices {'at or below' if ceiling else 'at or above'} {amount:g}.")
        return result

    def _search_catalog(self, query: str) -> RetrievalResult:
        q = (query or "").strip()
        if not q:
            return RetrievalResult(query=query or "", method="empty_query",
                                   message="Please type a fashion question, e.g. “Show blue jeans”.")
        brand_comparison = self._brand_comparison(q)
        if brand_comparison:
            return brand_comparison
        pq = parse_query(q, self.catalog.vocab)
        if any(p.get("sizeValues") for p in self.catalog.records):
            pq.unsupported = [f for f in pq.unsupported if f != "size"]
        if any(p.get("discountPercentage") is not None for p in self.catalog.records):
            pq.unsupported = [f for f in pq.unsupported if f != "discount"]
        if re.search(r"\b(?:price|cost)\b", q, re.I) and not any(p.get("price") is not None for p in self.catalog.records):
            pq.unsupported.append("price")
        base = dict(query=q, filters=pq.filters, unsupported=pq.unsupported)

        # 1. exact product ID
        found = self._detect_id(q)
        if found:
            pid, explicit = found
            if pid in self.catalog.by_id:
                return RetrievalResult(method="exact_id", products=[self.catalog.by_id[pid]], total_matches=1,
                                       **{**base, "filters": {}})
            if explicit:
                return RetrievalResult(method="no_match", message=f"No product with ID {pid} exists in this catalogue.", **base)

        # 2. exact product name
        ids = self.catalog.find_exact_name(q)
        if ids:
            res = RetrievalResult(method="exact_name", products=self._products(ids), total_matches=len(ids),
                                  **{**base, "filters": {}})
            if len(ids) > 1:
                res.notes.append(f"{len(ids)} different products share this exact name; all are shown.")
            return res

        # 3/4. structured filters (+ semantic ranking when the query also contains descriptive words)
        if pq.has_filters:
            return self._structured(q, pq, base)

        # 4. dense semantic / keyword discovery
        if pq.residual_tokens:
            return self._semantic(q, pq, base)

        # 5. nothing to search on
        if pq.unsupported:
            return RetrievalResult(method="clarification", message=(
                "That information (" + ", ".join(pq.unsupported) + ") is not in this catalogue. "
                "Tell me which product (name or ID) or type of item you mean and I'll show what the dataset does contain."), **base)
        return RetrievalResult(method="clarification", message=(
            "I couldn't tell what to look for. Try a colour, gender, article type, season or usage "
            "(e.g. “black formal shoes”) or a product ID/name."), **base)

    def _brand_comparison(self, query: str) -> RetrievalResult | None:
        q = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        if not re.search(r"\b(?:vs|versus|compare|comparison|and|between|both|which|better|best)\b", q):
            return None
        brands = sorted({str(p.get("brand")) for p in self.catalog.records if p.get("brand")})
        aliases = {"reebok": {"reebok", "rebook"}}
        mentioned = []
        for brand in brands:
            tokens = aliases.get(brand.lower(), {brand.lower()})
            if any(re.search(rf"\b{re.escape(token)}\b", q) for token in tokens):
                mentioned.append(brand)
        if len(mentioned) < 2:
            return None
        products = []
        for brand in mentioned:
            matches = [p for p in self.catalog.records if str(p.get("brand", "")).lower() == brand.lower()]
            products.extend(matches[:3])
        return RetrievalResult(
            query=query, method="brand_comparison", products=products, total_matches=len(products),
            notes=["Products and retail details are from the local retail catalogue; listed prices may change."],
        )

    def _structured(self, q, pq, base) -> RetrievalResult:
        cands = self.catalog.filter(pq.filters)
        total = len(cands)
        if total == 0:
            return RetrievalResult(method="no_match", message=(
                "No products in the catalogue match all of: " + _describe(pq.filters) + "."), **base)
        notes = []
        if pq.residual_tokens:
            allowed = {r["id"] for r in cands}
            n = min(total, max(self.max_results * 4, 100))
            ranked = [(i, d) for i, d in self.store.query(q, n, build_where(pq.filters)) if i in allowed]
            if ranked:
                kw = set(self.catalog.keyword_hits(pq.residual_tokens))
                ranked.sort(key=lambda x: 0 if x[0] in kw else 1)  # stable: keyword hits first, then by similarity
                shown = ranked[: self.max_results]
                res = RetrievalResult(method="structured+semantic", products=self._products([i for i, _ in shown]),
                                      total_matches=total, truncated=total > len(shown),
                                      distances={i: d for i, d in shown}, **base)
                res.notes.append("All results satisfy the structured filters; the extra words "
                                 f"({', '.join(pq.residual_tokens)}) were used only to rank them.")
                return res
            notes.append("Semantic ranking unavailable; showing filtered matches in ID order.")
        shown = sorted(cands, key=lambda r: r["id"])[: self.max_results]
        res = RetrievalResult(method="structured", products=shown, total_matches=total,
                              truncated=total > len(shown), notes=notes, **base)
        if res.truncated:
            res.notes.append(f"Showing the first {len(shown)} of {total} matches (limit MAX_RESULTS={self.max_results}).")
        return res

    def _semantic(self, q, pq, base) -> RetrievalResult:
        kw = self.catalog.keyword_hits(pq.residual_tokens)[: self.max_results]
        ranked = [(i, d) for i, d in self.store.query(q, self.semantic_top_k, None) if d <= self.max_distance]
        ids = list(kw) + [i for i, _ in ranked if i not in set(kw)]
        ids = ids[: max(self.semantic_top_k, len(kw))]
        if not ids:
            return RetrievalResult(method="no_match", message=(
                "I couldn't find products that are similar enough to that description in this catalogue."), **base)
        res = RetrievalResult(method="keyword+semantic" if kw else "semantic", products=self._products(ids),
                              total_matches=len(ids), distances=dict(ranked), **base)
        res.notes.append("Semantic results are similarity-based suggestions; attributes such as colour or "
                         "gender were NOT verified as hard constraints.")
        if kw:
            res.notes.append("Products whose names contain all of your keywords are listed first.")
        return res


def _describe(filters: dict) -> str:
    return ", ".join(f"{f} = {' or '.join(map(str, v))}" for f, v in filters.items())
