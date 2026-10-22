"""Fashion Product RAG chatbot. Run with: streamlit run app.py"""
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.chat_memory import ChatMemory
from src.config import get_config
from src.engine import load_engine
from src.filters import parse_query, translate_query
from src.retrieval import RetrievalResult

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Friendly Fashion Assistant", page_icon="👗", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] {display: none;}
div[data-testid="stAppViewBlockContainer"] {max-width: 1450px; padding-top: 2rem;}
div[data-testid="stMetric"] {background: rgba(255,255,255,.035); padding: .7rem; border-radius: .6rem;}
div[data-testid="stImage"] img {height: 230px; object-fit: contain; background: #fff; border-radius: .45rem;}
.product-image-placeholder {height: 230px; display: grid; place-items: center; background: #f5f5f5;
    color: #666; border-radius: .45rem; font-size: 2rem;}
.product-name {height: 3.2em; overflow: hidden; line-height: 1.5em; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;}
@media (max-width: 760px) {
    [class*="st-key-product_grid_"] [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
    [class*="st-key-product_grid_"] [data-testid="stColumn"] {flex: 1 1 calc(50% - .5rem) !important; min-width: 0;}
}
@media (max-width: 480px) {
    [class*="st-key-product_grid_"] [data-testid="stColumn"] {flex-basis: 100% !important;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def fetch_product_image(url):
    try:
        request = Request(url, headers={"User-Agent": "FashionRAGDemo/1.0"})
        with urlopen(request, timeout=1.5) as response:
            if not response.headers.get("Content-Type", "").lower().startswith("image/"):
                return None
            # Product thumbnails should stay small; don't let one huge remote image
            # delay the whole result grid.
            data = response.read(3 * 1024 * 1024 + 1)
            return data if len(data) <= 3 * 1024 * 1024 else None
    except (URLError, TimeoutError, ValueError, OSError):
        return None


EXAMPLES = [
    ("Find blue jeans", "Find blue jeans"),
    ("Show men's casual shirts", "Show men's casual shirts"),
    ("Traditional wear", "traditional wear"),
]
CARD_LIMIT = 10
THANKS_PATTERN = r"(?:thanks?|thank\s*you+|thank\s*u+|thx+|thnx+|ty)"
ACKNOWLEDGEMENTS = re.compile(
    rf"^(?:ok(?:ay)?|alright|all right|got it|understood|sounds good|cool|great|sure)"
    rf"(?:[,\s]+{THANKS_PATTERN})?[.!?,\s]*$",
    re.I,
)
GREETINGS = re.compile(
    rf"^(?:h+i+|hell+o+|he+y+|hiya+|good morning|good afternoon|good evening|how are(?: you)?|"
    rf"what can you do|who are you|{THANKS_PATTERN})[.!?,\s]*$", re.I,
)
GOODBYES = re.compile(
    rf"^(?:(?:stop|quit|exit|that's all|that is all|i'?m done|i am done|goodbye|bye|see you)"
    rf"(?:\s+(?:and\s+)?{THANKS_PATTERN})?|"
    rf"{THANKS_PATTERN}\s+(?:for (?:your|the) (?:time|help)|that'?s all))"
    r"[.!?,\s]*$",
    re.I,
)
DECLINES = re.compile(
    r"^(?:no|nope|nah+|na+h+|not really|no thanks?|no thank you+|nahi|nhi|na|"
    r"not\s+(?:this|that|these|those)(?:\s+(?:types?|kinds?|ones?|items?|products?|categories?|styles?|options?))?|"
    r"(?:this|that|these|those)\s+(?:is|are)\s+not\s+(?:what\s+i\s+want|right)|"
    r"not what i want|not interested|i\s+don'?t\s+want\s+(?:this|that|these|those))"
    r"[.!?,\s]*$",
    re.I,
)
NEEDS_LLM_SYNTHESIS = re.compile(
    r"\b(?:compare|comparison|versus|vs|which|better|best|recommend|recommendation|"
    r"difference|why|explain|suggest|price|cost|rating|discount|how|what)\b",
    re.I,
)
CONTEXTUAL_FOLLOW_UP = re.compile(
    r"\b(?:(?:suggest|recommend|show|give|pick)\s+me\s+one|"
    r"which\s+(?:one|option|product)(?:\s+is)?\s+(?:better|best|more|less)|"
    r"which\s+is\s+(?:better|best)|which\s+(?:better|best)|which\s+one\b|compare\s+(?:them|these|both)|"
    r"between\s+them|what\s+about\s+(?:it|that|them|those)|how\s+much|"
    r"(?:what(?:'s|\s+is\s+)?\s+)?(?:the\s+)?(?:price|cost)\s+(?:of\s+)?(?:it|that|them|those)?|"
    r"where\s+can\s+i\s+buy\s+(?:it|that|them|those)?|tell\s+me\s+more\s+about\s+(?:it|that|them)|"
    r"inme\s+se|kaunsa\s+(?:better|behtar|accha)|kaunsi\s+(?:better|behtar|acchi)|"
    r"kaun\s+sa\s+(?:better|behtar)|dono\s+me\s+se|compare\s+karo|"
    r"\b(?:it|that|these|those|them)\s+(?:price|cost)\b|"
    r"\b(?:iska|iski|iske|unka|unki|unke|inka|inki|inhe|dono|inme|isme)\b)\b",
    re.I,
)


def previous_products(messages):
    for message in reversed(messages):
        result = message.get("result")
        if message.get("role") == "assistant" and result:
            products = result.get("products") or result.get("suggestions")
            if products:
                return products
    return []


def previous_user_query(messages):
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return ""


def rank_for_recommendation(products):
    """Put catalogue-supported options first for a 'which is best' follow-up."""
    def score(product):
        try:
            rating = float(product.get("rating"))
        except (TypeError, ValueError):
            rating = -1.0
        try:
            reviews = int(product.get("reviewCount") or 0)
        except (TypeError, ValueError):
            reviews = 0
        try:
            price = float(product.get("price"))
        except (TypeError, ValueError):
            price = float("inf")
        # Smooth tiny review samples toward a neutral 4/5 prior so a 4.7 rating
        # from 15 reviews does not automatically outrank a well-reviewed 4.3.
        supported_rating = ((rating * reviews) + (4.0 * 100)) / (reviews + 100) if rating >= 0 else -1
        return supported_rating, reviews, -price

    return sorted(products, key=score, reverse=True)


def combine_filter_refinement(previous_query, query, vocab):
    """Carry prior search intent into a clear filter-only follow-up such as 'for men'."""
    previous = parse_query(previous_query, vocab)
    current = parse_query(query, vocab)
    if not previous_query or not previous.filters and not previous.residual_tokens:
        return query
    # A follow-up must be made only of recognized filters. Do not combine a new
    # descriptive request, since that is likely a fresh search rather than a refinement.
    if not current.filters or current.residual_tokens or current.unsupported:
        return query
    # Avoid combining contradictory values for the same field (for example, changing
    # a previous men's request to women's); a same-field request should stand alone.
    if set(previous.filters).intersection(current.filters):
        return query
    return f"{previous_query} {query}"


def budget_amount(query):
    text = translate_query(query).lower().replace(",", "")
    match = re.search(r"\bbudget\b[^0-9]{0,24}(\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)[^a-z0-9]{0,12}\bbudget\b", text)
    if not match:
        match = re.search(
            r"\b(?:under|below|less than|up to|upto|within|at most)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)",
            text,
        )
    return float(match.group(1)) if match else None


def filter_previous_request_by_budget(engine, previous_query, query, previous=None):
    """Apply a budget to the previous cards, previous request, or catalogue if new."""
    amount = budget_amount(query)
    if amount is None:
        return None
    # If this message names a product/category, treat it as a fresh constrained
    # search. Otherwise a prior shirt result could incorrectly satisfy "watch under 1000".
    current = parse_query(query, engine.catalog.vocab)
    if current.filters:
        return None
    filters = {}
    if previous:
        candidates = [engine.catalog.by_id.get(p.get("id"), p) for p in previous]
    elif previous_query:
        prior = engine.retriever.search(previous_query)
        filters = prior.filters or {}
        candidates = engine.catalog.filter(filters) if filters else prior.products
    else:
        candidates = engine.catalog.records
    if not candidates:
        return None

    def price_of(product):
        try:
            return float(product.get("price"))
        except (TypeError, ValueError):
            return None

    affordable = [product for product in candidates
                  if price_of(product) is not None and price_of(product) <= amount]
    affordable.sort(key=lambda product: (-(product.get("rating") or 0), price_of(product)))
    if affordable:
        result = RetrievalResult(query=query, method="budget_follow_up", products=affordable[:50],
                                 total_matches=len(affordable), filters=filters,
                                 notes=[f"Applied a maximum budget of {amount:g} to the previous request."])
        return result, [], amount

    closest = [product for product in candidates if price_of(product) is not None]
    closest.sort(key=price_of)
    result = RetrievalResult(query=query, method="budget_no_match", total_matches=0, filters=filters,
                             notes=[f"No products from the previous request cost {amount:g} or less."])
    return result, closest[:3], amount


@st.cache_resource(show_spinner="Loading the fashion catalogue…")
def get_engine():
    cfg = get_config()
    return load_engine(cfg, include_retail=cfg.dataset_path.suffix.lower() != ".csv")


def _card(product, image):
    with st.container(border=True):
        if image:
            st.image(image, width="stretch")
        else:
            st.markdown('<div class="product-image-placeholder">🛍️</div>', unsafe_allow_html=True)

        rating = product.get("rating")
        reviews = int(product.get("reviewCount") or 0)
        if rating and rating > 0:
            st.caption(f"⭐ {rating:g} · {reviews:,} reviews" if reviews else f"⭐ {rating:g}")
        st.markdown(f"**{escape(str(product.get('brand') or 'Fashion'))}**")
        description = (product.get("sourceFields") or {}).get("product_description") or product.get("productDisplayName") or "Fashion item"
        st.markdown(f'<div class="product-name">{escape(str(description))}</div>', unsafe_allow_html=True)
        subcategory = product.get("subCategory") or product.get("articleType")
        gender = product.get("gender")
        if subcategory or gender:
            st.caption(" · ".join(str(value) for value in (gender, subcategory) if value))

        currency = "₹" if str(product.get("currency", "")).upper() == "INR" else (product.get("currency") or "")
        price = product.get("price")
        original = product.get("originalPrice")
        price_text = f"**{currency} {price:,.0f}**" if price is not None else "**Price unavailable**"
        if original is not None and price is not None and original > price:
            price_text += f"  \n~~{currency} {original:,.0f}~~"
        if product.get("discountPercentage") is not None and product.get("discountPercentage") > 0:
            price_text += f" · {product['discountPercentage']:g}% off"
        st.markdown(price_text)
        if product.get("productUrl"):
            st.link_button("View product", product["productUrl"], width="stretch")
        else:
            st.caption("Link unavailable")


def show_cards(products, engine, key, heading=None):
    if not products:
        return
    if heading:
        st.markdown(f"#### {heading}")
    visible = products[:CARD_LIMIT]
    images = {product["id"]: engine.images.get_bytes(product["id"]) for product in visible}
    pending = {
        product["id"]: engine.images.get_url(product["id"])
        for product in visible
        if images[product["id"]] is None and engine.images.get_url(product["id"])
    }
    if pending:
        with ThreadPoolExecutor(max_workers=min(8, len(pending))) as pool:
            tasks = {pool.submit(fetch_product_image, url): product_id for product_id, url in pending.items()}
            for task in as_completed(tasks):
                try:
                    images[tasks[task]] = task.result()
                except Exception:
                    images[tasks[task]] = None
    with st.container(key=f"product_grid_{key}"):
        for start in range(0, len(visible), 3):
            row = st.columns(3, gap="medium")
            for col, product in zip(row, visible[start:start + 3]):
                with col:
                    _card(product, images.get(product["id"]))
    if len(products) > CARD_LIMIT:
        st.caption(f"Showing {CARD_LIMIT} of {len(products)} matches. Add a colour, size, or budget to narrow the list.")


def render_refinement_buttons(products, query, key, vocab):
    """Offer one-click refinements that are actually present in the current matches."""
    if len(products) < 8:
        return
    parsed = parse_query(query, vocab)
    choices = []
    for field, label in (("gender", "Audience"), ("baseColour", "Colour"), ("size", "Size")):
        if field in parsed.filters:
            continue
        values = sorted({
            str(value).strip()
            for product in products
            for value in ([product.get(field)] if field != "size" else (product.get("sizeValues") or []))
            if value and str(value).strip()
        }, key=str.casefold)
        # Keep the prompt useful and compact; skip fields with only one available value.
        if len(values) > 1:
            choices.extend((label, value) for value in values[:2])
    if not choices:
        return
    st.caption("Narrow these results:")
    columns = st.columns(min(4, len(choices)), gap="small")
    for index, (label, value) in enumerate(choices[:4]):
        if columns[index % len(columns)].button(f"{label}: {value}", key=f"refine_{key}_{label}_{value}", width="stretch"):
            st.session_state["pending"] = f"{query} {value}"


def find_related_products(engine, result, query, limit=3):
    """Find clearly related alternatives without dropping product type or audience."""
    filters = dict(result.filters or {})
    category_fields = {"articleType", "subCategory", "masterCategory"}
    # Only relax optional attributes when the query also pins down a product type.
    # Preserve product type and audience so alternatives cannot drift into unrelated
    # items. A style/occasion tag may be relaxed last when the requested type exists
    # but the catalogue has no products carrying that tag.
    if filters and category_fields.intersection(filters):
        relaxable = ("baseColour", "size", "material", "season", "brand")
        if filters.get("subCategory") or filters.get("articleType"):
            relaxable += ("usage",)
        relaxed = set()
        for field in relaxable:
            if field not in filters:
                continue
            relaxed.add(field)
            candidates = {key: values for key, values in filters.items() if key not in relaxed}
            matches = engine.catalog.filter(candidates)
            if matches:
                return sorted(matches, key=lambda p: (
                    -(p.get("rating") or 0), float(p.get("price") or 10**12)
                ))[:limit]
        return []
    if filters:
        return []

    parsed = parse_query(query, engine.catalog.vocab)
    keyword_hits = engine.catalog.keyword_hits(parsed.residual_tokens)
    if keyword_hits:
        return [engine.catalog.by_id[pid] for pid in keyword_hits[:limit]]
    try:
        similar = engine.store.query(query, 12, None)
        return [engine.catalog.by_id[pid] for pid, distance in similar
                if pid in engine.catalog.by_id and distance <= engine.retriever.max_distance][:limit]
    except Exception:
        logging.exception("Could not find related catalogue items")
        return []


def requested_missing_brand(engine, query):
    """Recognize explicit 'from/by/brand X' requests when X is absent from this catalogue."""
    match = re.search(
        r"\b(?:from|by|brand)\s+([a-z0-9&'-]+(?:\s+[a-z0-9&'-]+){0,2})",
        query, re.I,
    )
    if not match:
        return None
    words = match.group(1).split()
    known = {re.sub(r"[^a-z0-9]+", " ", str(p.get("brand", "")).lower()).strip()
             for p in engine.catalog.records if p.get("brand")}
    # A product type can follow a known brand ("by Puma running shoes"), so test
    # progressively shorter leading phrases before deciding that the brand is missing.
    for size in range(len(words), 0, -1):
        candidate = " ".join(words[:size]).lower()
        if re.sub(r"[^a-z0-9]+", " ", candidate).strip() in known:
            return None
    return words[0]


def requested_missing_item_type(engine, query):
    """Stop unresolved request terms from being treated as semantic matches."""
    # A contextual comparison refers to products already in the conversation;
    # words such as "one" or "option" must not be mistaken for a product type.
    if CONTEXTUAL_FOLLOW_UP.search(query):
        return None
    parsed = parse_query(query, engine.catalog.vocab)
    request_words = {"suggest", "suggestion", "suggestions", "recommend", "recommendation",
                     "show", "find", "search", "get", "give", "want", "need", "looking",
                     # Intent/comparison language is not a requested product type.
                     "compare", "comparison", "versus", "vs", "between", "both", "which",
                     "better", "best", "difference", "differences", "cheaper", "cheapest",
                     # Budget operators are handled by HybridRetriever, not product categories.
                     "under", "below", "less", "than", "up", "to", "upto", "at", "most",
                     "within", "over", "above", "more", "least", "andar"}
    unresolved = [token for token in parsed.residual_tokens if token not in request_words]
    if not unresolved:
        return None

    labels = {
        "perfume": "Perfumes", "perfumes": "Perfumes",
        "belt": "Belts", "belts": "Belts",
        "handbag": "Handbags", "handbags": "Handbags",
    }
    return " ".join(labels.get(token, token.title()) for token in unresolved)


def is_general_brand_question(engine, query):
    """Handle requests for a universal 'best brand' without inventing a ranking."""
    normalized = re.sub(r"\s+", " ", translate_query(query).lower()).strip()
    if not re.search(r"\bbrand(?:s|ed)?\b", normalized) or not re.search(
        r"\b(?:best|better|good|top|popular|leading|recommend|recommendation|suggest|which|what|list|have|show|any|available)\b",
        normalized,
    ):
        return False
    # Keep named-brand questions and comparisons on the existing catalogue/LLM path.
    for brand in {str(p.get("brand") or "").strip().lower() for p in engine.catalog.records} - {""}:
        if re.search(rf"\b{re.escape(brand)}\b", normalized):
            return False
    return True


def is_conversational(query):
    """Routing only: a greeting/thanks/goodbye/refusal needs no catalogue lookup.

    The reply itself is still written by Gemini; nothing here returns canned text.
    """
    normalized = re.sub(r"\s+", " ", translate_query(query).strip())
    return any(pattern.fullmatch(normalized)
               for pattern in (GREETINGS, GOODBYES, ACKNOWLEDGEMENTS, DECLINES))


BRAND_WORDS = re.compile(
    r"\b(?:top|best|popular|leading|good|famous|most|which|what|whats|are|is|the|brands?|branded|in|for|of|from|"
    r"do|does|u|you|have|has|show|me|list|any|available|give|tell|about|please|can|some|only|\d+)\b"
)


def brand_ranking_context(engine, query, previous_query, previous):
    """'top 3 brands in jeans': rank brands inside the requested category from catalogue facts."""
    normalized = translate_query(query).lower()
    if not re.search(r"\bbrands?\b", normalized):
        return None
    if not re.search(r"\b(?:top|best|popular|leading|most|good|famous)\b", normalized):
        return None
    scope = re.sub(r"\s+", " ", BRAND_WORDS.sub(" ", normalized)).strip(" ?!.,")
    if not scope and previous and previous_query and not re.search(r"\bbrands?\b", previous_query, re.I):
        scope = previous_query          # "top 3 brands" right after a product search
    if not scope:
        return None                     # no category given: handled by the general brand path
    wanted = re.search(r"\btop\s+(\d+)\b", normalized)
    top_n = min(int(wanted.group(1)), 10) if wanted else 5

    found = engine.retriever.search(scope)
    candidates = engine.catalog.filter(found.filters) if found.filters else found.products
    if not candidates:
        return RetrievalResult(
            query=query, method="brand_ranking_no_match",
            notes=[f"No catalogue products matched '{scope}', so no brands can be ranked for it."],
        )

    stats = {}
    for product in candidates:
        brand = str(product.get("brand") or "").strip()
        if not brand:
            continue
        entry = stats.setdefault(brand, {"n": 0, "rated": [], "reviews": 0, "prices": [], "best": None})
        entry["n"] += 1
        try:
            rating = float(product.get("rating"))
            reviews = int(product.get("reviewCount") or 0)
            if rating > 0:
                entry["rated"].append((rating, reviews))
        except (TypeError, ValueError):
            pass
        entry["reviews"] += int(product.get("reviewCount") or 0)
        try:
            entry["prices"].append(float(product.get("price")))
        except (TypeError, ValueError):
            pass
        best = entry["best"]
        if best is None or float(product.get("rating") or 0) > float(best.get("rating") or 0):
            entry["best"] = product

    def avg_rating(entry):
        # Same smoothing idea as rank_for_recommendation: few reviews shouldn't win on a lucky 4.8.
        total = sum(r * max(c, 0) for r, c in entry["rated"])
        count = sum(max(c, 0) for r, c in entry["rated"])
        return (total + 4.0 * 100) / (count + 100) if entry["rated"] else 0.0

    ranked = sorted(stats.items(), key=lambda kv: (avg_rating(kv[1]), kv[1]["n"]), reverse=True)[:top_n]
    lines = []
    for brand, entry in ranked:
        price_text = (f"price range {min(entry['prices']):,.0f}-{max(entry['prices']):,.0f} INR"
                      if entry["prices"] else "price unavailable")
        rating_text = (f"review-weighted rating {avg_rating(entry):.2f}" if entry["rated"] else "no ratings")
        lines.append(f"{brand}: {entry['n']} product(s), {rating_text}, {entry['reviews']:,} total reviews, {price_text}")
    return RetrievalResult(
        query=query, method="brand_ranking",
        products=[entry["best"] for _, entry in ranked if entry["best"]],
        total_matches=len(candidates), filters=found.filters,
        notes=[f"The user wants the top {top_n} brands for '{scope}'. {len(candidates)} matching catalogue "
               f"products were analysed. Brand ranking (best first, by review-weighted rating, then product "
               f"count): " + " | ".join(lines),
               "Present the brands in this order with these catalogue facts. State that the ranking comes only "
               "from this catalogue sample, not from the whole market. The products supplied are each brand's "
               "best-rated item."],
    )


def _retrieve_context(engine, query, earlier_messages, refers_previous=False):
    """Retrieval step only. Returns (RetrievalResult, alternatives).

    Nothing user-facing is written here. Every situation (no match, missing brand,
    clarification needed, small talk...) is described in ``result.notes`` so Gemini
    can phrase the reply from facts instead of the app using canned strings.
    """
    previous = previous_products(earlier_messages)
    previous_query = previous_user_query(earlier_messages)

    if is_conversational(query):
        return RetrievalResult(
            query=query, method="conversation",
            notes=["Conversational message (greeting, thanks, goodbye or refusal). "
                   "No catalogue retrieval was needed. Reply briefly and, if suitable, "
                   "invite the user to search for a fashion product."],
        ), []

    ranking = brand_ranking_context(engine, query, previous_query, previous)
    if ranking is not None:
        return ranking, []

    if is_general_brand_question(engine, query):
        best_example_by_brand = {}
        for product in engine.catalog.records:
            brand = str(product.get("brand") or "").strip()
            if not brand:
                continue
            score = (float(product.get("rating") or 0), int(product.get("reviewCount") or 0))
            current = best_example_by_brand.get(brand)
            if current is None or score > current[0]:
                best_example_by_brand[brand] = (score, product)
        examples = [product for _, product in sorted(
            best_example_by_brand.values(), key=lambda entry: entry[0], reverse=True
        )[:15]]
        brand_counts = {}
        for product in engine.catalog.records:
            brand = str(product.get("brand") or "").strip()
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
        brand_list = ", ".join(f"{b} ({n})" for b, n in sorted(brand_counts.items(), key=lambda kv: (-kv[1], kv[0])))
        return RetrievalResult(
            query=query, method="brand_guidance", products=examples, total_matches=len(examples),
            notes=["Examples selected from catalogue records with available ratings and review counts; "
                   "this is not an overall brand ranking.",
                   f"All brands in the catalogue with record counts: {brand_list}"],
        ), []

    budget_context = filter_previous_request_by_budget(engine, previous_query, query, previous)
    if budget_context:
        result, alternatives, amount = budget_context
        if result.products:
            return result, []
        result.notes = list(result.notes) + [
            f"Nothing from the previous request is within {amount:,.0f} INR. "
            "The alternatives are the closest-priced items of the same type and may be over budget."
        ]
        return result, alternatives

    effective_query = query
    if previous:
        effective_query = combine_filter_refinement(previous_query, query, engine.catalog.vocab)
    result = engine.retriever.search(effective_query)
    translated_query = translate_query(query)
    one_product_request = bool(re.search(
        r"\b(?:suggest|recommend|show|give|pick)\s+me\s+one\b", query, re.I
    ))
    contextual_follow_up = (CONTEXTUAL_FOLLOW_UP.search(query)
                            or CONTEXTUAL_FOLLOW_UP.search(translated_query))

    if previous and (contextual_follow_up or refers_previous):
        def _price(product):
            try:
                return float(product.get("price"))
            except (TypeError, ValueError):
                return float("inf")

        follow_up_products = previous
        if re.search(r"\b(?:cheap|cheaper|cheapest|lowest|least expensive|affordable|lower price)\b", query, re.I):
            follow_up_products = sorted(previous, key=_price)
        elif re.search(r"\b(?:expensive|costliest|priciest|highest|costlier|higher price)\b", query, re.I):
            follow_up_products = sorted(previous, key=lambda p: -_price(p) if _price(p) != float("inf") else 0)
        elif re.search(r"\b(?:best|better|recommend|recommendation)\b", query, re.I):
            follow_up_products = rank_for_recommendation(previous)
        if one_product_request:
            follow_up_products = rank_for_recommendation(previous)[:1]
        return RetrievalResult(
            query=query, method="follow_up", products=follow_up_products,
            total_matches=len(follow_up_products),
            notes=["The user is referring to the products from the previous turn; "
                   "compare or answer using only these products."],
        ), []

    if one_product_request:
        return RetrievalResult(
            query=query, method="clarification",
            notes=["The user asked for one suggestion but no product type is known. "
                   "Ask which product type they want. Do not recommend anything."],
        ), []

    if result.products:
        return result, []

    # Hard filters (type, colour, budget, rating...) matched nothing. Never ignore them:
    # relax only the optional ones and keep the product type, then let Gemini explain.
    if result.filters:
        alternatives = find_related_products(engine, result, query)
        result.notes = list(result.notes) + ["No catalogue product matched all of the user's filters."]
        return result, alternatives

    # No recognisable filter: fall back to pure semantic search and let Gemini decide
    # from the retrieved records whether they really answer the question.
    try:
        nearest = engine.store.query(query, 10, None)
        semantic = [engine.catalog.by_id[pid] for pid, _ in nearest if pid in engine.catalog.by_id]
    except Exception:
        logging.exception("Semantic fallback failed")
        semantic = []
    if semantic:
        return RetrievalResult(
            query=query, method="semantic_fallback", products=semantic, total_matches=len(semantic),
            notes=([result.message] if result.message else []) + [
                   "No exact filter match, so these are the nearest catalogue records by meaning. "
                   "They may not match the request. If they do not, say the catalogue has no such item "
                   "and do not present them as matches. Mention only brands and product types that "
                   "appear in these records."],
        ), []
    result.notes = list(result.notes) + ["Nothing in the catalogue matched this request."]
    return result, []


REWRITE_PROMPT = (
    "You are the query-understanding step of a fashion shopping assistant. Fluently understand any "
    "English, Hindi, Hinglish (Hindi or other Indian languages typed in Roman script), mixed-language text, "
    "slang, typos and SMS-style spelling, and rewrite the shopper's message as a short, clear English "
    "search for a fashion catalogue. Translate meaning, not word by word: colours, garments, occasions, "
    "fabrics, sizes, budgets, audience (men/women/kids) and any filters. Drop filler and generic nouns "
    "(kapda, kapde, clothes, items, cheez, wala, mujhe, dikhao, chahiye) once their meaning is captured. "
    "Keep brand names, numbers, prices, sizes and ratings exactly. Garment words with no good English "
    "equivalent (kurta, saree, lehenga, jutti, dupatta) stay as they are. If the message refers to earlier "
    "results (it, this, them, iska, inme se, dono me se, which one, the cheapest one), begin the output with "
    "the tag @prev and a space, then a short English reference like 'which of them is cheaper'; never "
    "invent product names. If it is a greeting, "
    "thanks, goodbye or refusal, or already a clear English search, return it unchanged. "
    "Examples: 'kala kapda' -> 'black'; 'mujhe shaadi ke liye sasta lehenga dikhao' -> 'cheap wedding lehenga'; "
    "'1500 ke andar mens jeans' -> 'men jeans under 1500'; 'ladkiyon ke liye neeli top' -> 'blue top for women'; "
    "'inme se kaunsa sabse accha hai' -> '@prev which of them is the best'; 'puma ke shoes 4 star se upar' -> "
    "'Puma shoes rating above 4'. Output only the query, with no quotes or explanation."
)


def rewrite_query(engine, query, history):
    """RAG query rewriting: let Gemini understand Hinglish/slang before retrieval."""
    llm = engine.llm
    if not llm.available or getattr(llm, "provider", "") != "gemini":
        return query, False
    recent = []
    for message in (history or [])[-4:]:
        content = str(message.get("content") or "").strip()
        if message.get("role") in {"user", "assistant"} and content:
            recent.append(f"{message['role'].title()}: {content[:200]}")
    prompt = ("Recent conversation:\n" + ("\n".join(recent) or "None") +
              f"\n\nShopper's message: {query}\n\nEnglish search query:")
    for attempt in range(2):
        try:
            response = llm._get_client().models.generate_content(
                model=llm.model, contents=prompt,
                config={"system_instruction": REWRITE_PROMPT, "temperature": 0,
                        "max_output_tokens": 80, "thinking_config": {"thinking_level": "minimal"}},
            )
            text = (response.text or "").strip().splitlines()
            rewritten = text[0].strip().strip("\"'`") if text else ""
            refers_previous = rewritten.lower().startswith("@prev")
            if refers_previous:
                rewritten = rewritten[5:].strip()
            if 0 < len(rewritten) <= 200:
                return rewritten, refers_previous
        except Exception:
            logging.exception("Query rewrite attempt %s failed", attempt + 1)
    return query, False


def retrieve_context(engine, query, earlier_messages):
    """Rewrite (Hinglish -> English), then retrieve. Gemini still answers the original message."""
    search_query, refers_previous = rewrite_query(engine, query, earlier_messages)
    result, alternatives = _retrieve_context(engine, search_query, earlier_messages, refers_previous)
    if search_query.strip().casefold() != query.strip().casefold():
        result.notes = list(result.notes) + [
            f"The shopper wrote: '{query}'. It was interpreted as the English catalogue search: '{search_query}'."
        ]
    return result, alternatives, search_query


def generate_answer(engine, query, result, alternatives, history):
    """Every reply is written by Gemini from the retrieved context."""
    grounded = result
    if not result.products and alternatives:
        grounded = RetrievalResult(
            query=query, method=f"{result.method}_alternatives", products=alternatives,
            total_matches=len(alternatives), filters=result.filters,
            notes=list(result.notes) + ([result.message] if result.message else []) + [
                "These are the closest alternatives from the catalogue, not exact matches. "
                "Say clearly that they are not exact matches."
            ],
        )
    try:
        return engine.llm.generate(query, grounded, history=history)
    except Exception as exc:  # never let a Gemini failure crash the chat
        logging.exception("Gemini generation failed")
        return SimpleNamespace(source="error", text="", error=str(exc))


def response_text(answer):
    """Show Gemini's answer. If Gemini is unavailable, show a status message, not a fake answer."""
    if answer is not None and answer.source == "llm" and answer.text:
        return answer.text
    detail = (getattr(answer, "error", None) or "no response") if answer is not None else "no response"
    lowered = detail.lower()
    if "not configured" in lowered:
        return "⚠️ Gemini API key is missing. Add GEMINI_API_KEY to `.env` (or Streamlit Secrets) so I can answer."
    if any(marker in lowered for marker in ("permissiondenied", "unauthorized", "401", "403")):
        return "⚠️ Gemini rejected the configured API key. Replace GEMINI_API_KEY, then restart the app."
    return f"⚠️ Gemini could not generate an answer ({detail}). Please try again in a moment."



def refresh_saved_results(messages, catalog):
    """Rehydrate persisted product cards from the active catalogue, dropping stale rows."""
    for message in messages:
        payload = message.get("result") or {}
        stale = False
        for key in ("products", "suggestions"):
            saved = payload.get(key) or []
            refreshed = []
            for product in saved:
                current = catalog.by_id.get(product.get("id"))
                if current is None:
                    stale = True
                else:
                    refreshed.append(current)
            if len(refreshed) != len(saved):
                payload[key] = refreshed
        if stale and message.get("role") == "assistant":
            message["content"] = (
                "These saved product results came from an older catalogue. "
                "Please search again to see current product photos, prices, and reviews."
            )


try:
    engine = get_engine()
except Exception as exc:
    st.error(f"Could not start the app: {exc}")
    st.info("Check DATASET_PATH in .env and that dependencies are installed (see README → Troubleshooting).")
    st.stop()

memory = ChatMemory()
if "conversation_id" not in st.session_state:
    conversation_id = st.query_params.get("conversation_id")
    if not conversation_id or not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
        conversation_id = uuid.uuid4().hex
        st.query_params["conversation_id"] = conversation_id
    st.session_state["conversation_id"] = conversation_id
    st.session_state["messages"] = memory.load_messages(conversation_id)

# Saved cards may be from an older Arrow catalogue or an earlier deployment.
# Rehydrate them from the active, price/link-verified catalogue on every rerun.
refresh_saved_results(st.session_state["messages"], engine.catalog)

top_left, top_right = st.columns([7, 1])
with top_right:
    if st.button("Reset chat", width="stretch"):
        memory.clear(st.session_state["conversation_id"])
        st.session_state["messages"] = []
        st.rerun()

st.title("👋 Hi, I’m your friendly fashion assistant")
st.caption("Tell me what you’re looking for. I can find products and share the details available in the Myntra catalogue.")
if engine.llm.available:
    st.caption(f"RAG pipeline ready · retrieval from catalogue, answers by {engine.config.llm_provider.title()} / {engine.config.llm_model}")
else:
    st.warning("Every answer is generated by Gemini from retrieved catalogue records, but no API key is configured. "
               "Set GEMINI_API_KEY in `.env`, then restart Streamlit.")

example_cols = st.columns(len(EXAMPLES))
for col, (label, query) in zip(example_cols, EXAMPLES):
    if col.button(label, key=f"example_{label}", width="stretch"):
        st.session_state["pending"] = query

visible_messages = st.session_state["messages"][-20:]
for index, message in enumerate(visible_messages):
    with st.chat_message(message["role"]):
        content = message["content"]
        result_payload = message.get("result") or {}
        # Older saved turns contained verbose internal fallback/debug text; display them cleanly.
        if message["role"] == "assistant" and result_payload.get("products") and (
            "Catalogue-search result" in content or "Source product IDs" in content
        ):
            content = "Here are some products from the catalogue that match your request."
        st.markdown(content)
        # Keep old turns in the transcript, but only load images for the latest result.
        if result_payload.get("products") and index == len(visible_messages) - 1:
            show_cards(result_payload["products"], engine, f"history_{index}")
            render_refinement_buttons(
                result_payload["products"], previous_user_query(visible_messages[:index]),
                f"history_{index}", engine.catalog.vocab,
            )
        if result_payload.get("suggestions") and index == len(visible_messages) - 1:
            show_cards(result_payload["suggestions"], engine, f"history_suggestions_{index}", "Similar items you could try")

typed = st.chat_input("Ask about products, prices, ratings, or styles…")
query = typed or st.session_state.pop("pending", None)
if query:
    user_message = {"role": "user", "content": query}
    st.session_state["messages"].append(user_message)
    memory.save_message(st.session_state["conversation_id"], user_message)
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        earlier_messages = st.session_state["messages"][:-1]
        with st.spinner("Searching the catalogue and writing an answer…"):
            result, alternatives, search_query = retrieve_context(engine, query, earlier_messages)
            answer = generate_answer(engine, query, result, alternatives, earlier_messages)
        user_facing = response_text(answer)
        displayed_products = result.products
        suggestions = [] if displayed_products else alternatives

        if search_query.strip().casefold() != query.strip().casefold():
            st.caption(f"🔎 Understood as: {search_query}")
        st.markdown(user_facing)
        if displayed_products:
            show_cards(displayed_products, engine, f"new_{len(st.session_state['messages'])}")
            render_refinement_buttons(
                displayed_products, query, f"new_{len(st.session_state['messages'])}", engine.catalog.vocab,
            )
        if suggestions:
            show_cards(suggestions, engine, f"suggestions_{len(st.session_state['messages'])}",
                       "Similar items you could try")

    payload = {"method": result.method, "filters": result.filters, "total_matches": result.total_matches,
               "products": displayed_products, "suggestions": suggestions,
               "unsupported": result.unsupported, "notes": result.notes, "distances": result.distances}
    assistant_message = {"role": "assistant", "content": user_facing, "result": payload}
    st.session_state["messages"].append(assistant_message)
    memory.save_message(st.session_state["conversation_id"], assistant_message)
