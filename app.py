"""Fashion Product RAG chatbot. Run with: streamlit run app.py"""
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
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


EXAMPLES = ["Find blue jeans", "Show men's casual shirts", "Find women's handbags"]
CARD_LIMIT = 10
THANKS_PATTERN = r"(?:thanks?|thank\s*you+|thank\s*u+|thx+|thnx+|ty)"
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
NEEDS_LLM_SYNTHESIS = re.compile(
    r"\b(?:compare|comparison|versus|vs|which|better|best|recommend|recommendation|"
    r"difference|why|explain|suggest|price|cost|rating|discount|how|what)\b",
    re.I,
)
CONTEXTUAL_FOLLOW_UP = re.compile(
    r"\b(?:which\s+(?:one|option|product)(?:\s+is)?\s+(?:better|best|more|less)|"
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


def budget_amount(query):
    text = translate_query(query).lower().replace(",", "")
    match = re.search(r"\bbudget\b[^0-9]{0,24}(\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)[^a-z0-9]{0,12}\bbudget\b", text)
    return float(match.group(1)) if match else None


def filter_previous_request_by_budget(engine, previous_query, query):
    """Apply a budget follow-up to the immediately preceding product request."""
    amount = budget_amount(query)
    if amount is None or not previous_query:
        return None
    prior = engine.retriever.search(previous_query)
    filters = prior.filters or {}
    candidates = engine.catalog.filter(filters) if filters else prior.products
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
    for start in range(0, len(visible), 5):
        row = st.columns(5, gap="small")
        for col, product in zip(row, visible[start:start + 5]):
            with col:
                _card(product, images.get(product["id"]))
    if len(products) > CARD_LIMIT:
        st.caption(f"Showing {CARD_LIMIT} of {len(products)} matches. Add a colour, size, or budget to narrow the list.")


def find_related_products(engine, result, query, limit=3):
    """Relax optional attributes one at a time while retaining the main product category."""
    filters = dict(result.filters or {})
    relax_order = ("brand", "baseColour", "material", "size", "usage", "season", "gender",
                   "articleType", "subCategory", "masterCategory")
    category_fields = {"articleType", "subCategory", "masterCategory"}
    for field in relax_order:
        if field not in filters:
            continue
        # Keep at least one product-type/category constraint so a failed search does not
        # turn into a random recommendation from the entire catalogue.
        if field in category_fields and len(category_fields.intersection(filters)) == 1:
            continue
        filters.pop(field)
        matches = engine.catalog.filter(filters)
        if matches:
            return sorted(matches, key=lambda p: (
                -(p.get("rating") or 0), float(p.get("price") or 10**12)
            ))[:limit]

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


def is_general_brand_question(engine, query):
    """Handle requests for a universal 'best brand' without inventing a ranking."""
    normalized = re.sub(r"\s+", " ", translate_query(query).lower()).strip()
    if not re.search(r"\bbrands?\b", normalized) or not re.search(
        r"\b(?:best|better|good|recommend|recommendation|suggest|which)\b", normalized
    ):
        return False
    # Keep named-brand questions and comparisons on the existing catalogue/LLM path.
    for brand in {str(p.get("brand") or "").strip().lower() for p in engine.catalog.records} - {""}:
        if re.search(rf"\b{re.escape(brand)}\b", normalized):
            return False
    return True


def greeting_reply(query):
    normalized = re.sub(r"\s+", " ", translate_query(query).strip())
    if GOODBYES.fullmatch(normalized):
        return localized(
            query,
            "Sure, I’ll stop here. Thank you for your time—have a nice day ahead! 👋",
            "ठीक है, मैं यहीं रुकता हूँ। आपका समय देने के लिए धन्यवाद—आपका दिन शुभ हो! 👋",
            "Theek hai, main yahin rukta hoon. Aapke time ke liye dhanyavaad—have a nice day ahead! 👋",
        )
    if not GREETINGS.fullmatch(normalized):
        return None
    if re.search(THANKS_PATTERN, normalized, re.I):
        return localized(
            query,
            "You're welcome! Thank you for your time—have a nice day ahead! 👋",
            "आपका स्वागत है! आपका समय देने के लिए धन्यवाद—आपका दिन शुभ हो! 👋",
            "Aapka welcome! Aapke time ke liye thank you—have a nice day ahead! 👋",
        )
    if re.search(r"\bhow are(?: you)?\b", normalized, re.I):
        return localized(
            query,
            "I'm doing well and ready to help you find fashion products. What are you looking for?",
            "मैं बढ़िया हूँ और फैशन प्रोडक्ट ढूँढने में आपकी मदद के लिए तैयार हूँ। आप क्या ढूँढ रहे हैं?",
            "Main badhiya hoon aur fashion products dhoondhne mein help ke liye ready hoon. Aap kya dhoondh rahe hain?",
        )
    if re.search(r"\b(?:what can you do|who are you)\b", normalized, re.I):
        return localized(
            query,
            "Hi! I'm your friendly fashion assistant. I can find Myntra products and share their prices, ratings, sizes, and links.",
            "नमस्ते! मैं आपका फैशन असिस्टेंट हूँ। मैं Myntra के प्रोडक्ट, कीमत, रेटिंग, साइज़ और लिंक ढूँढ सकता हूँ।",
            "Namaste! Main aapka fashion assistant hoon. Main Myntra products, prices, ratings, sizes aur links dhoondh sakta hoon.",
        )
    return localized(
        query,
        "Hello! 👋 How can I help you find something today?",
        "नमस्ते! 👋 आज मैं आपके लिए क्या ढूँढूँ?",
        "Namaste! 👋 Aaj main aapke liye kya dhoondhun?",
    )


def localized(query, english, hindi, hinglish):
    # Always use English for assistant replies, regardless of the user's input language.
    return english


def no_match_reply(query, suggestions, missing_brand=None):
    if missing_brand:
        return (f"I couldn’t find **{missing_brand}** in this catalogue, but I can suggest these similar items:"
                if suggestions else
                f"I couldn’t find **{missing_brand}** in this catalogue. Try another brand or product type and I’ll look again.")
    return ("I couldn’t find an exact match for that request in this catalogue, but I can suggest these similar items:"
            if suggestions else
            "I couldn’t find a close match in this catalogue. Try another brand, product type, colour, or budget and I’ll look again.")


def response_text(result, answer, query):
    if answer is not None and answer.source == "llm":
        return answer.text
    if result.method == "follow_up":
        return localized(
            query,
            "Here are the products we were discussing. I can compare their prices, ratings, materials, and other listed details.",
            "ये वही प्रोडक्ट हैं जिनकी हम बात कर रहे थे। मैं इनकी कीमत, रेटिंग और उपलब्ध जानकारी की तुलना कर सकता हूँ।",
            "Ye wahi products hain jinki hum baat kar rahe the. Main inki prices, ratings aur available details compare kar sakta hoon.",
        )
    if result.method == "exact_id":
        return localized(query, "Here are the product details from the catalogue.",
                         "कैटलॉग में इस प्रोडक्ट की जानकारी यहाँ है।",
                         "Catalogue mein is product ki details yahan hain.")
    count = result.total_matches or len(result.products)
    return localized(
        query,
        f"I found {count} matching items. Here are some options from the catalogue.",
        f"आपकी रिक्वेस्ट से मेल खाते {count} प्रोडक्ट मिले। कुछ विकल्प नीचे हैं।",
        f"Aapki request se match karte {count} products mile. Kuch options neeche hain.",
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

top_left, top_right = st.columns([7, 1])
with top_right:
    if st.button("Reset chat", width="stretch"):
        memory.clear(st.session_state["conversation_id"])
        st.session_state["messages"] = []
        st.rerun()

st.title("👋 Hi, I’m your friendly fashion assistant")
st.caption("Tell me what you’re looking for. I can find products and share the details available in the Myntra catalogue.")
if engine.llm.available:
    st.caption(f"LLM configured · {engine.config.llm_provider.title()} / {engine.config.llm_model}")
else:
    if engine.config.llm_provider == "gemini":
        st.caption("Catalogue search is ready · Gemini key missing. Set GEMINI_API_KEY in this terminal, then restart Streamlit.")
    else:
        st.caption(f"Catalogue search is ready · {engine.config.llm_provider.title()} API key missing. Set its API key in this terminal, then restart Streamlit.")

example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    if col.button(example, key=f"example_{example}", width="stretch"):
        st.session_state["pending"] = example

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
        smalltalk = greeting_reply(query)
        suggestions = []
        answer = None
        if smalltalk:
            result = RetrievalResult(query=query, method="smalltalk", message=smalltalk)
            user_facing = smalltalk
            displayed_products = []
        elif is_general_brand_question(engine, query):
            result = RetrievalResult(query=query, method="brand_guidance")
            user_facing = localized(
                query,
                "There isn’t one best brand for everyone, and this catalogue doesn’t rank brands overall. Tell me what you’re shopping for, your budget, and what matters most (such as style, comfort, or ratings), and I’ll suggest and compare options available here.",
                "हर किसी के लिए एक ही सबसे अच्छा ब्रांड नहीं होता, और इस कैटलॉग में ब्रांड्स की कुल रैंकिंग उपलब्ध नहीं है। आप क्या खरीदना चाहते हैं, बजट और अपनी प्राथमिकता (जैसे स्टाइल, आराम या रेटिंग) बताइए—मैं यहाँ उपलब्ध विकल्प सुझाकर उनकी तुलना करूँगा।",
                "Har kisi ke liye ek hi best brand nahi hota, aur is catalogue mein brands ki overall ranking nahi hai. Aap kya lena chahte ho, budget kya hai, aur priority (style, comfort ya rating) kya hai—batao, main available options suggest karke compare kar dunga.",
            )
            displayed_products = []
        else:
            with st.spinner("Finding the best matches…"):
                earlier_messages = st.session_state["messages"][:-1]
                budget_context = filter_previous_request_by_budget(
                    engine, previous_user_query(earlier_messages), query
                )
                if budget_context:
                    result, suggestions, amount = budget_context
                    if result.products:
                        user_facing = localized(
                            query,
                            f"I found {result.total_matches} options from your previous request priced at or below ₹{amount:,.0f}.",
                            f"आपकी पिछली रिक्वेस्ट के लिए ₹{amount:,.0f} तक के {result.total_matches} विकल्प मिले।",
                            f"Aapki pehli request ke liye ₹{amount:,.0f} tak ke {result.total_matches} options mile.",
                        )
                        displayed_products = result.products
                    else:
                        user_facing = localized(
                            query,
                            f"I couldn’t find any products from your previous request within ₹{amount:,.0f}. These are the closest-priced options of the same type; they may be over budget.",
                            f"आपकी पिछली रिक्वेस्ट के लिए ₹{amount:,.0f} तक का प्रोडक्ट नहीं मिला। ये उसी प्रकार के सबसे कम कीमत वाले विकल्प हैं, जो बजट से ऊपर हो सकते हैं।",
                            f"Aapki pehli request ke liye ₹{amount:,.0f} ke andar item nahi mila. Ye usi type ke sabse kam daam wale options hain, par budget se zyada ho sakte hain.",
                        )
                        displayed_products = []
                else:
                    result = engine.retriever.search(query)
                    previous = previous_products(earlier_messages)
                    translated_query = translate_query(query)
                    if previous and (CONTEXTUAL_FOLLOW_UP.search(query)
                                     or CONTEXTUAL_FOLLOW_UP.search(translated_query)):
                        result = RetrievalResult(query=query, method="follow_up", products=previous,
                                                 total_matches=len(previous),
                                                 notes=["Using products from your previous query for this comparison."])
                    missing_brand = requested_missing_brand(engine, translated_query)
                    if missing_brand:
                        suggestions = result.products[:3] if result.products else find_related_products(engine, result, query)
                        user_facing = no_match_reply(query, suggestions, missing_brand)
                        displayed_products = []
                    elif result.products and (result.method in {"follow_up", "brand_comparison"}
                                              or NEEDS_LLM_SYNTHESIS.search(translated_query)):
                        answer = engine.llm.generate(query, result, history=earlier_messages) if result.products else None
                    else:
                        answer = None
                    if missing_brand:
                        pass
                    elif result.products:
                        user_facing = response_text(result, answer, query)
                        displayed_products = result.products
                    else:
                        suggestions = find_related_products(engine, result, query)
                        user_facing = no_match_reply(query, suggestions)
                        displayed_products = []

        st.markdown(user_facing)
        if displayed_products:
            show_cards(displayed_products, engine, f"new_{len(st.session_state['messages'])}")
        if suggestions:
            show_cards(suggestions, engine, f"suggestions_{len(st.session_state['messages'])}", "Similar items you could try")

    payload = {"method": result.method, "filters": result.filters, "total_matches": result.total_matches,
               "products": displayed_products, "suggestions": suggestions,
               "unsupported": result.unsupported, "notes": result.notes, "distances": result.distances}
    assistant_message = {"role": "assistant", "content": user_facing, "result": payload}
    st.session_state["messages"].append(assistant_message)
    memory.save_message(st.session_state["conversation_id"], assistant_message)
