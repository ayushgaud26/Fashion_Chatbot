"""Structured query understanding: map natural language onto *actual dataset fields*.

Filters are applied to the structured records (never delegated to the LLM). Vocabulary is built from the
data itself; the small synonym table below is documented and only keeps values that exist in the dataset."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from .preprocessing import normalize_text, stem, stem_phrase

FILTER_FIELDS = ("gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage", "year", "brand", "material", "size")
# If one phrase exists in several fields (e.g. "Accessories", "Watches"), the broader/earlier field wins.
FIELD_PRIORITY = ("gender", "brand", "usage", "season", "masterCategory", "subCategory", "articleType",
                  "baseColour", "material", "size")

# phrase -> [(field, [values])]. Values not present in the dataset vocabulary are dropped at build time.
SYNONYMS: dict[str, list[tuple[str, list[str]]]] = {
    "male": [("gender", ["Men"])], "man": [("gender", ["Men"])], "gents": [("gender", ["Men"])],
    "female": [("gender", ["Women"])], "woman": [("gender", ["Women"])], "ladies": [("gender", ["Women"])],
    "lady": [("gender", ["Women"])],
    "sportswear": [("usage", ["Sports"]), ("masterCategory", ["Apparel"])],
    "sports wear": [("usage", ["Sports"]), ("masterCategory", ["Apparel"])],
    "activewear": [("usage", ["Sports"]), ("masterCategory", ["Apparel"])],
    "clothing": [("masterCategory", ["Apparel"])], "clothes": [("masterCategory", ["Apparel"])],
    "sneakers": [("articleType", ["Casual Shoes", "Sports Shoes"])],
    "sneaker": [("articleType", ["Casual Shoes", "Sports Shoes"])],
    "trainers": [("articleType", ["Sports Shoes"])],
    "t shirt": [("articleType", ["Tshirts"])], "tee": [("articleType", ["Tshirts"])],
    "pants": [("articleType", ["Trousers", "Track Pants"])],
    "perfume": [("subCategory", ["Fragrance"])], "perfumes": [("subCategory", ["Fragrance"])],
    "lal": [("baseColour", ["Red"])], "laal": [("baseColour", ["Red"])],
    "gray": [("baseColour", ["Grey"])], "navy": [("baseColour", ["Navy Blue"])],
}

STOPWORDS = set("""show find me i my want a an the for in of to products product items item please some any all list give
get looking look need is are there do does you have with and or on suitable category categories what which can could would
like something whats s that this it be are occasion occasions price pricing cost budget need chahiye dikhao dikhaiye
color colour how kuch koi kuchh please
mujhe mera meri mere ka ki ke ko hai hain ho mein me se aur batao please""".split())

# Normalize common Hindi script and Hinglish terms to the English product vocabulary used by the dataset.
# This is deliberately a small, deterministic phrase dictionary; it does not change catalogue records.
_QUERY_WORDS = {
    # Colours
    "lal": "red", "laal": "red", "लाल": "red",
    "neela": "blue", "nila": "blue", "नीला": "blue", "नीली": "blue", "नीले": "blue",
    "kala": "black", "kaala": "black", "काला": "black", "काली": "black", "काले": "black",
    "safed": "white", "सफेद": "white",
    "hara": "green", "हरा": "green", "हरी": "green",
    "gulabi": "pink", "गुलाबी": "pink",
    "pila": "yellow", "peela": "yellow", "पीला": "yellow", "पीली": "yellow", "पीले": "yellow",
    "bhoora": "brown", "bhura": "brown", "भूरा": "brown",
    "narangi": "orange", "नारंगी": "orange", "baingani": "purple",
    "jamuni": "purple", "जामुनी": "purple",
    "marun": "maroon", "मैरून": "maroon",
    "badami": "beige", "बादामी": "beige",
    "sunehra": "gold", "sunahra": "gold", "सुनहरा": "gold",
    "chaandi": "silver", "chandi": "silver", "चांदी": "silver",
    "malai": "cream", "मलाई": "cream",
    "firozi": "teal", "फिरोज़ी": "teal", "firouzi": "teal",
    "aasmani": "blue", "asmani": "blue", "आसमानी": "blue",
    "nevi": "navy", "navy": "navy",
    "slatee": "grey", "gray": "grey", "grey": "grey", "स्लेटी": "grey",
    # Fashion terms
    "kapda": "clothing", "kapde": "clothing", "कपड़ा": "clothing", "कपड़े": "clothing",
    "rang": "", "रंग": "",
    "joota": "shoes", "jute": "shoes", "जूता": "shoes", "जूते": "shoes", "chappal": "sandals",
    "jeans": "jeans", "जींस": "jeans", "जीन्स": "jeans",
    "shirt": "shirt", "shirts": "shirts", "शर्ट": "shirt",
    "tshirt": "tshirt", "टी": "", "टीशर्ट": "tshirt", "टी-शर्ट": "tshirt",
    "kurta": "kurta", "कुर्ता": "kurta", "kurti": "kurti", "कुर्ती": "kurti",
    "saree": "saree", "sari": "saree", "साड़ी": "saree",
    "lehenga": "lehenga", "लहंगा": "lehenga",
    "bag": "bag", "बैग": "bag",
    "ladies": "women", "mahila": "women", "महिला": "women", "महिलाओं": "women",
    "mahilaon": "women", "ladki": "women", "ladkiyon": "women", "लड़की": "women",
    "purush": "men", "पुरुष": "men", "पुरुषों": "men", "ladka": "men", "ladkon": "men", "लड़कों": "men",
    # Common Hinglish/Hindi request and comparison words
    "mujhe": "", "mujhko": "", "मुझे": "", "चाहिए": "", "दिखाओ": "", "दिखाइए": "", "ढूंढो": "",
    "chahiye": "", "dikhao": "", "dikha": "", "dikhana": "", "dikhaiye": "",
    "dikhado": "", "dhundho": "", "dhundo": "", "dhoondo": "", "khojo": "",
    "batao": "", "dijiye": "", "karo": "", "karna": "", "hai": "are", "hain": "are", "ho": "are",
    "है": "are", "हैं": "are", "aap": "", "ap": "", "liye": "", "liya": "",
    "का": "", "की": "", "के": "", "को": "", "में": "", "और": "and", "क्या": "",
    "ka": "", "ki": "", "ke": "", "ko": "", "mein": "", "me": "", "se": "", "par": "",
    "aur": "and", "ya": "or", "kya": "", "kaunsa": "which", "kaun": "which", "kaunsi": "which",
    "कौन": "which", "कौनसा": "which", "कौनसी": "which", "कैसे": "how", "kaise": "how",
    "kaisa": "how", "kaisi": "how", "behtar": "better", "बेहतर": "better",
    "अच्छा": "better", "अच्छी": "better", "अच्छे": "better",
    "accha": "better", "achha": "better", "achi": "better", "achhi": "better",
    "sabse": "best", "dono": "both", "farq": "difference", "antar": "difference",
    "kimat": "price", "keemat": "price", "daam": "price", "kitna": "price", "kitne": "price",
    "कीमत": "price", "दाम": "price",
    "sasta": "cheap", "sasti": "cheap", "mehenga": "expensive", "mehngi": "expensive",
    "namaste": "hello", "namaskar": "hello", "नमस्ते": "hello", "नमस्कार": "hello",
    "dhanyavaad": "thanks", "shukriya": "thanks", "धन्यवाद": "thanks",
    "ruko": "stop", "rukna": "stop", "bas": "stop", "बस": "stop", "रुको": "stop", "रुकिए": "stop",
    "इनमें": "", "इनमे": "", "इसमें": "", "इसमे": "", "इनका": "them", "इनकी": "them",
    "इसका": "it", "इसकी": "it", "इनको": "them", "इनसे": "them",
    "यह": "it", "ये": "these", "वह": "that", "वे": "those", "इस": "it", "उस": "that", "उन": "them",
    "isme": "", "inme": "", "iska": "it", "iski": "it", "iske": "it", "inki": "them", "inka": "them",
}


def detect_query_language(query: str) -> str:
    """Return hi for Devanagari, hinglish for common Roman Hindi, otherwise en."""
    if re.search(r"[\u0900-\u097f]", query or ""):
        return "hi"
    words = set(re.findall(r"[a-z]+", (query or "").lower()))
    hinglish = {"mujhe", "chahiye", "dikhao", "dikha", "dikhana", "dikhaiye", "batao",
                "kaunsa", "kaunsi", "kaise", "kya", "hai", "hain", "ka", "ki", "ke", "mein", "me",
                "behtar", "achha", "accha", "ladko", "ladkiyon", "kapde", "laal", "lal",
                "neela", "kala", "safed", "keemat", "kimat", "daam", "shukriya", "namaste",
                "dono", "inme", "ladkon", "purush", "bas", "ruko", "dhundho", "dhoondo",
                "kapda", "pila", "peela", "hara", "gulabi", "bhoora", "bhura", "narangi",
                "baingani", "jamuni", "marun", "badami", "sunehra", "sunahra", "chaandi",
                "chandi", "malai", "firozi", "firouzi", "aasmani", "asmani", "nevi", "slatee"}
    return "hinglish" if words.intersection(hinglish) else "en"


def translate_query(query: str) -> str:
    """Map common Hindi/Hinglish words onto the dataset's English search vocabulary."""
    text = unicodedata.normalize("NFKC", str(query or "")).lower()
    # Treat English gender possessives as the gender itself. Without this, "men's"
    # tokenizes to "men s" and the final s can be mistaken for catalogue size S.
    text = re.sub(r"\bmen['’]s\b", "men", text)
    text = re.sub(r"\bwomen['’]s\b", "women", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # Convert common Hindi-script price limits before tokenization loses word order.
    text = re.sub(r"([\d,]+)\s*(?:रुपये|रुपए)?\s*(?:से\s*)?(?:कम|नीचे)", r"under \1", text)
    text = re.sub(r"([\d,]+)\s*(?:रुपये|रुपए)?\s*(?:से\s*)?(?:ज्यादा|अधिक|ऊपर)", r"over \1", text)
    text = re.sub(r"([\d,]+)\s*(?:rupees?|rs\.?)?\s*(?:se\s*)?(?:kam|neeche)\b", r"under \1", text)
    text = re.sub(r"([\d,]+)\s*(?:rupees?|rs\.?)?\s*(?:se\s*)?(?:zyada|adhik|upar)\b", r"over \1", text)
    tokens = re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", text)
    translated = []
    for token in tokens:
        replacement = _QUERY_WORDS.get(token, token)
        if replacement:
            translated.extend(replacement.split())
    return " ".join(translated)

# Attributes the dataset does NOT contain.
UNSUPPORTED = {"discount": "discount", "size": "size",
               "stock": "stock", "inventory": "stock", "availability": "stock",
               "delivery": "shipping", "shipping": "shipping"}
UNSUPPORTED_PHRASES = {("in", "stock"): "stock", ("out", "of", "stock"): "stock"}


@dataclass
class ParsedQuery:
    original: str
    filters: dict = field(default_factory=dict)          # field -> list of allowed values (OR within a field)
    matched_terms: list = field(default_factory=list)    # (phrase, field, values)
    residual_tokens: list = field(default_factory=list)  # meaningful words not mapped to any field
    unsupported: list = field(default_factory=list)      # e.g. ["price"]

    @property
    def has_filters(self) -> bool:
        return bool(self.filters)


class Vocabulary:
    def __init__(self, records: list[dict]):
        self.values: dict[str, list] = {}
        for f in FILTER_FIELDS:
            if f == "size":
                vals = {size for r in records for size in (r.get("sizeValues") or [])}
                vals.update(r.get(f) for r in records if r.get(f) is not None and not r.get("sizeValues"))
                self.values[f] = sorted(vals, key=str)
            elif f == "baseColour":
                # Myntra often stores multiple colours in one field ("Navy Blue, Blue").
                # Index each colour as a searchable value; Catalog.matches still checks
                # against the full, comma-separated source value.
                vals = {colour.strip() for r in records for colour in str(r.get(f) or "").split(",")
                        if colour.strip()}
                self.values[f] = sorted(vals, key=str)
            else:
                self.values[f] = sorted({r[f] for r in records if r.get(f) is not None}, key=str)
        self.years = set(self.values["year"])
        self.table: dict[tuple, list[tuple[str, list]]] = {}
        chosen: dict[tuple, tuple[str, str]] = {}
        for f in FIELD_PRIORITY:  # earlier fields claim a phrase first
            for v in self.values[f]:
                key = stem_phrase(v)
                if key and key not in chosen:
                    chosen[key] = (f, v)
        grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for key, (f, v) in chosen.items():
            grouped[key][f].append(v)
        for phrase, constraints in SYNONYMS.items():
            key = stem_phrase(phrase)
            for f, vals in constraints:
                keep = [v for v in vals if v in self.values.get(f, [])]
                if keep:
                    grouped[key][f] = keep  # synonym overrides raw vocab for that phrase/field
        dynamic = defaultdict(list)
        for alias, field_name, fragment in (
            ("shoes", "articleType", "shoe"), ("sandals", "articleType", "sandal"),
            ("sandals", "articleType", "slipper"), ("bag", "articleType", "bag"),
            ("bag", "subCategory", "bag"), ("cap", "articleType", "cap"),
        ):
            keep = [v for v in self.values.get(field_name, [])
                    if fragment in normalize_text(v)]
            dynamic[(alias, field_name)].extend(v for v in keep if v not in dynamic[(alias, field_name)])
        for (alias, field_name), values in dynamic.items():
            if values:
                grouped[stem_phrase(alias)][field_name] = values
        for term in ("cotton", "polyester", "silk", "leather", "linen", "wool", "denim", "viscose", "rayon", "nylon"):
            keep = [v for v in self.values.get("material", []) if term in normalize_text(v).split()]
            if keep:
                grouped[stem_phrase(term)]["material"] = keep
        self.table = {k: list(d.items()) for k, d in grouped.items()}
        self.max_len = max((len(k) for k in self.table), default=1)


def parse_query(query: str, vocab: Vocabulary) -> ParsedQuery:
    pq = ParsedQuery(original=query)
    translated = translate_query(query)
    tokens = normalize_text(translated).split()
    stems = [stem(t) for t in tokens]
    n, i = len(tokens), 0
    unsupported: list[str] = []
    while i < n:
        matched = False
        for L in range(min(vocab.max_len, n - i), 0, -1):
            key = tuple(stems[i : i + L])
            if key in vocab.table:
                phrase = " ".join(tokens[i : i + L])
                for f, vals in vocab.table[key]:
                    cur = pq.filters.setdefault(f, [])
                    cur.extend(v for v in vals if v not in cur)
                    pq.matched_terms.append((phrase, f, list(vals)))
                i += L
                matched = True
                break
        if matched:
            continue
        # unsupported multi-word phrases, e.g. "how much"
        hit = False
        for phrase, label in UNSUPPORTED_PHRASES.items():
            if tuple(tokens[i : i + len(phrase)]) == phrase:
                if label not in unsupported:
                    unsupported.append(label)
                i += len(phrase)
                hit = True
                break
        if hit:
            continue
        tok = tokens[i]
        if tok.isdigit() and len(tok) == 4 and int(tok) in vocab.years:
            pq.filters.setdefault("year", []).append(int(tok))
            pq.matched_terms.append((tok, "year", [int(tok)]))
        elif stem(tok) in UNSUPPORTED:
            label = UNSUPPORTED[stem(tok)]
            if label not in unsupported:
                unsupported.append(label)
        elif tok not in STOPWORDS and not tok.isdigit():
            pq.residual_tokens.append(tok)
        i += 1
    pq.unsupported = unsupported
    return pq


def matches(record: dict, filters: dict) -> bool:
    def matches_field(field: str, values: list) -> bool:
        if field == "size":
            return bool(set(record.get("sizeValues") or []) & set(values))
        if field == "baseColour":
            # Myntra descriptions can list multiple extracted colours, e.g. "Red, Black".
            present = {normalize_text(value) for value in str(record.get(field) or "").split(",")}
            return any(normalize_text(value) in present for value in values)
        return record.get(field) in values

    return all(matches_field(field, values) for field, values in filters.items())


def build_where(filters: dict) -> dict | None:
    """Translate filters into a ChromaDB metadata `where` clause (same semantics as `matches`)."""
    conds = []
    for f, vals in filters.items():
        if f == "size":
            conds.append({"sizeValues": {"$contains": vals[0]}} if len(vals) == 1 else
                         {"$or": [{"sizeValues": {"$contains": v}} for v in vals]})
        else:
            conds.append({f: vals[0]} if len(vals) == 1 else {f: {"$in": list(vals)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class Catalog:
    """In-memory structured view over the normalised records (source of truth for exact/structured search)."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.vocab = Vocabulary(records)
        self._names: dict[str, list[int]] = defaultdict(list)
        self._name_tokens: dict[int, set] = {}
        for r in records:
            nn = normalize_text(r.get("productDisplayName"))
            if nn:
                self._names[nn].append(r["id"])
            self._name_tokens[r["id"]] = {stem(t) for t in nn.split()}

    def filter(self, filters: dict) -> list[dict]:
        return [r for r in self.records if matches(r, filters)]

    _PREFIX = re.compile(
        r"^(?:please\s+)?(?:show me|show|find me|find|search for|search|tell me about|details of|details for|"
        r"give me|get me|i want|i am looking for|looking for|do you have)\s+(?:the\s+)?"
    )

    def find_exact_name(self, query: str) -> list[int]:
        """Exact product-name match only: whole query (minus a leading command word) equals a name, or the
        query contains a complete name of >= 3 words. Never a partial-word or loose substring match."""
        nq = normalize_text(query)
        if not nq:
            return []
        cand = self._PREFIX.sub("", nq).removesuffix(" please").strip()
        for c in (nq, cand):
            if c in self._names and len(c.split()) >= 2:
                return list(self._names[c])
        padded = f" {nq} "
        best = None
        for name, ids in self._names.items():
            if len(name.split()) >= 3 and f" {name} " in padded:
                if best is None or len(name) > len(best):
                    best = name
        return list(self._names[best]) if best else []

    def keyword_hits(self, tokens: list[str]) -> list[int]:
        """Products whose name contains ALL the given words (plural-insensitive)."""
        want = {stem(t) for t in tokens}
        if not want:
            return []
        return [pid for pid, toks in self._name_tokens.items() if want <= toks]
