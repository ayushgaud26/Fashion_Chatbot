from src.response_formatter import compose


def test_nonexistent_combination_is_no_match(retriever, records):
    combos = {(r["baseColour"], r["articleType"]) for r in records}
    assert ("Purple", "Formal Shoes") not in combos
    res = retriever.search("Show purple formal shoes")
    assert res.method == "no_match" and not res.products
    out = compose(res)
    assert out.source == "system" and "No products" in out.markdown


def test_empty_and_blank_queries(retriever):
    for q in ("", "   ", None):
        res = retriever.search(q)
        assert res.method == "empty_query" and not res.products


def test_price_and_stock_not_fabricated(retriever, records):
    rec = records[0]
    res = retriever.search(f"What is the price of {rec['productDisplayName']}?")
    assert "price" in res.unsupported and res.ids
    md = compose(res).markdown
    assert "price" in md.lower() and "$" not in md and "₹" not in md
    res2 = retriever.search("is it in stock?")
    assert res2.method == "clarification" and not res2.products


def test_gibberish_is_no_match_or_clarification(retriever):
    res = retriever.search("zzqx wvvk")
    assert not res.products
